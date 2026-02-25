import json
from typing import NamedTuple
from functools import partial

import numpy as np
import jax
from jax import scipy as jsp, numpy as jnp


class Normal(NamedTuple):
    mean: jax.Array
    cov: jax.Array

    @property
    def variance(self):
        if jnp.ndim(self.cov) == 1:
            return self.cov

        return self.cov.diagonal()

    @property
    def covariance(self):
        if jnp.ndim(self.cov) == 1:
            return jnp.diag(self.cov)

        return self.cov

    @property
    def std(self):
        return jnp.sqrt(self.variance)

    @property
    def uncertainties(self):
        from uncertainties.unumpy import uarray
        return uarray(self.mean, self.std)


def condition(self, X: jax.Array, y: jax.Array, weight=None):
    prior_mean, prior_prec, a, b = self
    if weight is None:
        weight = jnp.ones_like(y)

    XTw = X.T * weight
    XTX = XTw @ X
    XTy = XTw @ y + prior_prec @ prior_mean
    post_prec = XTX + prior_prec
    post_mean = jsp.linalg.solve(
        post_prec, XTy, assume_a='pos')

    diff_y = y - X @ post_mean
    diff_b = post_mean - prior_mean

    an = a + weight.sum() / 2
    bn = b + (
        diff_y.dot(diff_y * weight) + diff_b.dot(prior_prec @ diff_b)
    )
    return type(self)(post_mean, post_prec, an, bn)


def kl(P, Q):
    m1, P1, a1, b1 = P
    m2, P2, a2, b2 = Q
    P2divP1 = jnp.linalg.solve(P1, P2)
    return (
        (a1 / b1 / 2) * ((m2 - m1) @ P2 @ (m2 - m1))
        + P2divP1.diagonal().sum() / 2
        - jnp.linalg.slogdet(P2divP1)[1] / 2
        - m1.size / 2
        + a2 * jnp.log(b1/b2)
        - jax.scipy.special.gammaln(a1)
        + jax.scipy.special.gammaln(a2)
        + (a1 - a2) * jax.scipy.special.digamma(a1)
        - (b1 - b2) * a1 / b1
    )


class NormalGamma(NamedTuple):
    mean: jax.Array
    prec: jax.Array
    a: jax.Array | float = 1/2
    b: jax.Array | float = 1/2

    def to_dict(self):
        ret = {
            k: jnp.asarray(v).tolist()
            for k, v in zip(self._fields, self)
        }
        ret['__cls__'] = type(self).__name__
        return ret

    @property
    def size(self):
        return self.mean.size

    def append(self, *others):
        dists = [self, *others]
        mean = jnp.concat([d.mean for d in dists])
        prec = jnp.block([
            [
                di.prec if i == j
                else jnp.zeros_like(di.prec, shape=(di.size, dj.size))
                for j, dj in enumerate(dists)
            ]
            for i, di in enumerate(dists)
        ])
        return self._replace(
            mean=mean,
            prec=prec,
        )

    @classmethod
    def from_dict(cls, data):
        return cls(**{k: jnp.array(data[k]) for k in cls._fields})

    def __pow__(self, p):
        return self._replace(
            prec=self.prec * p, a=self.a * p, b=self.b * p)

    def __mul__(self, other):
        eta = (n1 + n2 for n1, n2 in zip(self.natural, other.natural))
        return self.from_natural(eta)

    def __truediv__(self, other):
        eta = (n1 - n2 for n1, n2 in zip(self.natural, other.natural))
        return self.from_natural(eta)

    def __matmul__(self, b):
        m = self.mean @ b
        p = jnp.linalg.pinv(
            b.T @ (jnp.linalg.pinv(self.prec) @ b))
        return self._replace(mean=m, prec=p)

    def __getitem__(self, index):
        return self._replace(
            mean=self.mean[index],
            prec=self.prec[np.ix_(index, index)]
        )

    @property
    def natural(self):
        return self.prec @ self.mean, self.prec, self.a, self.b

    @classmethod
    def from_natural(cls, eta):
        eta1, prec, a, b = eta
        mean = jsp.linalg.solve(prec, eta1, assume_a='pos')
        return cls(mean, prec, a, b)

    @classmethod
    def init(cls, mean, cov, sigma=1., v0=1.):
        if jnp.ndim(cov) == 1:
            cov = jnp.diag(cov)

        prec = jnp.linalg.inv(cov) / sigma**2
        a = v0 / 2
        b = v0 * sigma**2 / 2
        return cls(mean, prec, a, b)

    condition = condition
    kl = kl
    # def condition(self, X: jax.Array, y: jax.Array, weight=None):
    #     prior_mean, prior_prec, a, b = self
    #     if weight is None:
    #         weight = jnp.ones_like(y)

    #     XTw = X.T * weight
    #     XTX = XTw @ X
    #     XTy = XTw @ y + prior_prec @ prior_mean
    #     post_prec = XTX + prior_prec
    #     post_mean = jsp.linalg.solve(
    #         post_prec, XTy, assume_a='pos')

    #     diff_y = y - X @ post_mean
    #     diff_b = post_mean - prior_mean

    #     an = a + weight.sum() / 2
    #     bn = b + (
    #         diff_y.dot(diff_y * weight) + diff_b.dot(prior_prec @ diff_b)
    #     )
    #     return type(self)(post_mean, post_prec, an, bn)

    @property
    def variance(self) -> jax.Array:
        return self.b / (self.a - 1)

    @property
    def covariance(self) -> jax.Array:
        return jnp.linalg.inv(self.prec) * self.variance

    @property
    def std(self):
        return jnp.sqrt(self.covariance.diagonal())

    @property
    def uncertainties(self):
        from uncertainties.unumpy import uarray
        return uarray(self.mean, self.std)

    def posterior(self, func=lambda x: x, *args, jac=None, var=False) -> Normal:
        post, _ = self.post_jac(func, *args, jac=jac, var=var)
        return post

    def post_jac(self, func, *args, jac=None, var=False) -> tuple[Normal, jax.Array]:
        return _post_jac(self, func, *args, jac=jac, var=var)

    def log_evidence(self, prior, n=None):
        _, logdet0 = jnp.linalg.slogdet(prior.prec)
        _, logdetn = jnp.linalg.slogdet(self.prec)
        n = n or (self.a - prior.a) * 2
        return (
            n / 2 * jnp.log(2 * jnp.pi)
            + logdet0/2 - logdetn/2
            + prior.a * jnp.log(prior.b)
            - self.a * jnp.log(self.b)
            + jsp.special.gammaln(self.a)
            - jsp.special.gammaln(self.b)
        )

    def log_likelihood(self, p, y, *args, func, jac=None, weight=None, sign=1):

        prior_mean, prior_prec, a, b = self
        if jac is None:
            jac = vgrad(func)

        f = func(p, *args)
        X = jac(p, *args)

        if weight is None:
            weight = jnp.ones_like(y)

        XTw = X.T * weight
        XTX = XTw @ X
        post_prec = XTX + prior_prec

        diff_y = y - f
        diff_b = p - prior_mean

        an = a + weight.sum() / 2
        bn = b + (
            diff_y.dot(diff_y * weight) + diff_b.dot(prior_prec @ diff_b)
        )
        post = type(self)(p, post_prec, an, bn)

        # posterior = self.condition(J, y + J @ p - f)
        logZ = post.log_evidence(self) * sign
        return logZ, post


@partial(jax.jit, static_argnames=['func', 'jac', 'var'])
def _post_jac(self, func, *args, jac=None, var=False) -> tuple[Normal, jax.Array]:
    jac = jac or jax.jacobian(func)

    f0 = func(self.mean, *args)
    J = jac(self.mean, *args)
    if var:
        var0 = ((J @ self.covariance) * J).sum(axis=1)
        return Normal(f0, var0), J

    cov0 = J @ self.covariance @ J.T
    return Normal(f0, cov0), J
