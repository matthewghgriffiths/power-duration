from typing import Optional, Callable
from functools import partial, cache, wraps
import gzip

import numpy as np
import jax
from jax import scipy as jsp, numpy as jnp

from power_duration.distributions import NormalGamma

Model = Callable[[jax.Array, jax.Array], jax.Array]


def jac(func: Model):
    return jax.vmap(jax.grad(func), in_axes=(None, 0))


vgrad = jac


def between(x: jax.Array, lo=None, hi=None) -> jax.Array:
    sel = jnp.full_like(x, True, bool)
    if lo is not None:
        sel &= x > lo
    if hi is not None:
        sel &= x < hi

    return sel


Weight = Callable[[jax.Array, jax.Array, jax.Array], jax.Array]


def weight(y: jax.Array, pred: jax.Array, cov: jax.Array) -> jax.Array:
    z = (y - pred) / cov.diagonal()**0.5
    return (1 + z.clip(None, 0)**2)**-1


@partial(jax.jit, static_argnames=['func', 'jac'])
def log_evidence(
    p, y, t, *args, prior: NormalGamma, func: Model | None = None,
    limits=(0, None), jac=None, weight=1, f=None, X=None, sign=1
) -> tuple[jax.Array, NormalGamma]:
    prior_mean, prior_prec, a, b = prior

    if f is None:
        f = func(p, t, *args)
    if X is None:
        if jac is None:
            X = vgrad(func)(p, t, *args)
        elif callable(jac):
            X = jac(p, t, *args)

    weight = weight * between(t, *limits)
    y = jnp.where(jnp.isfinite(y), y, 0)

    XTw = X.T * weight
    XTX = XTw @ X
    post_prec = XTX + prior_prec

    diff_y = jnp.where(weight > 0, y - f, 0)
    diff_b = p - prior_mean

    an = a + weight.sum() / 2
    bn = b + (diff_y.dot(diff_y * weight) + diff_b.dot(prior_prec @ diff_b))
    post = NormalGamma(p, post_prec, an, bn)
    logZ = post.log_evidence(prior) * sign
    return logZ, post


@partial(jax.jit, static_argnames=['func', 'jac', 'weight', 'n_iter'])
def weighted_log_evidence(
    p, y, *args,
    prior: NormalGamma,
    func: Model,
    weight: Weight = weight,
    jac: Optional[Callable] = None,
    sign: int = 1,
    n_iter: int = 3,
    limits=(None, None)
):
    if jac is None:
        jac = vgrad(func)

    f = func(p, *args)
    J = jac(p, *args)
    w = jnp.ones_like(f)
    for _ in range(n_iter):
        logZ, posterior = log_evidence(
            p, y, *args,
            prior=prior, f=f, X=J, weight=w,
        )
        obspost = posterior.posterior(func, *args, jac=jac)
        w = weight(y, *obspost) * between(*args, *limits)

    return logZ * sign, posterior


def _backtrack(
    y, p, t, args, w,
    prior,
    func: Model,
    jac: Model,
    logZ,
    lam,
    factor,
    max_iter,
    f=None,
    J=None,
):
    f = func(p, t, *args) if f is None else f
    J = jac(p, t, *args) if J is None else J
    yp = y + J @ p - f
    jj = jax.vmap(jnp.dot, in_axes=1)(J, J)

    def body(state):
        j, *_, lam = state
        priorlam = prior * prior._replace(mean=p, prec=jnp.diag(jj * lam))
        postnew = priorlam.condition(J, yp, w)
        logZnew, postnew = log_evidence(
            postnew.mean, y, t, *args, prior=prior, func=func, jac=jac, weight=w)

        better = logZnew > logZ
        # Increase dampling if no increase
        # Decrease damping if success on first iteration
        # Maintain dampling on later success
        lamnew = jnp.select(
            jnp.r_[~ better, j == 0, True],
            jnp.r_[lam * factor, lam / factor, lam]
        )
        return (j + 1, better, logZnew, postnew, lamnew)

    def cond(state):
        j, better, *_ = state
        return ~ better & (j < max_iter)

    init = body((0, True, logZ, prior, lam))
    return jax.lax.while_loop(cond, body, init)


@partial(jax.jit, static_argnames=['func', 'weight', 'jac', 'n_iter', 'init'])
def _fit_lm(
    y: jax.Array, p: jax.Array, t: jax.Array, *args,
    prior: NormalGamma,
    func: Model,
    jac: Optional[Callable] = None,
    n_iter: int = 30,
    limits=(None, None),
    lam: float = 1e-4,
    factor: float = 3.,
    tol: float = 1e-2,
    n_backtrack: int = 10,
    in_axes: tuple | None = None,
    **kws
):
    if jac is None:
        in_axes = in_axes or (None, 0) + tuple(0 for _ in args)
        jac = jax.vmap(jax.grad(func), in_axes=in_axes)

    w = jnp.isfinite(t) & jnp.isfinite(y) & between(t, *limits)
    t = jnp.where(w, t, 1)
    y = jnp.where(w, y, 0)
    logZ, post = log_evidence(
        p, y, t, *args, prior=prior, func=func, jac=jac, weight=w)

    def body(state):
        i, logZ, post, old, lam = state
        (j, worse, logZnew, postnew, lam) = _backtrack(
            y, post.mean, t, args, w,
            prior,
            func,
            jac,
            logZ,
            lam,
            factor,
            n_backtrack
        )
        new_state = i + 1, logZnew, postnew, post, lam
        return new_state

    def cond(state):
        i, logZ, post, old, lam = state
        return (post.kl(old) > tol) & (i < n_iter)

    init = body((0, logZ, post, prior, lam))
    i, logZ, post, prior, lam = jax.lax.while_loop(cond, body, init)
    
    return post


@partial(jax.jit, static_argnames=['func', 'weight', 'jac', 'n_iter', 'init'])
def _fit_lm_weighted(
    y: jax.Array, p: jax.Array, t: jax.Array, *args,
    prior: NormalGamma,
    func: Model,
    jac: Optional[Callable] = None,
    weight: Weight = weight,
    n_iter: int = 30,
    limits=(None, None),
    lam: float = 1e-4,
    factor: float = 3.,
    tol: float = 1e-2,
    n_backtrack: int = 10,
    in_axes: tuple | None = None,
    **kws
):
    if jac is None:
        in_axes = in_axes or (None, 0) + tuple(0 for _ in args)
        jac = jax.vmap(jax.grad(func), in_axes=in_axes)

    w = w0 = jnp.isfinite(t) & jnp.isfinite(y) & between(t, *limits)
    t = jnp.where(w, t, 1)
    y = jnp.where(w, y, 0)
    logZ, post = log_evidence(p, y, t, *args, prior=prior, func=func, weight=w)

    def body(state):
        i, logZ, post, old, lam = state
        obspost, J = post.post_jac(func, t, *args, jac=jac)
        w = weight(y, *obspost) * w0
        (j, worse, logZnew, postnew, lam) = _backtrack(
            y, post.mean, t, args, w,
            prior,
            func,
            jac,
            logZ,
            lam,
            factor,
            n_backtrack,
            f=obspost.mean, J=J
        )
        new_state = i + 1, logZnew, postnew, post, lam
        return new_state

    def cond(state):
        i, logZ, post, old, lam = state
        return (post.kl(old) > tol) & (i < n_iter)

    init = body((0, logZ, post, prior, lam))
    i, logZ, post, prior, lam = jax.lax.while_loop(cond, body, init)
    return post


def _fit_posteriors(
    y: jax.Array, p: jax.Array, t: jax.Array, *args,
    prior: NormalGamma,
    func: Model,
    jac: Optional[Callable] = None,
    n_iter: int = 5,
    limits=(None, None),
    **kws
):
    posteriors = []
    if jac is None:
        jac = jax.vmap(jax.grad(func), in_axes=(None, 0))

    w = jnp.isfinite(t) & jnp.isfinite(y) & between(t, *limits)
    t = jnp.where(w, t, 1)
    y = jnp.where(w, y, 0)
    for _ in range(n_iter):
        f = func(p, t)
        J = jac(p, t)
        posterior = prior.condition(J, y + J @ p - f, w)
        p = posterior.mean
        posteriors.append(posterior)

    return posteriors


@partial(jax.jit, static_argnames=['func', 'weight', 'jac', 'n_iter', 'init'])
def _fit_posterior(y, p, t, prior, func, **kws):
    return _fit_posteriors(y, p, t, prior, func, **kws)[-1]


@partial(jax.jit, static_argnames=['func', 'weight', 'jac', 'n_iter', 'init'])
def _fit_weighted_posterior(
    y: jax.Array, p: jax.Array, t: jax.Array, *args,
    prior: NormalGamma,
    func: Model,
    jac: Optional[Callable] = None,
    weight: Weight = weight,
    n_iter: int = 5,
    limits=(None, None),
    **kws
):

    if jac is None:
        jac = vgrad(func)

    f = func(p, t)
    J = jac(p, t)
    w = between(t, *limits).astype(f.dtype)
    posterior = prior.condition(J, y + J @ p - f, w)

    for _ in range(n_iter):
        p = posterior.mean
        obspost, J = posterior.post_jac(func, t, jac=jac)
        w = weight(y, *obspost) * between(t, *limits)
        f = obspost.mean
        posterior = prior.condition(J, y + J @ p - f, w)

    # def body(i, carry):
    #     p, f, J, w, prev = carry
    #     posterior = prior.condition(J, y + J @ p - f, w)
    #     p = posterior.mean
    #     obspost, J = posterior.post_jac(func, t, jac=jac)
    #     w = weight(y, *obspost) * between(t, *limits)
    #     f = obspost.mean
    #     return p, f, J, w, posterior

    # *_, posterior = jax.lax.fori_loop(0, n_iter, body, (p, f, J, w, prior))
    return posterior


@cache
def _get_opt(func, method='L-BFGS-B', **kws):
    from jaxopt import ScipyMinimize
    return ScipyMinimize(
        method,
        fun=partial(log_evidence, func=func, sign=-1),
        has_aux=True,
        **kws
    )


def _opt_posterior(
    y: jax.Array, p: jax.Array, t: jax.Array,
    prior: NormalGamma,
    func: Model,
    jac: Optional[Callable] = None,
    limits=(None, None), opt_kws=None, **kws
):
    opt = _get_opt(func, **(opt_kws or {}))
    res = opt.run(p, y, t, prior=prior, limits=limits)
    logZ, post = log_evidence(
        res.params, y, t, func=func, prior=prior, limits=limits)
    return post


@cache
def _get_opt_weighted(func, weight=weight, n_iter=3, method='L-BFGS-B', **kws):
    from jaxopt import ScipyMinimize
    return ScipyMinimize(
        method,
        fun=partial(weighted_log_evidence, func=func,
                    weight=weight, n_iter=n_iter, sign=-1),
        has_aux=True,
        **kws
    )


def _opt_weighted_posterior(
    y: jax.Array, p: jax.Array, t: jax.Array,
    prior: NormalGamma,
    func: Model,
    jac: Optional[Callable] = None,
    weight: Weight = weight, n_iter=3,
    limits=(None, None),
    opt_kws=None, **kws
):
    opt = _get_opt_weighted(
        func, weight=weight, n_iter=n_iter, **(opt_kws or {}))
    res = opt.run(p, y, t, prior=prior, limits=limits)
    logZ, post = weighted_log_evidence(
        res.params, y, t,
        func=func, weight=weight, prior=prior, limits=limits, n_iter=n_iter
    )
    return post
