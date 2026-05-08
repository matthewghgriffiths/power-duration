
from functools import partial
from typing import NamedTuple, Callable
from dataclasses import dataclass, field

import jax
from jax import scipy as jsp, numpy as jnp

import diffrax
import optimistix as optx

from tqdm.auto import tqdm


@jax.tree_util.register_dataclass
@dataclass
class OptStatus:
    history: list | None = field(default_factory=list, metadata=dict(static=True))
    loss_history: list | None = field(default_factory=list, metadata=dict(static=True))
    progress: tqdm | None = field(default=None, metadata=dict(static=True))

    def update(self, **kwargs):
        _, loss = kwargs['loss_this_step']
        _, y = kwargs['y']

        loss = float(loss)
        self.progress.update(1)
        self.progress.set_postfix(loss=loss)

        self.loss_history.append(loss)
        self.history.append(y)

    def print(self, **kwargs):
        _, loss = kwargs['loss_this_step']
        jax.debug.print("loss = {:.2f}", loss)

    def __call__(self, **kwargs):
        if self.progress:
            jax.debug.callback(self.update, **kwargs)
        else:
            self.print(**kwargs)

    def tqdm(self, *args, **kwargs):
        self.progress = progress = tqdm(*args, **kwargs)
        return progress

    def reset(self):
        self.progress = None
        self.loss_history = []
        self.history = []

    def minimise(
        self,
        fn,
        solver,
        y0,
        args=None,
        options=None,
        *,
        max_steps=256,
        **kwargs
    ):
        self.reset()
        with self.tqdm(total=max_steps):
            return optx.minimise(
                fn, solver, y0, args, options,
                max_steps=max_steps, **kwargs
            )


def _gated_fun(t, y, args):
    P, params = args
    Pmax, P2, M, t0, t1, t2, g, tG, a, tA = params
    G, A, R0, R1, *R2s = y

    # If R2 isn't passed then modelling quasi-steady state
    R2, = R2s if R2s else (1,)

    P = jnp.clip(P, 0, R0 * Pmax) + M  # Physiological power
    LP = R1 * G                        # Anerobic/Glycolytic power
    AP = R2 * A                        # Aerobic power

    dR0 = - (P - LP) / Pmax / t0       # Phosphagen depletion
    dR1 = - (LP - AP) / Pmax / t1      # Glycolytic depletion
    dR2 = - (AP - M) / P2 / t2         # Fatigue/glycogen exhaustion

    dG = (  # Glycolytic gate/activation
        g * P * (1 - R0)  # glycolytic pathway activated by ATP use
        * (1 - G / Pmax)  # sigmoidal activation
        - G               # negative feedback
    ) / tG
    dA = (  # Aerobic gate/activation
        a * G * (1 - R1) # aerobic pathway activated by glycolytic pathway
        * (1 - A / P2)   # sigmoidal activation
        - A              # negative feedback
    ) / tA

    if R2s:  # Modelling fatigue/glycogen exhaustion
        return jnp.r_[dG, dA, dR0, dR1, dR2].reshape(y.shape)
    else:    # Modelling quasi-steady state
        return jnp.r_[dG, dA, dR0, dR1].reshape(y.shape)


def _gated_condition(t, y, args, **kwargs):
    P, (Pmax, *_) = args
    R0 = y[2]
    return R0 * Pmax - P


def _gated_init(
        params, solver: optx.AbstractRootFinder = optx.Newton(1e-5, 1e-5)):
    M = params[2]
    sol = optx.root_find(
        partial(_gated_fun, 0),
        y0=jnp.r_[M, M, 1, 1],
        args=(0, params),
        solver=solver,
    )
    return jnp.r_[sol.value, 1.]


def interval_power(t, y, args):
    (ts, Ps), _ = args
    i = jnp.searchsorted(ts, t).clip(0, Ps.size - 1)
    return Ps[i]


def interp_power(t, y, args, **kwargs):
    (ts, Ps), _ = args
    return jnp.interp(t, ts, Ps, **kwargs)


def interp_effort(t, y, args, **kwargs):
    (ts, Fs), params = args
    Pmax = params[0]
    R0 = y[..., 2]
    F = jnp.interp(t, ts, Fs, **kwargs).clip(0, 1) # between 0 and 1
    P = F * R0 * Pmax
    return P


def wrap_fun(func, power_fun, **kws):
    def wrapped(t, y, args, **kwargs):
        _, func_params = args
        P = power_fun(t, y, args, **kws)
        return func(t, y, (P, func_params), **kwargs)

    return wrapped


class ODESystem(NamedTuple):
    terms: diffrax.AbstractTerm
    solver: diffrax.AbstractSolver
    event: diffrax.Event | None = None
    stepsize_controller: diffrax.AbstractStepSizeController = diffrax.ConstantStepSize()
    init: Callable | None = None

    def solve(
            self,
            params: jax.Array, P, y0: jax.Array | None = None,
            t0: float = 0., t1: float = 1., dt0: float | None = None,
            **kwargs
    ):
        if y0 is None:
            y0 = self.init(params)

        kwargs.setdefault('event', self.event)
        kwargs.setdefault('stepsize_controller', self.stepsize_controller)

        return diffrax.diffeqsolve(
            self.terms, self.solver,
            t0=t0, t1=t1, dt0=dt0, y0=y0,
            args=(P, params),
            **kwargs
        )

    def time_to_event(
            self,
            params: jax.Array, P: float, y0: jax.Array | None = None,
            t0: float = 0., t1: float = 1., dt0: float | None = None,
            **kwargs
    ):
        kwargs['saveat'] = diffrax.SaveAt(t1=True, steps=False)
        sol = self.solve(params, P, y0, t0, t1, dt0, **kwargs)
        return sol.ts[-1]

    def _time_root(self, P, args):
        params, t, kws = args
        return self.time_to_event(
            params, P, t0=0, t1=t * 2, throw=False, **kws) - t

    def event_time(
        self,
        params: jax.Array, t: float, y0: jax.Array | None = None,
        solver=optx.Bisection(1e-3, 1e-3),
        **kwargs
    ):
        Px = params[0] - params[2]
        res = optx.root_find(
            self._time_root,
            y0=Px, args=(params, t, kwargs),
            solver=solver,
            options=dict(lower=0., upper=Px)
        )
        return res.value

    def times_to_event(self, params, Ps, **kwargs):
        return jax.vmap(
            partial(self.time_to_event, params, **kwargs),
        )(Ps)

    def loss(self, params, args, **kwargs):
        return self.get_loss(**kwargs)(params, args)

    def wrap(self, power_fun=interp_power, **kwargs):
        interp_func = wrap_fun(self.terms.vector_field, power_fun, **kwargs)
        interp_event = wrap_fun(self.event.cond_fn, power_fun, **kwargs)
        return self._replace(
            terms=diffrax.ODETerm(interp_func),
            event=diffrax.Event(
                cond_fn=interp_event,
                direction=self.event.direction,
                root_finder=self.event.root_finder,
            )
        )

    def get_loss(self, **kwargs):
        def times_to_event(params, Ps):
            return jax.vmap(
                partial(self.time_to_event, params, **kwargs),
            )(Ps)

        def loss(params, args):
            ts, Ps, *priors = args
            odets = times_to_event(params, Ps)
            odePs = jnp.interp(ts, odets, Ps, left=Ps[0], right=Ps[-1])
            reg = 0
            if priors:
                t_prior, m_prior = priors
                reg = jnp.square(jnp.log(params / m_prior)).dot(t_prior)

            return (
                jnp.sum(jnp.square(jnp.log(odets / ts)))
                + jnp.sum(jnp.square(jnp.log(odePs / Ps)))
                + reg
            )

        return loss
    
    def pacing(self, logits, ts, params, dt0=0.125, max_steps=40_000):
        """logits is the sigmoid of 'effort' (fraction of max power used)

        this function calculates the resulting power for the interpolated 
        effort over ts
        """
        Fs = jax.nn.sigmoid(logits)
        sol = self.solve(
            params, (ts, Fs),
            t0=ts[0], t1=ts[-1], dt0=dt0,
            max_steps=max_steps,
            saveat=diffrax.SaveAt(ts=ts), 
            adjoint=diffrax.DirectAdjoint(), 
            throw=False,
            event=None,
        )
        ys = sol.ys
        P = interp_effort(ts, ys, ((ts, Fs), params))
        return P
    
    def pace_loss(self, logits, args, link=jnp.cbrt, **kwargs):
        ## Looking to maximise distance (d = \int P^{1/3} dt)
        ## In theory could augment ODE with a distance variable
        ts, params = args
        P = self.pacing(logits, ts, params, **kwargs)
        return - jsp.integrate.trapezoid(link(P), ts)
    
    def optimise_pacing(
            self, 
            params, y0, ts,
            max_steps=200, link=jnp.cbrt, optx=optx, **opt_kws
        ):
        ## Fit optimal pacing strategy
        return optx.minimise(
            partial(self.pace_loss, link=link),
            y0=y0, 
            args=(ts, params), 
            max_steps=max_steps,
            **opt_kws
        )
    
    def optimal_power(self, params, y0, ts, **opt_kws):
        res = self.optimise_pacing(params, y0, ts, **opt_kws)
        return self.pacing(res.value, ts, params)


def gated_system(
    root_solver: optx.AbstractRootFinder = optx.Newton(1e-5, 1e-5),
    solver: diffrax.AbstractSolver = diffrax.Tsit5(),
    stepsize_controller: diffrax.AbstractStepSizeController = diffrax.ConstantStepSize(),
) -> ODESystem:
    return ODESystem(
        terms=diffrax.ODETerm(_gated_fun),
        event=diffrax.Event(_gated_condition, root_solver),
        solver=solver,
        stepsize_controller=stepsize_controller,
        init=jax.jit(partial(_gated_init, solver=root_solver))
    )
