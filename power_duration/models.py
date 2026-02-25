from functools import partial

import numpy as np
import jax
from jax import scipy as jsp, numpy as jnp


from power_duration.distributions import NormalGamma
from power_duration.fits import (
    jac, weight, _fit_weighted_posterior, _fit_posterior,
    _fit_posteriors, _opt_posterior, _opt_weighted_posterior,
    _fit_lm, _fit_lm_weighted, between
)

jax.config.update('jax_enable_x64', True)

Wp_prior = NormalGamma.init(
    jnp.r_[750. * 30], jnp.r_[1e2], 30, 10)
CP_prior = NormalGamma.init(
    jnp.r_[250.], jnp.r_[6.2e-2], 30, 10)
t1_prior = NormalGamma.init(
    jnp.r_[3.], jnp.r_[1e-6], 30, 10)
Pmax_prior = NormalGamma.init(
    jnp.r_[1000.], jnp.r_[1e0], 30, 10)
t2_prior = NormalGamma.init(
    jnp.r_[10.], jnp.r_[1e-6], 30, 10)
a_prior = NormalGamma.init(
    jnp.r_[0.], jnp.r_[1e-7], 30, 10)
A_prior = NormalGamma.init(
    jnp.r_[30.], jnp.r_[1e-4], 30, 10)
ttf_prior = NormalGamma.init(
    jnp.r_[420.], jnp.r_[1e-2], 30, 10)

PD_MODELS = {}


def init_power(t, P, prior, tmin=60):
    s = t > tmin
    t, P = t[s], P[s]
    (logP0, alpha), *info = np.linalg.lstsq(
        np.c_[np.ones_like(t), np.log(t)], np.log(P)
    )
    return jnp.r_[jnp.exp(logP0), alpha]


def power_law(params, t):
    P0, alpha = params
    return P0 * t ** alpha


power_prior = NormalGamma.init(
    jnp.r_[500, -0.2], jnp.r_[3e-1, 1e-6], 30, 10
)
PD_MODELS['power_law'] = power_law_info = dict(
    func=power_law, prior=power_prior, init=init_power,
    limits=(90, None),
    params=['P_0', r'\alpha'],
    name='Power law',
    formula=r"P_0 t^{\alpha}"
)
PD_MODELS['power_law30'] = power_law30_info = dict(
    func=power_law,
    prior=power_prior,
    init=partial(init_power, tmin=30),
    limits=(30, None),
    params=['P_0', r'\alpha'],
    name='Power law',
    formula=r"P_0 t^{\alpha}"
)


def _hill3(params, t):
    P1, t1, a1 = params
    logt = jnp.log(t)
    a1 = jnp.exp(a1)
    return P1 * jax.nn.sigmoid(- a1 * (logt - t1))


def _hill2(params, t):
    P1, t1 = params
    return P1 * jax.nn.sigmoid(t1 - jnp.log(t))


def cp2(params, t):
    CP, Wp = params
    return Wp / t + CP


def cp3(params, t):
    CP, Pmax, t1 = params
    return _hill2((Pmax - CP, t1), t) + CP


def cp4_exact(params, t):
    CP, Pmax, logt1, logt2 = params
    t1 = jnp.exp(logt1)
    t2 = jnp.exp(logt2)
    # t1 = t1 * Pmax / (Pmax - CP)
    expt = jnp.exp(- CP * t / Pmax / t1)
    Pw = (Pmax - CP) * expt / (Pmax / CP + expt * (1 - Pmax / CP))
    Pcp = CP / (1 + t / t2)
    return Pw + Pcp


def cp4(params, t):
    CP, Pmax, logt1, logt2 = params

    aP = _hill2((Pmax - CP, logt1), t)
    cP = _hill2((CP, logt2), t)

    return aP + cP


def cp4_exp(params, t):
    CP, Pmax, logt1, logt2 = params
    t1 = jnp.exp(logt1)
    Wp = (Pmax - CP) * t1
    return _hill2((CP, logt2), t) - Wp / t * jnp.expm1(-t / t1)


def cp6(params, t):
    CP, Pmax, logt1, logt2, a1, a2 = params
    aP = _hill3((Pmax - CP, logt1, a1), t)
    cP = _hill3((CP, logt2, a2), t)
    return aP + cP


cp2_prior = NormalGamma.append(CP_prior, Wp_prior)
cp3_prior = NormalGamma.append(CP_prior, Pmax_prior, t1_prior)
cp4_prior = NormalGamma.append(cp3_prior, t2_prior)
cp6_prior = NormalGamma.append(cp4_prior, a_prior, a_prior)


def init_cp4(t, P, prior):
    CP, Pmax, lt1, lt2 = prior.mean
    t1, t2 = jnp.exp(jnp.r_[lt1, lt2])
    (P1, P2), *info = np.linalg.lstsq(
        np.c_[1 / (1 + t / t1), 1 / (1 + t / t2)], P)
    i = P.argsort()
    t, P = t[i], P[i]
    lt1 = np.log(np.interp(P2 + P1 / 2, P, t))
    p = jnp.r_[P2, P1 + P2, lt1, lt2]
    return p


def init_cp6(t, P, prior):
    i = np.r_[0, 1, 2, 3]
    prior4 = prior[i]
    p4 = init_cp4(t, P, prior4)
    p = jnp.zeros(6).at[i].set(p4)
    return p


PD_MODELS['CP2'] = cp2_info = dict(
    func=cp2, prior=cp2_prior, limits=(100, None),
    params=['CP', "W'"],
    name='2-CP',
    formula=r"\frac{W'}{t} + CP"
)
PD_MODELS['CP3'] = cp3_info = dict(
    func=cp3, prior=cp3_prior,
    params=['CP', r"P_{\max}", 't_1'],
    name='3-CP',
    formula=r"\frac{P_{\max} - CP}{1 + \frac{t}{t_1}} + CP"
)
PD_MODELS['CP4'] = cp4_info = dict(
    func=cp4, prior=cp4_prior, init=init_cp4,
    params=['CP', r"P_{\max}", 't_1', 't_2'],
    name='4-CP',
    formula=r"\frac{P_{\max} - CP}{1 + \frac{t}{t_1}} + \frac{CP}{1 + \frac{t}{t_2}}"
)
PD_MODELS['CP4_exact'] = cp4_exact_info = dict(
    func=cp4_exact, prior=cp4_prior, init=init_cp4,
    params=['CP', r"P_{\max}", 't_1', 't_2'],
    name='4-CP ode',
    formula=(
        r"\frac{P_{\max} - CP}{\left(1 + \frac{LP}{CP}\right) e^{\frac{CP t}{LP t_1}} - \frac{LP}{CP}}"
        r" + \frac{CP}{1 + \frac{t}{t_2}}"
    )
)
PD_MODELS['CP4_exp'] = cp4_exact_info = dict(
    func=cp4_exp, prior=cp4_prior, init=init_cp4,
    params=['CP', r"P_{\max}", 't_1', 't_2'],
    name='4-CP exp',
    formula=(
        r"\left(P_{\max} - CP\right)\frac{t_1}{t} \left(1 - e^{-\frac{t}{t_1}}\right)"
        "$\n$"
        r"\frac{CP}{1 + \frac{t}{t_2}}"
    )
)
cp6_info = dict(
    func=cp6, prior=cp6_prior, init=init_cp6,
    params=['CP', r"P_{\max}", 't_1', 't_2', 'a_1', 'a_2'],
    name='6-CP',
    formula=(
        r"\frac{LP}{1 + \left(\frac{t}{t_1}\right)^{a_1}}"
        r" + \frac{CP}{1 + \left(\frac{t}{t_2}\right)^{a_2}}"
    )
)


def pt_metabolic(params, t, tmap=420, k1=30, k2=20, f=-0.233, BMR=20):
    MAP, Pmax, t1, A = params
    k1 = jnp.exp(t1)
    Wp = Pmax * k1

    log_term = jnp.log((t / tmap).clip(1, None)) * (t > tmap)
    exp1 = (1.0 + (k1 / t) * jnp.expm1(-t / k1))
    X0 = exp1
    X1 = - (1 + f * log_term) * jnp.expm1(-t/k2) / t
    X2 = exp1 * log_term

    anaerobic = Wp * X1
    aerobic = (MAP - BMR) * X0 - A * X2
    return anaerobic + aerobic


def init_pt_metabolic(t, P, prior, tmap=420, k1=30, k2=20, f=-0.233, BMR=20):
    MAP, Pmax, A, t1 = prior.mean
    k1 = jnp.exp(t1)

    log_term = jnp.log((t / tmap).clip(1, None)) * (t > tmap)
    X = np.c_[
        (1.0 + (k1 / t) * jnp.expm1(-t / k1)),
        - (1 + f * log_term) * jnp.expm1(-t/k2) / t,
        - (1.0 + (k1 / t) * jnp.expm1(-t / k1)) * log_term
    ]
    coef, *info = np.linalg.lstsq(X, P - BMR)
    coef[0] += BMR
    coef[1] /= k1
    return np.r_[coef[:2], t1, coef[-1]]


ptmeta_prior = NormalGamma.append(
    CP_prior, Pmax_prior, A_prior, t1_prior
)
PD_MODELS['PT'] = pt_info = dict(
    func=pt_metabolic, prior=ptmeta_prior, init=init_pt_metabolic,
    params=['CP', r"P_{\max}", 't_1', 'A'],
    name='Peronnet and Thibault',
    formula=r"P_{PT}(t)"
)
PD_MODELS['CP6'] = cp6_info


def ompd(params, t):
    CP, Pmax, t1, A, ttf = params
    t1 = jnp.exp(t1)
    Wp = (Pmax - CP) * t1
    return CP - Wp / t * jnp.expm1(-t / t1) - A * jnp.log((t / ttf).clip(1, None))


def ompd1p(params, t):
    CP, Pmax, t1, A, ttf = params
    t1 = jnp.exp(t1)
    Wp = (Pmax - CP) * t1
    return CP - Wp / t * jnp.expm1(-t / t1) - A * jnp.log1p((t / ttf))


def om3cp(params, t):
    CP, Pmax, t1, A, ttf = params
    t1 = jnp.exp(t1)
    Wp = (Pmax - CP) * t1
    return Wp / (t + t1) + CP - A * jnp.log((t / ttf).clip(1, None))


def om3cp1p(params, t):
    CP, Pmax, t1, A, ttf = params
    t1 = jnp.exp(t1)
    Wp = (Pmax - CP) * t1
    return Wp / (t + t1) + CP - A * jnp.log1p((t / ttf))


def omexp(params, t):
    CP, Pmax, t1, A, ttf = params
    t1 = jnp.exp(t1)
    return (Pmax - CP) * jnp.exp(- t / t1 / jnp.e) + CP - A * jnp.log((t / ttf).clip(1, None))


def init_ompd(t, P, prior):
    CP, Pmax, t1, A, ttf = prior.mean
    t1 = jnp.exp(t1)
    Wp = Pmax * t1
    (Wp, CP, A), *info = np.linalg.lstsq(
        np.c_[1 / (t1 + t), np.ones_like(t), - np.log1p(t / ttf)], P)
    Pmax = Wp / t1 + CP
    return jnp.r_[CP, Pmax, t1, A, ttf]


ompd_prior = NormalGamma.append(
    CP_prior, Pmax_prior, t1_prior, A_prior, ttf_prior)
ompd_params = ["CP", r'P_{\max}', 't_1', "A", r't_{ttf}']
PD_MODELS['OmPD'] = OmPD_info = dict(
    func=ompd, prior=ompd_prior, init=init_ompd, params=ompd_params,
    name='OmPD',
    formula=(
        r"\left(P_{\max} - CP\right)\frac{t_1 \left(1 - e^{\frac{t}{t_1}}\right) }{t}"
        "$\n$"
        r"+ CP - A \ln \max \left( \frac{t}{t_{ttf}}, 1 \right)"
    )
)
# PD_MODELS['OmPD1p'] = OmPD1p_info = dict(
#     func=ompd1p, prior=ompd_prior, init=init_ompd, params=ompd_params,
#     name="OmPD'",
#     formula=(
#         r"\left(P_{\max} - CP\right)\frac{t_1 \left(1 - e^{\frac{t}{t_1}}\right) }{t}"
#         "$\n$"
#         r"+ CP - A \ln \left(1 + \frac{t}{t_{ttf}}\right)"
#     )
# )
PD_MODELS['Om3CP'] = Om3CP_info = dict(
    func=om3cp, prior=ompd_prior, init=init_ompd, params=ompd_params,
    name='Om3CP',
    formula=(
        r"\frac{P_{\max} - CP}{1 + \frac{t}{t_1}} + CP"
        "$\n$"
        r" - A \ln \max \left( \frac{t}{t_{ttf}}, 1 \right)"
    )
)
PD_MODELS['OmExp'] = OmExp_info = dict(
    func=omexp, prior=ompd_prior, init=init_ompd, params=ompd_params,
    name='OmExp',
    formula=(
        r"\left(P_{\max} - CP\right) e^{- \frac{t}{e t_1}}"
        "$\n$"
        r" + CP - A \ln \max \left( \frac{t}{t_{ttf}}, 1 \right)"
    )
)


def omws(params, t):
    CP, Pmax, t1, A, ttf, t2 = params
    t1 = jnp.exp(t1)
    tau2 = jnp.exp(t2)
    Wp = Pmax * t1
    # Wp, CP, A, ttf, t1, tau2 = params

    log_term = jnp.log(jnp.maximum(t / ttf, 1))
    anaerobic = - Wp * jnp.expm1(-t/t1) / t
    aerobic = - CP * jnp.expm1(-t/tau2) - A * log_term
    # aerobic = CP - A * log_term

    return anaerobic + aerobic


def init_omws(t, P, prior):
    CP, Pmax, t1, A, ttf, t2 = prior.mean
    t1 = jnp.exp(t1)
    tau2 = jnp.exp(t2)
    log_term = jnp.log((t / ttf).clip(1, None))
    X = np.c_[
        - jnp.expm1(-t/tau2),
        - jnp.expm1(-t/t1) / t,
        - log_term,
    ]
    (CP, Wp, A), *info = np.linalg.lstsq(X, P)
    Pmax = Wp / t1
    return jnp.r_[CP, Pmax, t1, A, ttf, t2]


omws_prior = NormalGamma.append(
    ompd_prior,
    NormalGamma.init(jnp.r_[4], jnp.r_[5e-4], 30, 10)
)
PD_MODELS['OmWS'] = pt_info = dict(
    func=omws, prior=omws_prior, init=init_omws,
    params=['CP', r"P_{\max}", 't_1', "A", r't_{ttf}', r'\tau_2'],
    name='Omni Ward-Smith',
    formula=(
        r"\frac{P_{\max} t_1}{t} \left(1 - e^{-\frac{t}{t_1}}\right)"
        "$\n$"
        r" + CP \left(1 - e^{-\frac{t}{\tau_2}}\right)"
        "$\n$"
        r" - A \ln \max \left( \frac{t}{t_{ttf}} \right)"
    )
)


def extended_cp(
        params, t,
        ecp_del=-1., tau_del=-4.8 / 60, ecp_dec=-1.0 / 60, ecp_dec_del=-180 * 60, paa_pow=1.05
):
    CP, Pmax, t1, A, t2, etau, tau2 = params
    # CP, Pmax, t1, A, t2, ecp_del = params
    t1 = jnp.exp(t1)
    ecp_del = - jnp.exp(tau2)
    paa = Pmax  # * t1
    ecp_dec = - A / CP
    ecp = CP
    paa_dec = - 1 / t1
    ecp_dec_del = - jnp.exp(t2)
    # etau *= 10
    # etau = jnp.expm1(tau_del) * jnp.expm1(ecp_del)
    # ecp_del = - jnp.exp(-tau2)
    # paa, etau, ecp, paa_dec, ecp_del = params
    # paa_dec, ecp_del = - jnp.exp(paa_dec), - jnp.exp(ecp_del)
    return (
        paa * jnp.exp(paa_dec * t**paa_pow)
        + ecp
        * jnp.expm1(tau_del*t)
        * jnp.expm1(ecp_del*t)
        * (1 + ecp_dec * jnp.exp(ecp_dec_del/t))
        * (1 + etau / t)
    )


def init_extcp(
        t, P, prior,
        ecp_del=-1., tau_del=-4.8 / 60, ecp_dec=-1.0 / 60, ecp_dec_del=-180 * 60, paa_pow=1.05,
        ani1=180, ani2=240, aei1=600, aei2=1200,
        sani1=20, sani2=90, laei1=4000, laei2=30000,
):
    CP, Pmax, lt1, A, lt2, etau, tau2 = prior.mean
    # CP, Pmax, lt1, A, lt2, ecp_del = prior.mean
    t1 = jnp.exp(lt1)
    ecp_del = - jnp.exp(tau2)
    paa = Pmax  # * t1
    ecp_dec = - A / CP
    ecp = CP
    paa_dec = - 1 / t1
    ecp_dec_del = - jnp.exp(lt2)
    # etau = jnp.expm1(tau_del) * jnp.expm1(ecp_del)

    # paa_dec, ecp_del = - jnp.exp(paa_dec), - jnp.exp(ecp_del)
    tmp = (
        jnp.expm1(tau_del*t)
        * jnp.expm1(ecp_del*t)
        * (1 + ecp_dec * jnp.exp(ecp_dec_del/t))
    )
    (paa, ecp, ecp_etau), *info = np.linalg.lstsq(
        np.c_[
            jnp.exp(paa_dec * t**paa_pow), tmp, tmp / t,
        ],
        P
    )
    etau = ecp_etau / ecp

    return jnp.r_[
        ecp, paa, lt1, A, lt2, etau, tau2,
        # ecp, paa, lt1, A, lt2, ecp_del,
        # paa, etau, ecp, paa_dec, ecp_del
    ]


extcp_prior = NormalGamma.append(
    CP_prior, Pmax_prior, t1_prior, A_prior, t2_prior,
    NormalGamma.init(jnp.r_[3.5, 0], jnp.r_[1e-4, 1e-4], 30, 10)
    # NormalGamma.init(jnp.r_[-1], jnp.r_[1e-4], 30, 10)
)
PD_MODELS['ExtCP'] = dict(
    func=extended_cp, prior=extcp_prior, init=init_extcp,
    name='Extended CP',
    params=["CP", r"P_{\max}", "t_1", "A",
            r"\tau_2", r'e_{\tau}', r'CP_{del}'],
    formula="P_{ext}(t)"
)
models = sorted(PD_MODELS)


def running_power(params, t, v):
    coefs = params[:-1]

    tp = t / params[-1]
    vadj = v / (1 + jnp.expm1(-tp)/tp)
    P = vadj * 0
    vi = vadj
    for ci in coefs:
        P += vi * ci
        vi *= vadj

    return P



def fit_posteriors(
    data, models=None, n_iter=10,
    seconds='seconds', power='power',
    limits=(0, None),
    fit_data=_fit_posterior,
):
    models = models or PD_MODELS
    posts = {}

    # t = np.array(data[seconds])
    # sel = t > limits[0]
    # if limits[1]:
    #     sel &= t < limits[1]

    t = np.array(data[seconds])  # [sel]
    y = np.array(data[power])  # [sel]

    for model, info in models.items():
        info = models[model]
        prior = info['prior']
        p = prior.mean
        if init := info.get('init'):
            w = jnp.isfinite(t) & jnp.isfinite(y) & between(t, *limits)
            p = init(t[w], y[w], prior)

        kws = {
            k: info[k] for k in ['func', 'prior', 'limits'] if k in info}

        posts[model] = post = fit_data(
            y, p, t, **kws, n_iter=n_iter)

    return posts

# def opt_models(
#     data, models=None, n_iter=10, seconds='seconds', power='power',
#     limits=(0, None)
# ):
#     models = models or PD_MODELS
#     posts = {}
#     sel = data[seconds] > limits[0]
#     if limits[1]:
#         sel &= data[seconds] < limits[1]

#     sel_data = data[sel]
#     t = sel_data[seconds].values
#     y = sel_data[power].values

#     fits = data.copy()
#     tfit = fits[seconds].values

#     for model, info in models.items():
#         info = models[model]
#         func = info['func']
#         prior = info['prior']
#         p = prior.mean
#         if init := info.get('init'):
#             w = jnp.isfinite(t) & jnp.isfinite(y)
#             p = init(t[w], y[w], prior)

#         kws = {k: info[k] for k in [
#             'func', 'prior', 'limits',
#         ] if k in info}
#         posts[model] = post = _opt_posterior(
#             t, y, p, **kws, n_iter=n_iter)

#         powerpost = post.posterior(func, tfit, jac=jac(func))
#         fits[model] = powerpost.mean
#         fits[f"{model}.std"] = powerpost.std
#         # fits[f"{model}.weight"] = weight(y, *powerpost)

#     return fits, posts


# def fit_models(
#     data, models=None, n_iter=10, seconds='seconds', power='power',
#     limits=(0, None)
# ):
#     models = models or PD_MODELS
#     posts = {}
#     sel = data[seconds] > limits[0]
#     if limits[1]:
#         sel &= data[seconds] < limits[1]

#     sel_data = data[sel]
#     t = sel_data[seconds].values
#     y = sel_data[power].values

#     fits = data.copy()
#     tfit = fits[seconds].values

#     for model, info in models.items():
#         info = models[model]
#         func = info['func']
#         prior = info['prior']
#         p = prior.mean
#         if init := info.get('init'):
#             w = jnp.isfinite(t) & jnp.isfinite(y)
#             p = init(t[w], y[w], prior)

#         kws = {k: info[k] for k in [
#             'func', 'prior', 'limits',
#         ] if k in info}
#         posts[model] = post = _fit_posterior(
#             y, p, t, **kws, n_iter=n_iter)

#         powerpost = post.posterior(func, tfit, jac=jac(func))
#         fits[model] = powerpost.mean
#         fits[f"{model}.std"] = powerpost.std
#         # fits[f"{model}.weight"] = weight(y, *powerpost)

#     return fits, posts


# def fit_weighted(data, models=None, n_iter=10, weight=weight, seconds='seconds', power='power', limits=(0, None)):
#     models = models or PD_MODELS
#     posts = {}
#     sel = data[seconds] > limits[0]
#     if limits[1]:
#         sel &= data[seconds] < limits[1]

#     sel_data = data[sel]
#     t = sel_data[seconds].values
#     y = sel_data[power].values

#     fits = data.copy()
#     tfit = fits[seconds].values
#     yfit = fits[power].values
#     for model, info in models.items():
#         info = models[model]
#         func = info['func']
#         prior = info['prior']
#         p = prior.mean
#         if init := info.get('init'):
#             w = jnp.isfinite(t) & jnp.isfinite(y)
#             p = init(t[w], y[w], prior)

#         kws = {k: info[k] for k in [
#             'func', 'prior', 'limits',
#         ] if k in info}
#         posts[model] = post = _fit_weighted_posterior(
#             y, p, t, **kws, weight=weight, n_iter=n_iter)

#         powerpost = post.posterior(func, tfit, jac=jac(func))
#         fits[model] = powerpost.mean
#         fits[f"{model}.std"] = powerpost.std
#         fits[f"{model}.weight"] = weight(yfit, *powerpost)

#     return fits, posts
