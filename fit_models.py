
from streamable import Stream
from tqdm.auto import tqdm
import numpy as np
import pandas as pd

import jax
from jax import scipy as jsp, numpy as jnp


from power_duration import models as cp, utils


def load_gc_data(url='data/goldencheetah/Power Duration Profile.xlsx'):
    raw_data = pd.read_excel(
        url,
        sheet_name='PD Profile',
        skiprows=range(14),
        index_col=[0, 1, 2, 3, 4]
    ).droplevel(0)

    times = raw_data.columns[
        raw_data.columns.map(pd.api.types.is_integer)
    ].values.astype(int)

    gc_data = raw_data.melt(
        ignore_index=False,
        value_vars=times,
        var_name='time',
        value_name='power'
    ).infer_objects().dropna()
    gc_data['seconds'] = gc_data.time
    gc_data['minutes'] = gc_data.time / 60
    gc_data['hours'] = gc_data.time / 3600
    gc_data['work'] = gc_data.time * gc_data.power
    gc_data = gc_data.reset_index()

    return gc_data


def load_p10_data(path="data/thepowerof10/events.csv"):
    event = pd.read_csv(
        path, header=None,
        names=['id', 'event', 'distance', 'seconds', 'date']
    )
    event.date = pd.to_datetime(event.date.str.strip("[]"))
    event['speed'] = event['distance'] / event.seconds
    event['power'] = event['speed']**3
    event['year'] = event.date.dt.year

    p10_data = event[
        (event.speed > 1)
        & (event.speed < 11)
        & (event.event != 6)
    ].sort_values('seconds').groupby(
        ['id', 'year', 'event', 'distance']
    ).first().reset_index()
    p10_data['id'] = p10_data['id'].astype(str)
    return p10_data


def load_non_data(path="data/nonathlon.csv"):
    return pd.read_csv(path).rename(columns={'athlete': 'id'})


def fit_models(
    data,
    fit_data=cp._fit_posterior,
    groups=['id', 'year'],
    min_power=10,
    min_datapoints=4,
    merge_on=None,
    models=None,
    concurrency=4,
    **kws,
):
    models = models or cp.PD_MODELS

    athlete_groups = data[
        (data.power > min_power)
        & (data.groupby(groups).transform('size') >= min_datapoints)
    ].groupby(groups)

    athlete_posts = (
        Stream(tqdm(athlete_groups))
        # .truncate(5)
        .map(
            lambda x: (x[0], cp.fit_posteriors(
                x[1]
                if merge_on is None else
                x[1].merge(merge_on, on=merge_on.index.name, how='outer'),
                models=models,
                fit_data=fit_data
            )),
            concurrency=concurrency,
        )
        .pipe(dict)
    )

    # avoiding memory leaks
    return jax.tree.map(np.array, athlete_posts)


def posterior_power(
    athlete_posts, data, groups=['id', 'year'], index_on=['seconds'], **kws
):
    ath_index = pd.MultiIndex.from_tuples(list(athlete_posts), names=groups)
    fit_params = {
        m: jnp.vstack([posts[m].mean for posts in athlete_posts.values()])
        for m in cp.models
    }
    index = ath_index.droplevel([])
    ath_power = data.set_index(groups + index_on).power
    droplevels = ath_power.index.droplevel(groups).names
    ath_power = ath_power[
        ath_power.index.droplevel(droplevels).isin(index)]
    t = ath_power.index.get_level_values(-1).values
    i = index.get_indexer_for(ath_power.index.droplevel(droplevels))

    fit_powers = pd.concat({
        m: pd.Series(
            np.array(cp.PD_MODELS[m]['func'](params[i].T, t)),
            ath_power.index,
            name=m
        )
        for m, params in fit_params.items()
    }, axis=1).join(ath_power)

    return fit_powers


def load_posts(path):
    posts = jax.tree.map(
        lambda x: cp.NormalGamma(*x),
        utils.load_gz(path),
        is_leaf=lambda x: isinstance(x, list)
    )
    flat_posts = utils.unnestkeys(posts)
    post_index = pd.MultiIndex.from_tuples(flat_posts)
    models = post_index.levels[-1]
    ath_index = post_index.droplevel(-1).drop_duplicates()
    ath_index = pd.MultiIndex.from_frame(ath_index.to_frame())
    params = pd.concat({
        m: pd.DataFrame(
            np.vstack([flat_posts[*k, m].mean for k in ath_index]),
            index=ath_index,
            columns=cp.PD_MODELS[m]['params']
        )
        for m in models
    }, axis=1)
    return posts, params


def getbests(data, groups=['id', 'seconds']):
    return data.sort_values('power', ascending=False).groupby(
        groups
    ).first().reset_index()


def is_posteriors(x):
    if isinstance(x, dict):
        return all(
            isinstance(v, cp.NormalGamma) for v in x.values())
    return False

@jax.jit
def model_log_evidence(ps, models=cp.PD_MODELS):
    return {
        k: ps[k].log_evidence(models[k]['prior'])
        for k in ps.keys() & models.keys()
    }

def post_log_evidence(posts):
    logZs = jax.tree.map(model_log_evidence, posts, is_leaf=is_posteriors)
    return pd.Series(utils.unnestkeys(logZs), dtype=float).unstack()


def main():
    save = 'fits'
    # save = 'test'
    gc_data = load_gc_data()
    p10_data = load_p10_data()
    non_data = load_non_data()

    t, v = p10_data.seconds.values, p10_data.speed.values
    p10_v = p10_data.assign(power=v * 60)
    p10_v2 = p10_data.assign(
        power=cp.running_power(np.r_[31.12, 5.48, 2.31], t, v))
    p10_v3 = p10_data.assign(
        power=cp.running_power(np.r_[77.38, -10.02,  1.26, 1.33], t, v))

    gc_times = pd.Series(
        gc_data.seconds.unique(),
        gc_data.seconds.unique(), name='duration'
    ).rename_axis('seconds')

    for d, data, kws in [
        # ('p10_speed', p10_v, dict(index_on=['event', 'seconds'])),
        # ('p10_v2', p10_v2, dict(index_on=['event', 'seconds'])),
        # ('p10_v3', p10_v3, dict(index_on=['event', 'seconds'])),
        ('p10_speed_ath', getbests(p10_v, ['id', 'event']), dict(
            groups=['id'], index_on=['event', 'seconds'])),
        ('p10_v2_ath', getbests(p10_v2, ['id', 'event']), dict(
            groups=['id'], index_on=['event', 'seconds'])),
        ('p10_v3_ath', getbests(p10_v3, ['id', 'event']), dict(
            groups=['id'], index_on=['event', 'seconds'])),
        # ('non_ath', getbests(non_data, ['id', 'test']), dict(
        #     groups=['id'], index_on=['test', 'seconds'])),
        # ('gc_ath', getbests(gc_data, ['id', 'seconds']), dict(
        #     groups=['id'], merge_on=gc_times)),
        # ('gc', gc_data, dict(merge_on=gc_times)),
        # ('non', non_data, dict(index_on=['test', 'seconds'])),
        # ('p10', p10_data, dict(index_on=['event', 'seconds'])),
        # ('p10_ath', getbests(p10_data, ['id', 'event']), dict(
        #     groups=['id'], index_on=['event', 'seconds'])),
    ]:
        for n, fit in [
            ('weighted', cp._fit_lm_weighted),
            ('fits', cp._fit_lm),
            # ('opts', cp._opt_posterior),
            # ('fits', cp._fit_posterior),
            # ('weighted', cp._fit_weighted_posterior),
        ]:
            name = f"{d}_{n}"
            print(name)

            athlete_posts = fit_models(data, fit, **kws, concurrency=1)
            fit_powers = posterior_power(athlete_posts, data, **kws)

            utils.dump_gz(
                utils.nestkeys(athlete_posts), f"{save}/{name}.json.gz")
            fit_powers.to_parquet(f"{save}/{name}.parquet")

            del athlete_posts, fit_powers


if __name__ == "__main__":
    main()
