from functools import partial
import gzip
import json

import numpy as np
import jax
from jax import numpy as jnp


class JSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, dict):
            return {
                self.default_key(k): v
                for k, v in obj.items()
            }
        elif isinstance(obj, np.generic):
            return obj.item()
        elif isinstance(obj, (np.ndarray, jax.Array)):
            return dict(
                data=obj.tolist(),
                dtype=str(obj.dtype),
                __cls__='Array',
                __module__=type(obj).__module__,
            )

        return json.JSONEncoder.default(self, obj)

    def default_key(self, key):
        if isinstance(key, (int, float, bool, str)):
            return key

        return self.default(key)
    
    def _encode(self, obj):
        if isinstance(obj, dict):
            obj = {
                self.default_key(k): self.default(v)
                for k, v in obj.items()
            }

        return obj

    def encode(self, obj):
        return super().encode(self._encode(obj))
    
    def iterencode(self, obj):
        return super().iterencode(self._encode(obj))


def load_json_array(data):
    mod = data.get('__module__', 'numpy')
    if mod == 'numpy':
        return np.array(
            data['data'], dtype=data['dtype'],
        )
    else:
        return jnp.array(
            data['data'], dtype=data['dtype'],
        )


JSON_LOAD = dict(
    Array=load_json_array,
    # NormalGamma=NormalGamma.from_dict,
)


def json_hook(data):
    if isinstance(data, dict) and '__cls__' in data:
        load = JSON_LOAD[data['__cls__']]
        return load(data)

    return data


dump = partial(json.dump, cls=JSONEncoder)
dumps = partial(json.dumps, cls=JSONEncoder)
load = partial(json.load, object_hook=json_hook)
loads = partial(json.loads, object_hook=json_hook)


def getnested(obj, *keys):
    for key in keys:
        obj = obj[key]
    return obj


def setnested(obj, value, *keys):
    obj = getnested(obj, *keys[:-1])
    obj[keys[-1]] = value


def getnesteddefault(obj, *keys):
    for key in keys:
        obj = obj.setdefault(key, {})
    return obj


def setnesteddefault(obj, value, *keys):
    obj = getnesteddefault(obj, *keys[:-1])
    return obj.setdefault(keys[-1], value)


def nestkeys(data):
    nested = {}
    for ks, v in data.items():
        setnesteddefault(nested, v, *ks)

    return nested


def _unnest(data, parent=()):
    if isinstance(data, dict):
        for k, v in data.items():
            yield from _unnest(v, parent + (k,))
    else:
        yield parent, data


def unnestkeys(data):
    return dict(_unnest(data))


def dump_gz(obj, filepath, mode='wt', encoding='UTF-8'):
    with gzip.open(filepath, mode, encoding=encoding) as f:
        return dump(obj, f)


def load_gz(filepath, mode='rt', encoding='UTF-8'):
    with gzip.open(filepath, mode, encoding=encoding) as f:
        return load(f)


def append_to_parquet(dfs, path, row_group_size=None, **kwargs):
    import pyarrow as pa
    import pyarrow.parquet as pq

    schema = pqwriter = None
    try:
        for df in dfs:
            if df.empty:
                continue

            table = pa.Table.from_pandas(df, schema)
            if pqwriter is None:
                schema = table.schema
                pqwriter = pq.ParquetWriter(path, schema, **kwargs)

            pqwriter.write_table(table, row_group_size=row_group_size)
    finally:
        if pqwriter:
            pqwriter.close()

    return pqwriter
