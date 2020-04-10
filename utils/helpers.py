import numpy as np


def clean_params(params):
    params = params.copy()

    for key, val in params.items():
        if isinstance(params[key], np.int32):
            params[key] = int(val)
        elif isinstance(params[key], np.float):
            params[key] = float(round(val, 7))

    return params
