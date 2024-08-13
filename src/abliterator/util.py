import gc
from itertools import islice

import torch
from jaxtyping import Float
from torch import Tensor


def batch(iterable, n):
    it = iter(iterable)
    while True:
        chunk = list(islice(it, n))
        if not chunk:
            break
        yield chunk


def clear_mem():
    gc.collect()
    torch.cuda.empty_cache()


def measure_fn(
    measure: str, input_tensor: Tensor, *args, **kwargs
) -> Float[Tensor, "..."]:
    avail_measures = {
        "mean": torch.mean,
        "median": torch.median,
        "max": torch.max,
        "stack": torch.stack,
    }
    try:
        return avail_measures[measure](input_tensor, *args, **kwargs)
    except KeyError:
        raise NotImplementedError(
            f"Unknown measure function '{measure}'. Available measures:"
            + ", ".join([f"'{str(fn)}'" for fn in avail_measures.keys()])
        )
