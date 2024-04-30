import torch
import functools


def tensor_info(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        for i, arg in enumerate(args):
            if isinstance(arg, torch.Tensor):
                print(f"{arg.name if hasattr(arg, 'name') else 'No Name'}\t{arg.shape}")
        for key, value in kwargs.items():
            if isinstance(value, torch.Tensor):
                print(f"{key}\t{value.shape}")

        results = func(*args, **kwargs)

        if isinstance(results, torch.Tensor):
            print(f"{results.name if hasattr(results, 'name') else 'No Name'}\t{results.shape}")
        elif isinstance(results, (tuple, list)):
            for i, result in enumerate(results):
                if isinstance(result, torch.Tensor):
                    print(f"{result.name if hasattr(result, 'name') else 'No Name'}\t{result.shape}")
        return results

    return wrapper
