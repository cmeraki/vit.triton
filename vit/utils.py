import sys
import torch
import functools

from loguru import logger

logger.remove()
logger.add(sys.stdout, format="[{time: YYYY-MM-DD HH:mm:ss} {level}] {message}", level="INFO")

def tensor_info(func_name):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            logger.info(f"Function {func_name} called")
            for i, arg in enumerate(args):
                if isinstance(arg, torch.Tensor):
                    logger.info(f"Input\t{arg.name if hasattr(arg, 'name') else 'No Name'}\t{arg.shape}")
            for key, value in kwargs.items():
                if isinstance(value, torch.Tensor):
                    logger.info(f"Input\t{key}\t{value.shape}")

            results = func(*args, **kwargs)

            if isinstance(results, torch.Tensor):
                logger.info(f"Output\t{results.name if hasattr(results, 'name') else 'No Name'}\t{results.shape}")
            elif isinstance(results, (tuple, list)):
                for i, result in enumerate(results):
                    if isinstance(result, torch.Tensor):
                        logger.info(f"Output\t{result.name if hasattr(result, 'name') else 'No Name'}\t{result.shape}")
            return results

        return wrapper
    return decorator
