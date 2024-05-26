import sys
import torch
import functools
import numpy as np
from tqdm import tqdm
from typing import List, Tuple

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

            logger.info(f"Function {func_name} exited")
            if isinstance(results, torch.Tensor):
                logger.info(f"Output\t{results.name if hasattr(results, 'name') else 'No Name'}\t{results.shape}")
            elif isinstance(results, (tuple, list)):
                for i, result in enumerate(results):
                    if isinstance(result, torch.Tensor):
                        logger.info(f"Output\t{result.name if hasattr(result, 'name') else 'No Name'}\t{result.shape}")
            return results

        return wrapper
    return decorator


def capture_cuda_graph(model, static_input):
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())

    with torch.no_grad():
        with torch.cuda.stream(stream):
            for i in range(5):
                _ = model(static_input)

    torch.cuda.current_stream().wait_stream(stream)

    # Record the CUDA graph
    graph = torch.cuda.CUDAGraph()

    # Capture the graph
    with torch.cuda.graph(graph):
        static_output = model(static_input)

    return graph, static_output


def benchmark(
        model1: torch.nn.Module,
        model2: torch.nn.Module,
        input_shape: Tuple[int, int, int] = (3, 224, 224),
        batch_sizes: List[int] = [1, 4, 16, 32, 64, 128, 256],
        warmups: int = 50,
        reps: int = 100
    ):
    """
    Benchmark two models on different batch sizes

    Returns:
        Tuple[List]: 2 lists with model time for different models
    """

    for bs in tqdm(batch_sizes, total=len(batch_sizes)):
        model1_times = []
        model2_times = []

        logger.info(f'Running for batch size: {bs}')

        a = torch.randn((bs, *input_shape)).to(device=model1.device, dtype=model1.dtype)

        # warmup run
        logger.info('Doing a warmup run')
        for _ in range(warmups):
            with torch.no_grad():
                _ = model1(a)
                _ = model2(a)
        logger.info('Warmup run complete')

        for _ in range(reps):
            with torch.no_grad():
                o1, model1_time = timed(model1, a)
            model1_times.append(model1_time)

        for _ in range(reps):
            with torch.no_grad():
                o2, model2_time = timed(model2, a)
            model2_times.append(model2_time)
        
        logger.info(f'Diff: {torch.mean(torch.abs(o1[0]-o2))}')
        yield (bs, np.median(model1_times), np.median(model2_times))


def timed(fn, input):
    """
    Times a model call on GPU
    """
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    result = fn(input)
    end.record()
    torch.cuda.synchronize()
    return result, start.elapsed_time(end)
