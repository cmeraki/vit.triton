import sys
import torch
import functools
import numpy as np
from tqdm import tqdm

from typing import List, Tuple

from loguru import logger
from .load_weights import (
    map_non_attn_layers,
    map_attn_layers
)

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


def transfer_pretrained_weights(pretrained_model: torch.nn.Module, custom_model: torch.nn.Module) -> torch.nn.Module:
    """
    Transfer weights from a pretrained model to the custom model
    """

    pretrained_state_dict = pretrained_model.state_dict()
    custom_state_dict = custom_model.state_dict()

    num_layers = 12

    # Mapping dictionary from source model to destination model
    weight_mapping = {
        'embeddings.cls_token': 'embeddings.cls_token',
        'embeddings.position_embeddings': 'embeddings.position_embeddings',
        'embeddings.patch_embeddings.projection.weight': 'embeddings.projection.weight',
        'embeddings.patch_embeddings.projection.bias': 'embeddings.projection.bias',
        'layernorm.weight': 'layernorm.weight',
        'layernorm.bias': 'layernorm.bias',
        'pooler.dense.weight': 'pooler.dense.weight',
        'pooler.dense.bias': 'pooler.dense.bias'
    }
   
    # Adding mappings for each encoder layer's output and intermediate dense layers
    for i in range(num_layers):
        weight_mapping.update({
            f'encoder.layer.{i}.output.dense.weight': f'encoder.layer.{i}.output.weight',
            f'encoder.layer.{i}.output.dense.bias': f'encoder.layer.{i}.output.bias',
            f'encoder.layer.{i}.intermediate.dense.weight': f'encoder.layer.{i}.intermediate.weight',
            f'encoder.layer.{i}.intermediate.dense.bias': f'encoder.layer.{i}.intermediate.bias',
            f'encoder.layer.{i}.attention.output.dense.weight': f'encoder.layer.{i}.attention.output.weight',
            f'encoder.layer.{i}.attention.output.dense.bias': f'encoder.layer.{i}.attention.output.bias',
            f'encoder.layer.{i}.layernorm_before.weight': f'encoder.layer.{i}.layernorm_before.weight',
            f'encoder.layer.{i}.layernorm_before.bias': f'encoder.layer.{i}.layernorm_before.bias',
            f'encoder.layer.{i}.layernorm_after.weight': f'encoder.layer.{i}.layernorm_after.weight',
            f'encoder.layer.{i}.layernorm_after.bias': f'encoder.layer.{i}.layernorm_after.bias'
        })

    # Transfer the weights of query, key, value layers
    attention_layers = [
        f'encoder.layer.{k}.attention.attention' for k in range(num_layers)
    ]

    for layer_name, weight in pretrained_state_dict.items():
        for attn_layer in attention_layers:
            if attn_layer in layer_name:
                layer_num, proj, type = layer_name.split('.')[2], layer_name.split('.')[-2], layer_name.split('.')[-1]
                logger.debug(f'Mapping from source\t{layer_name}\tLayer number: {layer_num} \tProjection: {proj} \tType: {type}')
                custom_state_dict = map_attn_layers(layer_num, proj, type, weight, custom_state_dict)

    # Transfer rest of the layers
    custom_state_dict = map_non_attn_layers(
        source_state_dict=pretrained_state_dict,
        dest_state_dict=custom_state_dict,
        weight_mapping=weight_mapping
    )

    custom_model.load_state_dict(custom_state_dict, strict=False)

    custom_state_dict = custom_model.state_dict()
    uninitialized_layers = []
    for k, v in custom_state_dict.items():
        if torch.all(v == 0):
            uninitialized_layers.append(k)

    if uninitialized_layers:
        # Pooler bias has all zeros, so that is expected to be here
        print(f"Some layer are not initialized: {uninitialized_layers}")

    return custom_model

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
        input_shape: Tuple[int] = (3, 224, 224),
        batch_sizes: List[int] = [1, 4, 16, 32, 64, 128, 256],
        warmups: int = 50,
        reps: int = 100
    ) -> Tuple[List]:
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
        logger.info(f'Doing a warmup run')
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
        logger.info(f'Batch size: {bs}\tModel 1 Mean: {np.mean(model1_times)}, Median: {np.mean(model1_times)}\tModel 2 Mean: {np.mean(model2_times)}, Median: {np.mean(model2_times)}')


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
