import sys
import torch
import functools
from loguru import logger
from .load_weights import (
    load_pretrained_model,
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


def transfer_pretrained_weights(model_id: str, custom_model: torch.nn.Module) -> torch.nn.Module:
    """
    Transfer weights from a pretrained model to the custom model
    """

    pretrained_state_dict = load_pretrained_model(model_id)
    custom_state_dict = custom_model.state_dict()

    num_layers = 12

    # Mapping dictionary from source model to destination model
    weight_mapping = {
        'embeddings.position_embeddings': 'embeddings.position_embeddings',
        # TODO: P0 Fix mapping for projection layers - Need to write a conv2d kernel
        # 'embeddings.patch_embeddings.projection.weight': 'embeddings.projection.weight',
        # 'embeddings.patch_embeddings.projection.bias': 'embeddings.projection.bias'
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
                logger.info(f'Mapping from source\tLayer number: {layer_num} \tProjection: {proj} \tType: {type}')
                custom_state_dict = map_attn_layers(layer_num, proj, type, weight, custom_state_dict)

    # Transfer rest of the layers
    custom_state_dict = map_non_attn_layers(
        source_state_dict=pretrained_state_dict,
        dest_state_dict=custom_state_dict,
        weight_mapping=weight_mapping
    )

    custom_model.load_state_dict(custom_state_dict, strict=False)

    return custom_model