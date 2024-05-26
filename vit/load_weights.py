import sys
import torch
from transformers import ViTModel
from typing import Dict
from loguru import logger

logger.remove()
logger.add(sys.stdout, format="[{time: YYYY-MM-DD HH:mm:ss} {level}] {message}", level="INFO")


def map_attn_layers(source_layer_num: int, source_model: ViTModel, dest_state_dict: dict) -> Dict:
    """
    Maps query, key and value weight matrices from pretrained model to custom model.
    This is required to handle seperately because in HF, the weight matrices are [d_model, d_head*n_head].
    This means, the weights for all the heads are concatenated in the same matrix.
    Whereas in our custom model, the weights are separated in different matrices.
    """

    logger.debug(f"Transferring attention layer weights from {source_layer_num}")
    q = source_model.encoder.layer[source_layer_num].attention.attention.query
    k = source_model.encoder.layer[source_layer_num].attention.attention.key
    v = source_model.encoder.layer[source_layer_num].attention.attention.value

    dest_qkv_weight = torch.cat([q.weight.T, k.weight.T, v.weight.T], dim=-1)
    dest_qkv_weight = dest_qkv_weight.contiguous()

    dest_qkv_bias = torch.cat([q.bias, k.bias, v.bias], dim=-1)
    dest_qkv_bias = dest_qkv_bias.contiguous()

    dest_state_dict[f'encoder.layer.{source_layer_num}.attention.attention.qkv.weight'] = dest_qkv_weight.clone()
    dest_state_dict[f'encoder.layer.{source_layer_num}.attention.attention.qkv.bias'] = dest_qkv_bias.clone()

    return dest_state_dict

def map_non_attn_layers(source_state_dict: dict, dest_state_dict: dict, weight_mapping: Dict) -> Dict:
    """
    Map non attention layers (patching, feed forward, layernorm etc.)
    In MLP layer, weights are transferred after applying a transpose just due to the convention of how HF stores the weight
    and how I decide to store the weights.
    """

    for key, value in source_state_dict.items():
        mapped_key = weight_mapping.get(key)
        if mapped_key and mapped_key in dest_state_dict:
            logger.debug(f"Transferring weight from {key} to {mapped_key}")

            if ('output' in mapped_key or 'intermediate' in mapped_key):
                logger.debug(f"Tranferring transpose weights")
                dest_state_dict[mapped_key] = value.t().clone()
                continue

            dest_state_dict[mapped_key] = value.clone()
            continue

        else:
            logger.debug(f'Not transferring weight for: {key}')

    return dest_state_dict


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

    for k in range(12):
        custom_state_dict = map_attn_layers(k, pretrained_model, custom_state_dict)

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
