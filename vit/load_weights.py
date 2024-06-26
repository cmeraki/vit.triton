import sys
import torch
from transformers import ViTModel
from typing import OrderedDict, Dict
from loguru import logger

logger.remove()
logger.add(sys.stdout, format="[{time: YYYY-MM-DD HH:mm:ss} {level}] {message}", level="INFO")


def map_attn_layers(source_layer_num: str, source_proj: str, source_type: str, source_tensor: torch.Tensor, dest_state_dict: dict) -> Dict:
    """
    Maps query, key and value weight matrices from pretrained model to custom model.
    This is required to handle seperately because in HF, the weight matrices are [d_model, d_head*n_head].
    This means, the weights for all the heads are concatenated in the same matrix.
    Whereas in our custom model, the weights are separated in different matrices.
    """

    for layer, weight in dest_state_dict.items():
        if len(layer.split('.')) <= 2:
            continue

        layer_num, proj, type = layer.split('.')[2], layer.split('.')[-2], layer.split('.')[-1]
        if (layer_num == source_layer_num) and (proj == source_proj) and (type == source_type):
            num_head = int(layer.split('.')[5])

            if type == 'weight':
                src = source_tensor.T.contiguous()
                src = src[:, num_head*64:(num_head+1)*64]
            else:
                src = source_tensor[num_head*64:(num_head+1)*64]

            logger.debug(f'Mapping to destination\tLayer: {layer}\tSource tensor shape: {src.shape}\tDestination tensor shape: {weight.shape}')

            dest_state_dict[layer] = src.clone()

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


