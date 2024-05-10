import sys
import torch
from transformers import ViTModel
from typing import OrderedDict
from loguru import logger

logger.remove()
logger.add(sys.stdout, format="[{time: YYYY-MM-DD HH:mm:ss} {level}] {message}", level="INFO")


def load_pretrained_model(model_id):
    pretrained_model = ViTModel.from_pretrained(model_id)
    pretrained_state_dict = pretrained_model.state_dict()

    return pretrained_model, pretrained_state_dict

def map_attn_layers(source_layer_num: str, source_proj: str, source_type: str, source_tensor: torch.Tensor, dest_state_dict: OrderedDict):
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
                src = source_tensor[:, num_head*64:(num_head+1)*64]
            else:
                src = source_tensor[num_head*64:(num_head+1)*64]

            logger.info(f'Mapping to destination\tLayer: {layer}\tSource tensor shape: {src.shape}\tDestination tensor shape{weight.shape}')

            dest_state_dict[layer] = src.clone()

def map_nonattn_layers(source_state_dict: OrderedDict, dest_state_dict: OrderedDict, ):
    for key, value in pretrained_state_dict.items():
        mapped_key = weight_mapping.get(key)
        if mapped_key and mapped_key in custom_state_dict:
            # print(f"Transferring weight from {key} to {mapped_key}")

            if 'attention' not in mapped_key and ('output' in mapped_key or 'intermediate' in mapped_key):
                # print(f"Tranferring transpose weights")
                custom_state_dict[mapped_key] = value.t().clone()
                continue
            
            custom_state_dict[mapped_key] = value.clone()
            continue

        else:
            print(f'Not transferring weight for: {key}')