from typing import Union

import torch.nn

from modules.FFC import LaMa


def get_values(network_config: dict, key: str, value: Union[int, float]):
    if key in network_config:
        dictionary = network_config[key]
    else:
        dictionary = {
            'ratio_gin': value,
            'ratio_gout': value
        }
    return dictionary


def configure_network(network_config: dict) -> torch.nn.Module:
    input_channels = network_config["input_channels"] if "input_channels" in network_config else 3
    output_channels = network_config["output_channels"] if "output_channels" in network_config else 1
    block_number = network_config["block_number"] if "block_number" in network_config else 9

    init_conv_kwargs = get_values(network_config, "init_conv_kwargs", 0)
    down_sample_conv_kwargs = get_values(network_config, "down_sample_conv_kwargs", 0)
    resnet_conv_kwargs = get_values(network_config, "resnet_conv_kwargs", 0.75)

    model = LaMa(input_nc=input_channels,
                 output_nc=output_channels,
                 n_blocks=block_number,
                 init_conv_kwargs=init_conv_kwargs,
                 downsample_conv_kwargs=down_sample_conv_kwargs,
                 resnet_conv_kwargs=resnet_conv_kwargs)
    return model
