from typing import TypedDict

import numpy as np
import torch.utils.data

from dataloader import HandwrittenTextImageDataset
from modules.FFC import LaMa

if __name__ == '__main__':
    valid_dataset = HandwrittenTextImageDataset('patches/valid', '../patches/valid_gt', augmentation=False)
    Arguments = TypedDict('Arguments', {'ratio_gin': float, 'ratio_gout': float})
    init_conv_kwargs: Arguments = {'ratio_gin': 0, 'ratio_gout': 0}
    down_sample_conv_kwargs: Arguments = {'ratio_gin': 0, 'ratio_gout': 0}
    resnet_conv_kwargs: Arguments = {'ratio_gin': 0.75, 'ratio_gout': 0.75}
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataloader = torch.utils.data.DataLoader(valid_dataset, collate_fn=valid_dataset.collate_fn, shuffle=True,
                                             num_workers=0,
                                             pin_memory=True)

    model = LaMa(input_nc=3, output_nc=1, init_conv_kwargs=init_conv_kwargs,
                 downsample_conv_kwargs=down_sample_conv_kwargs, resnet_conv_kwargs=resnet_conv_kwargs)
    model.load_state_dict(torch.load("../weights/last_training.pt"))
    model.eval()

    for index, (valid_index, valid_dg_img, valid_gt_img) in enumerate(dataloader):
        batch_size = len(valid_index)

        input = valid_dg_img.to(device)
        output = valid_gt_img.to(device)

        result = model(input)
        for i in range(0, batch_size):
            img_dg = np.swapaxes(input[i].numpy(), 2, 0)
            img_gt = np.swapaxes(output[i].numpy(), 2, 0)
            img_pred = result[i].detach().numpy()
        exit(0)