import torch
from typing_extensions import TypedDict

from dataloader import HandwrittenTextImageDataset
from modules.FFC import LaMa
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test_dataloader():
    train_dataset = HandwrittenTextImageDataset('patches/train', 'patches/train_gt')
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=2, shuffle=True)
    for sample, gt_sample in train_loader:
        print(sample.size())

    valid_dataset = HandwrittenTextImageDataset('patches/valid', 'patches/valid_gt')
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=8, shuffle=False)

    test_dataset = HandwrittenTextImageDataset('patches/test', 'patches/test_gt')
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=3, shuffle=False)

    print("Train size: ", len(train_loader))
    print("Valid size: ", len(valid_loader))
    print("Test size: ", len(test_loader))


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    test_dataloader()

    # Arguments = TypedDict('Arguments', {'ratio_gin': float, 'ratio_gout': float})
    # init_conv_kwargs: Arguments = {'ratio_gin': 0, 'ratio_gout': 0}
    # down_sample_conv_kwargs: Arguments = {'ratio_gin': 0, 'ratio_gout': 0}
    # resnet_conv_kwargs: Arguments = {'ratio_gin': 0.75, 'ratio_gout': 0.75}
    #
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #
    # lama = LaMa(input_nc=3, output_nc=1, init_conv_kwargs=init_conv_kwargs,
    #             downsample_conv_kwargs=down_sample_conv_kwargs, resnet_conv_kwargs=resnet_conv_kwargs)
    # lama.to(device)
    #
    # for _, (test_in, test_out) in enumerate(valid_test):
    #     bs = len(test_in)
    #     inputs = test_in.to(device)
    #     outputs = test_out.to(device)
    #     with torch.no_grad():
    #         loss, _, pred_pixel_values = lama(inputs, outputs)
    #         rec_patches = pred_pixel_values
    #         rec_images = rearrange(rec_patches, 'b (h w) (p1 p2 c) -> b c (h p1) (w p2)',
    #                                p1=patch_size, p2=patch_size, h=image_size[0] // patch_size)
    #         for j in range(0, bs):
    #             imvisualize(inputs[j].cpu(), outputs[j].cpu(), rec_images[j].cpu(), test_index[j],
    #                         epoch, experiment)
    #         losses += loss.item()
    # print('test loss: ', losses / len(testloader))

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
# import torch
# from typing_extensions import TypedDict
#
# from dataloader import HandwrittenTextImageDataset
# from modules.FFC import LaMa
#
# if __name__ == '__main__':
#     htDatasetTrain = HandwrittenTextImageDataset('patches/train', 'patches/train_gt')
#     train_loader = torch.utils.data.DataLoader(htDatasetTrain)
#     print(f"Founded {len(train_loader)} images")
#
#     valid_dataset = HandwrittenTextImageDataset('patches/valid', 'patches/valid_gt')
#     valid_loader = torch.utils.data.DataLoader(valid_dataset)
#     print(f"Founded {len(valid_loader)} valid images")
#
#     # Test LaMa
#     Arguments = TypedDict('Arguments', {'ratio_gin': float, 'ratio_gout': float})
#     init_conv_kwargs: Arguments = {'ratio_gin': 0, 'ratio_gout': 0}
#     down_sample_conv_kwargs: Arguments = {'ratio_gin': 0, 'ratio_gout': 0}
#     resnet_conv_kwargs: Arguments = {'ratio_gin': 0.75, 'ratio_gout': 0.75}
#
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
#     lama = LaMa(input_nc=3, output_nc=1, init_conv_kwargs=init_conv_kwargs,
#                 downsample_conv_kwargs=down_sample_conv_kwargs, resnet_conv_kwargs=resnet_conv_kwargs)
#     lama.to(device=device)
#     x = torch.rand((8, 3, 256, 256), device=device)
#     out = lama(x)
#     print(out.shape)
