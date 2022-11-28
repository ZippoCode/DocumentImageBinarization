import torch

from dataloader import HandwrittenTextImageDataset
from train import LMSELoss
from trainer.CustomTransforms import create_train_transform, create_valid_transform
from trainer.LaMaTrainer import LaMaTrainingModule

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_data_path = 'patches/train'
valid_data_path = 'patches/valid'
input_channels = 3
output_channels = 1
train_batch_size = 8
valid_batch_size = 8
train_split_size = 256
valid_split_size = 512
workers = 0
debug = True
epochs = 150
path_checkpoint = 'weights/all_transform_v_size_512/'
criterion = LMSELoss().to(device)

trainer = LaMaTrainingModule(train_data_path=train_data_path, valid_data_path=valid_data_path,
                             input_channels=input_channels, output_channels=output_channels,
                             train_batch_size=train_batch_size, valid_batch_size=valid_batch_size,
                             train_split_size=train_split_size, valid_split_size=valid_split_size,
                             workers=workers, epochs=epochs, debug=debug, device=device, learning_rate=1,
                             criterion=criterion)


def test_dataloader():
    transform = create_train_transform(patch_size=256, angle=10)
    train_dataset = HandwrittenTextImageDataset('patches/train', 'patches/train_gt', transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=2, shuffle=True)
    for index, (sample, gt_sample) in enumerate(train_loader):
        print(sample.size(), gt_sample.size())
        if index == 10:
            break

    test_transform = create_valid_transform(patch_size=128)
    valid_dataset = HandwrittenTextImageDataset('patches/valid', 'patches/valid_gt', transform=test_transform)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=2, shuffle=False)
    for index, (sample, gt_sample) in enumerate(valid_loader):
        print(sample.size(), gt_sample.size())
        if index == 10:
            break

    test_dataset = HandwrittenTextImageDataset('patches/test', 'patches/test_gt')
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=3, shuffle=False)

    print("Train size: ", len(train_loader))
    print("Valid size: ", len(valid_loader))
    print("Test size: ", len(test_loader))


def test_validation():
    trainer.load_checkpoints(path_checkpoint)
    trainer.model.eval()

    with torch.no_grad():
        trainer.validation()


def test_save_and_load():
    trainer.save_checkpoints('weights/')
    trainer.load_checkpoints('weights/')


if __name__ == '__main__':
    # test_dataloader()
    test_validation()
    # test_save_and_load(trainer)
