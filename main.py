import torch

from train import LMSELoss
from trainer.LaMaTrainer import LaMaTrainingModule

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_data_path = 'patches/train'
train_gt_data_path = 'patches/train_gt'
valid_data_path = 'patches/valid/full'
valid_gt_data_path = 'patches/valid_gt/full'
input_channels = 3
output_channels = 1
train_batch_size = 8
valid_batch_size = 1
train_split_size = 256
valid_split_size = 256
workers = 0
debug = True
epochs = 150
path_checkpoint = 'weights/custom_loss/'
criterion = LMSELoss().to(device)

trainer = LaMaTrainingModule(train_data_path=train_data_path, train_gt_data_path=train_gt_data_path,
                             valid_data_path=valid_data_path, valid_gt_data_path=valid_gt_data_path,
                             input_channels=input_channels, output_channels=output_channels,
                             train_batch_size=train_batch_size, valid_batch_size=valid_batch_size,
                             train_split_size=train_split_size, valid_split_size=valid_split_size,
                             workers=workers, epochs=epochs, debug=debug, device=device, learning_rate=1,
                             criterion=criterion, experiment_name='main')


def test_validation():
    trainer.load_checkpoints(path_checkpoint)
    trainer.model.eval()

    with torch.no_grad():
        trainer.validation()


def test_save_and_load():
    #    trainer.save_checkpoints('blablabla/')
    trainer.load_checkpoints('blablabla/')


if __name__ == '__main__':
    # test_dataloader()
    # test_validation()
    test_save_and_load()
