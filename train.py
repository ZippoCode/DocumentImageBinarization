import argparse
import logging
import time

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
import torchvision.transforms.functional as functional

from trainer.LaMaTrainer import LaMaTrainingModule

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)


def train(args):
    start_time = time.time()
    trainer = LaMaTrainingModule(train_data_path=args.train_data_path, valid_data_path=args.valid_data_path,
                                 input_channels=args.input_channels, output_channels=args.output_channels,
                                 train_batch_size=args.train_batch_size, valid_batch_size=args.valid_batch_size,
                                 train_split_size=args.train_split_size, valid_split_size=args.valid_split_size,
                                 workers=args.workers, epochs=args.epochs, debug=args.debug, device=device)
    criterion = torch.nn.MSELoss()

    # Denormalize
    denormalize_transform = transforms.Compose(
        [transforms.Normalize(mean=[0., 0., 0.], std=[1 / 0.229, 1 / 0.224, 1 / 0.225]),
         transforms.Normalize(mean=[-0.485, -0.456, -0.406], std=[1., 1., 1.])])

    if torch.cuda.is_available():
        trainer.model.cuda()

    try:

        for epoch in range(1, args.epochs):

            trainer.model.train()
            train_loss = 0.0

            for i, (train_in, train_out) in enumerate(trainer.train_data_loader):
                elapsed_time = time.time() - start_time

                inputs = train_in.to(device)
                outputs = train_out.to(device)

                trainer.optimizer.zero_grad()

                pred = trainer.model(inputs)
                loss = criterion(pred, outputs)
                predicted_time = time.time() - start_time

                loss.backward()
                trainer.optimizer.step()

                train_loss += loss.item()

                # Build image
                transform = transforms.ToPILImage()
                train_img = np.array(transform(denormalize_transform(inputs[0])))
                train_pred_img = np.array(transform(pred[0]))
                train_pred_img = (train_pred_img > 127) * 255
                train_gt_img = np.array(transform(outputs[0]))

                images = [train_img, train_pred_img, train_gt_img]

                if not trainer.debug:
                    trainer.wandb.log({
                        "Images": [trainer.wandb.Image(image) for image in images]
                    })

                # Logging
                logging.info(f'i: {i} - epoch: {epoch}')
                logging.info(f'Loss: {train_loss:0.6f} time {elapsed_time:0.1f}, {predicted_time:0.1f}')

            avg_loss = train_loss / len(trainer.train_data_loader)

            # Validation
            trainer.model.eval()
            with torch.no_grad():
                psnr, valid_loss = trainer.validation()

                if psnr > trainer.best_psnr:
                    trainer.save_checkpoint(args.path_checkpoint)
                    trainer.best_psnr = psnr

            if not trainer.debug:
                trainer.wandb.log({
                    'epoch': trainer.epoch,
                    'PSNR': psnr,
                    'train_loss': avg_loss,
                    'valid_loss': valid_loss,
                })
            trainer.epoch += 1


    except KeyboardInterrupt:
        logging.info("Keyboard Interrupt. Stop training!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data_path', type=str, default='patches/train')
    parser.add_argument('--valid_data_path', type=str, default='patches/valid')
    parser.add_argument('--train_batch_size', type=int, default=4)
    parser.add_argument('--valid_batch_size', type=int, default=4)
    parser.add_argument('--train_split_size', type=int, default=256)
    parser.add_argument('--valid_split_size', type=int, default=256)
    parser.add_argument('--input_channels', type=int, default=3)
    parser.add_argument('--output_channels', type=int, default=1)
    parser.add_argument('--workers', default=4, type=int)
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--path_checkpoint', type=str, default='weights/')
    parser.add_argument('--debug', type=bool, default=False)

    args = parser.parse_args()
    train(args)
