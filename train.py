import argparse
import math
import sys
import time

import numpy as np
import torch

from trainer.LaMaTrainer import LaMaTrainingModule
from utils.htr_logging import get_logger

debug = False
get_trace = getattr(sys, 'gettrace', None)
if get_trace():
    print('Program runs in Debug mode')
    debug = True

logger = get_logger('main', debug)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"Using {device} device")
threshold = 0.5


def train(args):
    start_time = time.time()

    trainer = LaMaTrainingModule(train_data_path=args.train_data_path, valid_data_path=args.valid_data_path,
                                 input_channels=args.input_channels, output_channels=args.output_channels,
                                 train_batch_size=args.train_batch_size, valid_batch_size=args.valid_batch_size,
                                 train_split_size=args.train_split_size, valid_split_size=args.valid_split_size,
                                 workers=args.workers, epochs=args.num_epochs, learning_rate=args.learning_rate,
                                 debug=debug, device=device)
    if torch.cuda.is_available():
        trainer.model.cuda()

    criterion = torch.nn.MSELoss()

    try:
        for epoch in range(1, args.num_epochs):
            logger.info("Start training")
            logger.info(f"Epoch {trainer.epoch} of {trainer.num_epochs}\n ----------------------------")

            if args.train:

                trainer.model.train()

                train_loss = 0.0
                train_psnr = 0.0

                for i, (train_in, train_out) in enumerate(trainer.train_data_loader):
                    inputs, outputs = train_in.to(device), train_out.to(device)

                    trainer.optimizer.zero_grad()

                    pred = trainer.model(inputs)
                    loss = criterion(pred, outputs)
                    predicted_time = time.time() - start_time

                    loss.backward()
                    trainer.optimizer.step()

                    train_loss += loss.item()

                    with torch.no_grad():  # PSNR Train
                        pred = (pred > threshold) * 1.0
                        pred_img = pred.detach().cpu().numpy()
                        gt_img = outputs.detach().cpu().numpy()

                        mse = np.mean((pred_img - gt_img) ** 2)
                        psnr = 100 if mse == 0 else (20 * math.log10(1.0 / math.sqrt(mse)))
                        train_psnr += psnr

                        # Prompt Logging
                        if i % 10 == 0:
                            elapsed_time = time.time() - start_time
                            logger.info(
                                f'Train Loss: {loss.item():0.6f} [{i * len(inputs)} / {len(trainer.train_data_loader) * len(inputs)}]')
                            logger.info(f'Time {elapsed_time:0.1f}, {predicted_time:0.1f}')

                avg_train_loss = train_loss / len(trainer.train_data_loader)
                avg_train_psnr = train_psnr / len(trainer.train_data_loader)

                if trainer.wandb:
                    logs = {
                        'train_avg_loss': avg_train_loss,
                        'train_avg_psnr': avg_train_psnr,
                        'train_total_psnr': train_psnr,
                    }
                    trainer.wandb.on_log(logs)

            # Validation
            trainer.model.eval()
            with torch.no_grad():
                psnr, valid_loss = trainer.validation()

                if psnr > trainer.best_psnr:
                    trainer.save_checkpoint(args.path_checkpoint)
                    trainer.best_psnr = psnr

            logger.info(f'Current Peak Signal To Noise: {psnr:0.6f} - Best: {trainer.best_psnr} ')
            logger.info(f'Valid Loss: {valid_loss}')

            trainer.epoch += 1
            trainer.wandb({'epoch': trainer.epoch})

    except KeyboardInterrupt:
        logger.error("Keyboard Interrupt. Stop training!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Required
    parser.add_argument('-name', '--experiment_name',
                        metavar='<name>',
                        type=str,
                        help=f"The experiment name which will use on WandB",
                        default="All Transforms")

    # Default values
    parser.add_argument('--train_data_path', type=str, default='patches/train')
    parser.add_argument('--valid_data_path', type=str, default='patches/valid')
    parser.add_argument('--train_batch_size', type=int, default=4)
    parser.add_argument('--valid_batch_size', type=int, default=4)
    parser.add_argument('--train_split_size', type=int, default=256)
    parser.add_argument('--valid_split_size', type=int, default=256)
    parser.add_argument('--input_channels', type=int, default=3)
    parser.add_argument('--output_channels', type=int, default=1)
    parser.add_argument('--learning_rate', type=float, default=1.5e-4)
    parser.add_argument('--workers', default=4, type=int)
    parser.add_argument('--num_epochs', type=int, default=150)
    parser.add_argument('--path_checkpoint', type=str, default='weights/')

    parser.add_argument('--train', type=bool, default=True)

    args = parser.parse_args()
    args.num_gpu = torch.cuda.device_count()
    train(args)
