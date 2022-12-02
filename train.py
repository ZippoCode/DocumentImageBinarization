import argparse
import os
import sys
import time

import torch

from trainer.LaMaTrainer import LaMaTrainingModule, calculate_psnr
from utils.WandbLog import WandbLog
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


class LMSELoss(torch.nn.MSELoss):
    def forward(self, inputs, targets):
        mse = super().forward(inputs, targets)
        mse = torch.add(mse, 1e-10)
        return torch.log10(mse)


def train(args):
    start_time = time.time()

    if args.experiment_name == 'mse_loss':
        criterion = torch.nn.MSELoss().to(device)
    else:
        criterion = LMSELoss().to(device)

    # Configure WandB
    wandb_log = None
    if not debug:
        experiment_name = args.experiment_name + "_lr_" + str(args.learning_rate)
        wandb_log = WandbLog(experiment_name=experiment_name)
        params = {
            "learning_rate": args.learning_rate,
            "epochs": args.num_epochs,
            "train_batch_size": args.train_batch_size,
            "valid_batch_size": args.valid_batch_size,
            "architecture": "lama"
        }
        wandb_log.setup(**params)

    trainer = LaMaTrainingModule(train_data_path=args.train_data_path, train_gt_data_path=args.train_gt_data_path,
                                 valid_data_path=args.valid_data_path, valid_gt_data_path=args.valid_gt_data_path,
                                 input_channels=args.input_channels, output_channels=args.output_channels,
                                 train_batch_size=args.train_batch_size, valid_batch_size=args.valid_batch_size,
                                 train_split_size=args.train_split_size, valid_split_size=args.valid_split_size,
                                 workers=args.workers, epochs=args.num_epochs, learning_rate=args.learning_rate,
                                 debug=debug, device=device, criterion=criterion, wandb_log=wandb_log,
                                 experiment_name=args.experiment_name)
    if wandb_log:
        wandb_log.add_watch(trainer.model)

    if torch.cuda.is_available():
        trainer.model.cuda()

    try:
        logger.info("Start training")

        for epoch in range(1, args.num_epochs):

            num_images = 0

            if args.train:
                logger.info(f"Epoch {trainer.epoch} of {trainer.num_epochs}\n ----------------------------")

                train_loss = 0.0
                train_psnr = 0.0

                trainer.model.train()

                for i, (train_in, train_out) in enumerate(trainer.train_data_loader):
                    inputs, outputs = train_in.to(device), train_out.to(device)

                    trainer.optimizer.zero_grad()

                    pred = trainer.model(inputs)
                    loss = criterion(pred, outputs)

                    predicted_time = time.time() - start_time

                    loss.backward()
                    trainer.optimizer.step()

                    train_loss += loss.item()
                    num_images += len(inputs)

                    # trainer.model.eval()
                    with torch.no_grad():  # PSNR Train
                        psnr = calculate_psnr(predicted=pred, ground_truth=outputs, threshold=threshold)
                        train_psnr += psnr
                        # trainer.model.train()

                    if i % 100 == 0:
                        elapsed_time = time.time() - start_time
                        size = i * len(inputs)
                        all_size = len(trainer.train_data_loader) * len(inputs)
                        logger.info(
                            f'Train Loss: {loss.item():0.6f} [{size} / {all_size}] - Epoch: {trainer.epoch + 1}')
                        logger.info(f'Time {elapsed_time:0.1f}, {predicted_time:0.1f}')
                        logger.info(f"BATCH shape: {inputs.shape}")

                avg_train_loss = train_loss / num_images
                avg_train_psnr = train_psnr / num_images

                if wandb_log:
                    logs = {
                        'train_avg_loss': avg_train_loss,
                        'train_avg_psnr': avg_train_psnr,
                    }
                    wandb_log.on_log(logs)

            # Validation
            trainer.model.eval()
            with torch.no_grad():
                psnr, valid_loss, images = trainer.validation()

                if psnr > trainer.best_psnr:
                    trainer.save_checkpoints(args.path_checkpoint)
                    trainer.best_psnr = psnr

                    # Save images
                    folder = 'results/' + args.experiment_name + '/'
                    os.makedirs(folder, exist_ok=True)
                    for name_image, predicted_image in images.items():
                        path = folder + name_image
                        predicted_image.save(path)
                    logger.info("Store predicted images")

            trainer.epoch += 1
            if wandb_log:
                wandb_log.on_log({'epoch': trainer.epoch})

    except KeyboardInterrupt:
        logger.warning("Training interrupted by user")
    except Exception as e:
        logger.error(f"Training failed due to {e}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-name', '--experiment_name', metavar='<name>', type=str,
                        help=f"The experiment name which will use on WandB", default="debug")
    parser.add_argument('-lr', '--learning_rate', metavar="<float>", help="The learning rate for the experiment",
                        type=float, default=1.5e-4)

    # Default values
    parser.add_argument('--train_data_path', type=str, default='patches/train')
    parser.add_argument('--train_gt_data_path', type=str, default='patches/train_gt')
    parser.add_argument('--valid_data_path', type=str, default='patches/valid/full')
    parser.add_argument('--valid_gt_data_path', type=str, default='patches/valid_gt/full')
    parser.add_argument('--train_batch_size', type=int, default=4)
    parser.add_argument('--valid_batch_size', type=int, default=1)
    parser.add_argument('--train_split_size', type=int, default=256)
    parser.add_argument('--valid_split_size', type=int, default=512)
    parser.add_argument('--input_channels', type=int, default=3)
    parser.add_argument('--output_channels', type=int, default=1)
    parser.add_argument('--workers', default=1, type=int)
    parser.add_argument('--num_epochs', type=int, default=150)
    parser.add_argument('--path_checkpoint', type=str, default='weights/')

    parser.add_argument('--train', type=bool, default=True)

    args = parser.parse_args()
    args.num_gpu = torch.cuda.device_count()

    train(args)
