import argparse
import os
import sys
import time

import torch
import yaml

from trainer.LaMaTrainer import LaMaTrainingModule, calculate_psnr
from utils.WandbLog import WandbLog
from utils.htr_logging import get_logger

logger = get_logger('main')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"Using {device} device")
threshold = 0.5


def train(args, config):
    start_time = time.time()

    # Configure WandB
    wandb_log = None
    if args.use_wandb:
        wandb_log = WandbLog(experiment_name=args.experiment_name)
        params = {
            "Learning Rate": config['learning_rate'],
            "Epochs": config['num_epochs'],
            "Train Batch Size": config['train_batch_size'],
            "Valid Batch Size": config['valid_batch_size'],
            "Train Split Size": config['train_split_size'],
            "Valid Split Size": config['valid_split_size'],
            "Architecture": "lama",
        }
        wandb_log.setup(**params)

    trainer = LaMaTrainingModule(config, device=device, wandb_log=wandb_log)
    if torch.cuda.is_available():
        trainer.model.cuda()

    if wandb_log:
        wandb_log.add_watch(trainer.model)

    try:

        for epoch in range(1, config['num_epochs']):
            num_images = 0

            if args.train:
                logger.info("Start training") if epoch == 1 else None
                logger.info(f"Epoch {trainer.epoch} of {trainer.num_epochs}")

                train_loss = 0.0
                train_psnr = 0.0

                trainer.model.train()

                for i, (train_in, train_out) in enumerate(trainer.train_data_loader):
                    inputs, outputs = train_in.to(device), train_out.to(device)

                    trainer.optimizer.zero_grad()

                    pred = trainer.model(inputs)
                    loss = trainer.criterion(pred, outputs)

                    predicted_time = time.time() - start_time

                    loss.backward()
                    trainer.optimizer.step()

                    train_loss += loss.item()
                    num_images += len(inputs)

                    # trainer.model.eval()
                    with torch.no_grad():  # PSNR Train
                        psnr = calculate_psnr(predicted=pred, ground_truth=outputs, threshold=threshold)
                        train_psnr += psnr
                        # trainer.model.training()

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
                valid_psnr, valid_loss, images = trainer.validation()

                logger.info("Validation info:")
                logger.info(f"\tAverage Loss: {valid_loss:0.6f} - Average PSNR: {valid_psnr:0.6f}")

                if args.store and valid_psnr > trainer.best_psnr:
                    trainer.best_psnr = valid_psnr
                    logger.info(f"\tBest Loss: {trainer.best_psnr:0.6f}")
                    trainer.save_checkpoints(root_folder=config['path_checkpoint'], filename=args.experiment_name)

                    if wandb_log:
                        logs = {
                            'Best PSNR': trainer.best_psnr,
                            'valid_avg_loss': valid_loss,
                            'valid_avg_psnr': valid_psnr,
                        }
                        wandb_log.on_log(logs)

                    # Save images
                    folder = f'results/training{args.experiment_name}/'
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
    parser.add_argument('--use_wandb', type=bool, default=True)
    parser.add_argument('--train', type=bool, default=True)
    parser.add_argument('--store', type=bool, default=True)

    args = parser.parse_args()
    config_filename = args.experiment_name

    logger.info("Start process ...")
    configuration_path = f"configs/training/{config_filename}.yaml"
    logger.info(f"Selected \"{configuration_path}\" configuration file")

    with open(configuration_path) as file:
        train_config = yaml.load(file, Loader=yaml.Loader)
        file.close()

    train(args, train_config)
    sys.exit()
