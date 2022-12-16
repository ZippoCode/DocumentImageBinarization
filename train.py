import argparse
import os
import sys
import time
from datetime import timedelta

import torch
import wandb
import yaml
from torchvision.transforms import functional

from trainer.LaMaTrainer import LaMaTrainingModule, calculate_psnr
from utils.WandbLog import WandbLog
from utils.htr_logging import get_logger, DEBUG

logger = get_logger('main')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"Using {device} device")
threshold = 0.5


def train(config_args, config):
    wandb_log = None
    if config_args.use_wandb:  # Configure WandB
        wandb_log = WandbLog(experiment_name=config_args.experiment_name)
        params = {
            "Learning Rate": config['learning_rate'],
            "Epochs": config['num_epochs'],
            "Train Batch Size": config['train_batch_size'],
            "Valid Batch Size": config['valid_batch_size'],
            "Train Patch Size": config['train_patch_size'],
            "Valid Patch Size": config['valid_patch_size'],
            "Valid Stride": config['valid_stride'],
            # TO BE DELETED
            "Max Value": args.max_value,

            "Architecture": "lama",
        }
        wandb_log.setup(**params)

    trainer = LaMaTrainingModule(config, device=device)
    if torch.cuda.is_available():
        trainer.model.cuda()

    if wandb_log:
        wandb_log.add_watch(trainer.model)

    try:
        start_time = time.time()

        for epoch in range(1, config['num_epochs']):
            wandb_logs = dict()

            if config_args.train:
                logger.info("Start training") if epoch == 1 else None
                logger.info(f"Epoch {trainer.epoch} of {trainer.num_epochs}")

                train_loss = 0.0
                train_psnr = 0.0
                visualization = torch.zeros((1, config['train_patch_size'], config['train_patch_size']), device=device)

                trainer.model.train()

                for batch_idx, (train_in, train_out) in enumerate(trainer.train_data_loader):
                    inputs, outputs = train_in.to(device), train_out.to(device)

                    trainer.optimizer.zero_grad()

                    pred = trainer.model(inputs)
                    loss = trainer.criterion(pred, outputs)

                    tensor_bin = torch.where(pred > threshold, 1., 0.)
                    tensor_diff = torch.abs(tensor_bin - outputs)
                    visualization += torch.sum(tensor_diff, dim=0)

                    loss.backward()
                    trainer.optimizer.step()

                    train_loss += loss.item()

                    with torch.no_grad():
                        psnr = calculate_psnr(predicted=pred, ground_truth=outputs, threshold=threshold)
                        train_psnr += psnr

                        if batch_idx % config['train_log_every'] == 0:
                            size = batch_idx * len(inputs)
                            percentage = 100. * size / len(trainer.train_dataset)

                            elapsed_time = time.time() - start_time
                            time_per_iter = elapsed_time / (size + 1)
                            remaining_time = (len(trainer.train_dataset) - size - 1) * time_per_iter
                            eta = str(timedelta(seconds=remaining_time))

                            stdout = f"Train Loss: {loss.item():.6f} [{size} / {len(trainer.train_dataset)}]"
                            stdout += f" ({percentage:.2f}%)  Epoch eta: {eta}"
                            logger.info(stdout)

                avg_train_loss = train_loss / len(trainer.train_dataset)
                avg_train_psnr = train_psnr / len(trainer.train_dataset)

                stdout = f"AVG training loss: {avg_train_loss:0.4f} - AVG training PSNR: {avg_train_psnr:0.4f}"
                logger.info(stdout)

                # Make error images
                rescaled = torch.div(visualization, config_args.max_value)
                rescaled = torch.clamp(rescaled, min=0, max=1)
                rescaled_img = functional.to_pil_image(rescaled)

                wandb_logs['train_avg_loss'] = avg_train_loss
                wandb_logs['train_avg_psnr'] = avg_train_psnr
                wandb_logs['Errors'] = wandb.Image(rescaled_img)

            # Validation
            trainer.model.eval()
            with torch.no_grad():
                valid_psnr, valid_loss, images = trainer.validation()

                wandb_logs['valid_avg_loss'] = valid_loss
                wandb_logs['valid_avg_psnr'] = valid_psnr

                name_image, (valid_img, pred_img, gt_valid_img) = list(images.items())[0]
                name_image = name_image
                wandb_images = [wandb.Image(valid_img, caption=f"Sample: {name_image}"),
                                wandb.Image(pred_img, caption=f"Predicted Sample: {name_image}"),
                                wandb.Image(gt_valid_img, caption=f"Ground Truth Sample: {name_image}")]
                wandb_logs['Results'] = wandb_images
                wandb_logs['Best PSNR'] = trainer.best_psnr

                if valid_psnr > trainer.best_psnr:
                    trainer.best_psnr = valid_psnr
                    wandb_logs['Best PSNR'] = trainer.best_psnr

                    trainer.save_checkpoints(root_folder=config['path_checkpoint'],
                                             filename=config_args.experiment_name)

                    # Save images
                    folder = f'results/training/{config_args.experiment_name}/'
                    os.makedirs(folder, exist_ok=True)
                    for name_image, (_, predicted_image, _) in images.items():
                        path = folder + name_image
                        predicted_image.save(path)
                    logger.info("Stored predicted images")

                stdout = f"AVG Validation Loss: {valid_loss:.4f} - AVG Validation PSNR: {valid_psnr:.4f}"
                stdout += f" Best Loss: {trainer.best_psnr:.3f}"
                logger.info(stdout)

            trainer.epoch += 1
            wandb_logs['epoch'] = trainer.epoch
            logger.info('-' * 25)

            if wandb_log:
                wandb_log.on_log(wandb_logs)

    except KeyboardInterrupt:
        logger.warning("Training interrupted by user")
    except Exception as e:
        logger.error(f"Training failed due to {e}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-name', '--experiment_name', metavar='<name>', type=str,
                        help=f"The experiment name which will use on WandB", default="debug")
    parser.add_argument('--use_wandb', type=bool, default=not DEBUG)
    parser.add_argument('--train', type=bool, default=True)
    parser.add_argument('--max_value', type=int, default=100)

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
