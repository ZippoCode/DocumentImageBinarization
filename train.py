import argparse
import os
import random
import sys
import time
from datetime import timedelta

import numpy as np
import torch
import wandb
import yaml
from torchvision.transforms import functional

from trainer.LaMaTrainer import LaMaTrainingModule
from trainer.Validator import Validator
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
            "Architecture": "lama",
        }
        wandb_log.setup(**params)

    trainer = LaMaTrainingModule(config, device=device)
    if torch.cuda.is_available():
        trainer.model.cuda()

    if wandb_log:
        wandb_log.add_watch(trainer.model)

    validator = Validator()

    try:
        start_time = time.time()

        for epoch in range(1, config['num_epochs']):
            wandb_logs = dict()

            if config_args.train:
                logger.info("Start training") if epoch == 1 else None
                logger.info(f"Epoch {trainer.epoch} of {trainer.num_epochs}")

                train_loss = 0.0
                visualization = torch.zeros((1, config['train_patch_size'], config['train_patch_size']), device=device)
                training_images = []

                trainer.model.train()
                validator.reset()

                for batch_idx, (train_in, train_out) in enumerate(trainer.train_data_loader):
                    inputs, outputs = train_in.to(device), train_out.to(device)

                    trainer.optimizer.zero_grad()

                    predictions = trainer.model(inputs)
                    loss = trainer.criterion(predictions, outputs)

                    tensor_bin = torch.where(predictions > threshold, 1., 0.)
                    tensor_diff = torch.abs(tensor_bin - outputs)
                    visualization += torch.sum(tensor_diff, dim=0)

                    loss.backward()
                    trainer.optimizer.step()

                    train_loss += loss.item()

                    with torch.no_grad():
                        psnr, precision, recall = validator.compute(predictions, outputs)

                        if batch_idx % config['train_log_every'] == 0:
                            size = batch_idx * len(inputs)
                            percentage = 100. * size / len(trainer.train_dataset)

                            elapsed_time = time.time() - start_time
                            time_per_iter = elapsed_time / (size + 1)
                            remaining_time = (len(trainer.train_dataset) - size - 1) * time_per_iter
                            eta = str(timedelta(seconds=remaining_time))

                            stdout = f"Train Loss: {loss.item():.6f} - PSNR: {psnr:0.4f} -"
                            stdout += f" Precision: {precision:0.4f}% - Recall: {recall:0.4f}%"
                            stdout += f" \t[{size} / {len(trainer.train_dataset)}]"
                            stdout += f" ({percentage:.2f}%)  Epoch eta: {eta}"
                            logger.info(stdout)

                            for b in range(len(inputs)):
                                original = inputs[b]
                                pred = predictions[b].expand(3, -1, -1)
                                output = outputs[b].expand(3, -1, -1)
                                union = torch.cat((original, pred, output), 2)
                                training_images.append(wandb.Image(functional.to_pil_image(union), caption=f"Es. {b}"))

                avg_train_loss = train_loss / len(trainer.train_dataset)
                avg_train_psnr, avg_train_precision, avg_train_recall = validator.get_metrics()

                stdout = f"AVG training loss: {avg_train_loss:0.4f} - AVG training PSNR: {avg_train_psnr:0.4f}"
                stdout += f" AVG training precision: {avg_train_precision:0.4f}%"
                stdout += f" AVG training recall: {avg_train_recall:0.4f}%"
                logger.info(stdout)

                wandb_logs['train_avg_loss'] = avg_train_loss
                wandb_logs['train_avg_psnr'] = avg_train_psnr
                wandb_logs['train_avg_precision'] = avg_train_precision
                wandb_logs['train_avg_recall'] = avg_train_recall
                wandb_logs['Random Sample'] = random.choices(training_images, k=5)

                # Make error images
                rescaled = torch.div(visualization, config['train_max_value'])
                rescaled = torch.clamp(rescaled, min=0, max=1)
                wandb_logs['Errors'] = wandb.Image(functional.to_pil_image(rescaled))

                # Validation
                trainer.model.eval()
                validator.reset()

                with torch.no_grad():
                    valid_psnr, valid_precision, valid_recall, valid_loss, images = trainer.validation()

                    name_image, (valid_img, pred_img, gt_valid_img) = list(images.items())[0]

                    wandb_logs['valid_avg_loss'] = valid_loss
                    wandb_logs['valid_avg_psnr'] = valid_psnr
                    wandb_logs['valid_avg_precision'] = valid_precision
                    wandb_logs['valid_avg_recall'] = valid_recall

                    wandb_logs['Results'] = [wandb.Image(valid_img, caption=f"Sample: {name_image}"),
                                             wandb.Image(pred_img, caption=f"Predicted Sample: {name_image}"),
                                             wandb.Image(gt_valid_img, caption=f"Ground Truth Sample: {name_image}")]

                    if valid_psnr > trainer.best_psnr:
                        trainer.best_psnr = valid_psnr
                    trainer.best_precision = valid_precision
                    trainer.best_recall = valid_recall

                    trainer.save_checkpoints(root_folder=config['path_checkpoint'],
                                             filename=config_args.experiment_name)

                    # Save images
                    folder = f'results/training/{config_args.experiment_name}/'
                    os.makedirs(folder, exist_ok=True)
                    for name_image, (_, predicted_image, _) in images.items():
                        path = folder + name_image
                    predicted_image.save(path)
                    logger.info("Stored predicted images")

                # Log best values
                wandb_logs['Best PSNR'] = trainer.best_psnr
                wandb_logs['Best Precision'] = trainer.best_precision
                wandb_logs['Best Recall'] = trainer.best_recall

                stdout = f"Validation Loss: {valid_loss:.4f} - PSNR: {valid_psnr:.4f}"
                stdout += f" Precision: {valid_precision:.4f}% - Recall: {valid_recall:.4f}%"
                stdout += f" Best Loss: {trainer.best_psnr:.3f}"
                logger.info(stdout)

                trainer.epoch += 1
                wandb_logs['epoch'] = trainer.epoch
                logger.info('-' * 75)

                if wandb_log:
                    wandb_log.on_log(wandb_logs)

    except KeyboardInterrupt:
        logger.warning("Training interrupted by user")
    except Exception as e:
        logger.error(f"Training failed due to {e}")


def random_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-en', '--experiment_name', metavar='<name>', type=str,
                        help=f"The experiment name which will use on WandB", default="debug")
    parser.add_argument('-cfg', '--configuration', metavar='<name>', type=str,
                        help=f"The configuration name will use on WandB", default="debug")
    parser.add_argument('-wdb', '--use_wandb', type=bool, default=not DEBUG)
    parser.add_argument('-t', '--train', type=bool, default=True)

    args = parser.parse_args()
    config_filename = args.configuration

    logger.info("Start process ...")
    configuration_path = f"configs/training/{config_filename}.yaml"
    logger.info(f"Selected \"{configuration_path}\" configuration file")

    with open(configuration_path) as file:
        train_config = yaml.load(file, Loader=yaml.Loader)
        file.close()

    random_seed(train_config['seed'])

    train(args, train_config)
    sys.exit()
