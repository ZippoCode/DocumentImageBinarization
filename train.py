import argparse
import os
import random
import sys
import time
from datetime import timedelta

import numpy as np
import torch
import wandb
from torchvision.transforms import functional

from trainer.LaMaTrainer import LaMaTrainingModule
from utils.WandbLog import WandbLog
from utils.htr_logging import get_logger, DEBUG
from utils.ioutils import store_images, read_yaml
from utils.metrics import calculate_psnr

logger = get_logger(os.path.basename(__file__))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def random_seed(config: dict):
    if 'seed' in config:
        seed = config['seed']
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        logger.info(f"Configured random seed with value: {seed}")


def parser_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('-en', '--experiment_name', metavar='<name>', type=str,
                        help=f"The experiment name which will use on WandB", default="debug")
    parser.add_argument('-rt', '--resume_training', metavar='<bool>', type=bool,
                        help=f"Set if you want resume the checkpoint", default=True)
    parser.add_argument('-cfg', '--configuration', metavar='<name>', type=str,
                        help=f"The configuration name will use on WandB", default="configs/training/debug.yaml")
    parser.add_argument('-net_cfg', '--network_configuration', metavar='<name>', type=str,
                        help=f"The filename will be used to configure the network",
                        default="configs/network/network_blocks_9.yaml")
    parser.add_argument('-wdb', '--use_wandb', metavar='<value>', type=bool,
                        help=f"If TRUE the training will use WandDB to show logs.", default=not DEBUG)
    parser.add_argument('-tn', '--train_network', metavar='<value>', type=bool,
                        help=f"If TRUE the network will be trained also only validate.", default=True)
    parser.add_argument('-pt', '--patience_time', metavar='<value>', type=int,
                        help=f"The number of epochs after which the PSNR value must be updated before stopping",
                        default=30)

    return parser.parse_args()


if __name__ == '__main__':
    try:
        args = parser_arguments()
        experiment_name = args.experiment_name
        resume_training = args.resume_training
        configuration_path = args.configuration
        network_configuration_path = args.network_configuration
        use_wandb = args.use_wandb
        train_network = args.train_network
        patience = args.patience_time

        if use_wandb and train_network:
            wandb_log = WandbLog(experiment_name=experiment_name, resume=resume_training)
        else:
            wandb_log = None

        logger.info("Start process ...")
        logger.info(f"Using {device} device")

        train_config = read_yaml(configuration_path)
        network_config = read_yaml(network_configuration_path)

        if wandb_log:
            params = {
                "Architecture": "lama",

                "Learning Rate": train_config['learning_rate'],
                "Epochs": train_config['num_epochs'],

                "Train Batch Size": train_config['train_batch_size'],
                "Valid Batch Size": train_config['valid_batch_size'],
                "Train Patch Size": train_config['train_patch_size'],
                "Valid Patch Size": train_config['valid_patch_size'],
                "Valid Stride": train_config['valid_stride'],

                "Loss": train_config['kind_loss'],
                "Optimizer": train_config['kind_optimizer'],
                "Transform Variant": train_config['train_transform_variant'],
            }
            wandb_log.setup(**params)

        random_seed(train_config)

        trainer = LaMaTrainingModule(train_config, network_config, device=device)
        if torch.cuda.is_available():
            trainer.model.cuda()

        if resume_training:
            trainer.resume_checkpoints(folder=train_config['path_checkpoint'], filename=experiment_name)

        if wandb_log:
            wandb_log.add_watch(trainer.model)

        start_time = time.time()

        for epoch in range(1, train_config['num_epochs']):
            wandb_logs = dict()

            if train_network:
                logger.info("Training has been started") if epoch == 1 else None
                logger.info(f"Epoch {trainer.epoch} of {trainer.num_epochs}")

                train_loss = 0.
                train_psnr = 0.
                training_images = []

                trainer.model.train()

                for batch_idx, (train_in, train_out) in enumerate(trainer.train_data_loader):
                    inputs, outputs = train_in.to(device), train_out.to(device)

                    trainer.optimizer.zero_grad()
                    predictions = trainer.model(inputs)
                    loss = trainer.criterion(predictions, outputs)
                    loss.backward()
                    trainer.optimizer.step()
                    train_loss += loss.item()

                    with torch.no_grad():
                        psnr = calculate_psnr(predictions, outputs)
                        train_psnr += psnr

                        if batch_idx % train_config['train_log_every'] == 0:
                            size = batch_idx * len(inputs)
                            percentage = 100. * size / len(trainer.train_dataset)

                            elapsed_time = time.time() - start_time
                            time_per_iter = elapsed_time / (size + 1)
                            remaining_time = (len(trainer.train_dataset) - size - 1) * time_per_iter
                            eta = str(timedelta(seconds=remaining_time))

                            stdout = f"Train Loss: {loss.item():.6f} - PSNR: {psnr:0.4f} -"
                            stdout += f" \t[{size} / {len(trainer.train_dataset)}]"
                            stdout += f" ({percentage:.2f}%)  Epoch eta: {eta}"
                            logger.info(stdout)

                            for b in range(len(inputs)):
                                original = inputs[b]
                                pred = predictions[b].expand(3, -1, -1)
                                output = outputs[b].expand(3, -1, -1)
                                union = torch.cat((original, pred, output), 2)
                                training_images.append(wandb.Image(functional.to_pil_image(union), caption=f"Example"))

                avg_train_loss = train_loss / len(trainer.train_dataset)
                avg_train_psnr = train_psnr / len(trainer.train_dataset)

                stdout = f"AVG training loss: {avg_train_loss:0.4f} - AVG training PSNR: {avg_train_psnr:0.4f}"
                logger.info(stdout)

                wandb_logs['train_avg_loss'] = avg_train_loss
                wandb_logs['train_avg_psnr'] = avg_train_psnr
                wandb_logs['Random Sample'] = random.choices(training_images, k=5)

            # Validation
            trainer.model.eval()

            with torch.no_grad():
                valid_psnr, valid_loss, images = trainer.validation()

                wandb_logs['valid_avg_loss'] = valid_loss
                wandb_logs['valid_avg_psnr'] = valid_psnr

                name_image, (valid_img, pred_img, gt_valid_img) = list(images.items())[0]
                wandb_logs['Results'] = [wandb.Image(valid_img, caption=f"Sample: {name_image}"),
                                         wandb.Image(pred_img, caption=f"Predicted Sample: {name_image}"),
                                         wandb.Image(gt_valid_img, caption=f"Ground Truth Sample: {name_image}")]

                if valid_psnr > trainer.best_psnr:
                    patience = args.patience_time
                    trainer.best_psnr = valid_psnr

                    trainer.save_checkpoints(root_folder=train_config['path_checkpoint'],
                                             filename=experiment_name)

                    # Save images
                    names = images.keys()
                    predicted_images = [item[1] for item in list(images.values())]
                    store_images(parent_directory='results/training', directory=experiment_name,
                                 names=names, images=predicted_images)
                else:
                    patience -= 1

            # Log best values
            wandb_logs['Best PSNR'] = trainer.best_psnr

            stdout = f"Validation Loss: {valid_loss:.4f} - PSNR: {valid_psnr:.4f}"
            stdout += f" Best Loss: {trainer.best_psnr:.3f} \t Patience: [{patience}/{args.patience_time}]"
            logger.info(stdout)

            trainer.epoch += 1
            wandb_logs['epoch'] = trainer.epoch
            logger.info('-' * 75)

            if use_wandb:
                wandb_log.on_log(wandb_logs)

            if patience == 0 or not trainer:
                stdout = "There has been no update of Best PSNR value in the last 30 epochs."
                stdout += " Training will be stopped."
                logger.info(stdout)
                sys.exit()

    except KeyboardInterrupt:
        logger.warning("Training interrupted by user")
    except Exception as e:
        logger.error(f"Training failed due to {e}")
    finally:
        sys.exit()
