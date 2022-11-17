import argparse
import logging
import numpy as np
import time
import torch
import wandb
import cv2

from trainer.LaMaTrainer import LaMaTrainingModule

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)


def train(args):
    start_time = time.time()
    trainer = LaMaTrainingModule(args.train_data_path, args.valid_data_path, args.input_channels, args.output_channels,
                                 args.batch_size, args.epochs)
    criterion = torch.nn.MSELoss()
    if torch.cuda.is_available():
        trainer.model.cuda()

    # trainer.resume(args.path_checkpoint)
    trainer.model.train()

    try:

        for epoch in range(1, args.epochs):

            train_loss = 0.0

            for i, (train_in, train_out) in enumerate(trainer.train_data_loader):
                elapsed_time = time.time() - start_time

                inputs = train_in.to(device)
                outputs = train_out.to(device)

                trainer.optimizer.zero_grad()

                predictions = trainer.model(inputs)
                loss = criterion(predictions, outputs)
                predicted_time = time.time() - start_time

                loss.backward()
                trainer.optimizer.step()

                train_loss += loss.item()

                # Build image
                train_img = np.swapaxes(inputs[0].cpu().detach().numpy(), 2, 0)
                pred_img = np.swapaxes(predictions[0].cpu().detach().numpy(), 2, 0)
                pred_img = cv2.cvtColor(pred_img, cv2.COLOR_GRAY2RGB)
                gt_img = np.swapaxes(outputs[0].cpu().detach().numpy(), 2, 0)
                gt_img = cv2.cvtColor(gt_img, cv2.COLOR_GRAY2RGB)

                if i % 10 == 0:
                    logging.info(f'i: {i} - epoch: {epoch}')
                    logging.info(f'Loss: {train_loss:0.6f} time {elapsed_time:0.1f}, {predicted_time:0.1f}')
                    trainer.save_checkpoint(args.path_checkpoint)

                    trainer.wandb.log({
                        'iter': i,
                        'epoch': trainer.epoch,
                        'train_loss': loss.item(),
                        'timeperepoch': predicted_time,
                        "images": wandb.Image(cv2.hconcat([train_img, pred_img, gt_img]), caption="Top: Example")
                    })
            trainer.epoch += 1


    except KeyboardInterrupt:
        logging.info("Keyboard Interrupt. Stop training!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data_path', type=str, default='patches/train')
    parser.add_argument('--valid_data_path', type=str, default='patches/valid')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--input_channels', type=int, default=3)
    parser.add_argument('--output_channels', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--path_checkpoint', type=str, default='weights/')

    args = parser.parse_args()
    train(args)
