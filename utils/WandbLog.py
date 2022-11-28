import torch
import wandb

import torchvision.transforms.functional as functional


def rewrite_logs(dictionary: dict):
    new_dictionary = {}
    train_prefix, valid_prefix, test_prefix = 'train_', 'valid_', 'test_'
    train_prefix_len = len(train_prefix)
    valid_prefix_len = len(valid_prefix)
    test_prefix_len = len(test_prefix)
    for key, value in dictionary.items():
        if key.startswith(train_prefix):
            new_dictionary["train/" + key[train_prefix_len:]] = value
        elif key.startswith(valid_prefix):
            new_dictionary["valid/" + key[valid_prefix_len:]] = value
        elif key.startswith(test_prefix):
            new_dictionary["test/" + key[test_prefix_len:]] = value
        else:
            new_dictionary[key] = value
    return new_dictionary


class WandbLog(object):

    def __init__(self):
        self._wandb = wandb
        self._initialized = False
        self._project = "test-project"
        self._entity = "fomo_thesis"

    def setup(self, model, **kwargs):
        if self._wandb is None:
            return
        self._initialized = True

        # Configuration
        if self._wandb.run is None:
            name = kwargs['experiment_name'] if kwargs['experiment_name'] else "train_lama_htr"
            self._wandb.init(project=self._project, entity=self._entity, name=name, config={**kwargs})
        self._wandb.config = {**kwargs}
        self._wandb.config.update()
        self._wandb.watch(model, log="all")

    def on_log(self, logs=None):
        logs = rewrite_logs(logs)
        self._wandb.log(logs)

    def on_log_images(self, sample: torch.Tensor, predicted_sample: torch.Tensor, gt_sample: torch.Tensor):
        num_batches = len(sample)

        for b in range(num_batches):
            valid_img = sample[b].detach()
            pred_img = predicted_sample[b].detach()
            gt_valid_img = gt_sample[b].detach()

            valid_img = functional.to_pil_image(valid_img)
            pred_img = functional.to_pil_image(pred_img)
            gt_valid_img = functional.to_pil_image(gt_valid_img)

            wandb_images = [wandb.Image(valid_img, caption='Sample'), wandb.Image(pred_img, caption='Predicted Sample'),
                            wandb.Image(gt_valid_img, caption='Ground Truth Sample')]

            logs = {
                "Images": wandb_images
            }
            self._wandb.log(logs)
