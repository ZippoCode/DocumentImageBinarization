import wandb


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

    def __init__(self, experiment_name: str, project="test-project", entity="fomo_thesis"):
        self._wandb = wandb
        self._initialized = False
        self._project = project
        self._entity = entity
        self._experiment_name = experiment_name
        self._dir = '/tmp'

    def setup(self, **kwargs):
        if self._wandb is None:
            return
        self._initialized = True

        # Configuration
        if self._wandb.run is None:
            self._wandb.init(project=self._project, entity=self._entity, name=self._experiment_name, dir=self._dir,
                             config={**kwargs})

    def add_watch(self, model):
        self._wandb.watch(model, log="all")

    def on_log(self, logs=None):
        logs = rewrite_logs(logs)
        self._wandb.log(logs)
