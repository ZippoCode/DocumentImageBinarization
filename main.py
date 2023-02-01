import argparse
import os

import yaml

from data.create_training_patches import PatchImage

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-year', '--validation_year',
                        metavar='<path>',
                        type=str,
                        help=f"destination folder path with contains the patches",
                        default="2018")
    parser.add_argument('-dst', '--path_destination',
                        metavar='<path>',
                        type=str,
                        help=f"destination folder path with contains the patches",
                        default="patches")
    parser.add_argument('-cfg', '--configuration',
                        metavar='<filename>',
                        type=str,
                        help=f"destination folder path with contains the patches",
                        default="configs/create_patches.yaml")
    args = parser.parse_args()

    root_dir = os.path.dirname(os.path.abspath(__file__))
    path_configuration = os.path.join(root_dir, args.configuration)
    destination_path = f"{args.path_destination}/{args.validation_year}"

    with open(path_configuration) as file:
        config_options = yaml.load(file, Loader=yaml.Loader)
        file.close()

    patcher = PatchImage(config_options=config_options, destination_root=destination_path,
                         year_validation=args.validation_year)
    patcher.create_patches()
