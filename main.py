import os


from data.process_image import PatchImage, configure_args


def create_patches(path_configuration: str):
    args = configure_args(path_configuration)
    root_original = args.path_original
    root_ground_truth = args.path_ground_truth
    destination = args.path_destination
    patch_size = args.patch_size
    patch_size_valid = args.patch_size_valid
    overlap_size = args.overlap_size
    validation_dataset = args.validation_dataset
    testing_dataset = args.testing_dataset

    patcher = PatchImage(patch_size=patch_size,
                         patch_size_valid=patch_size_valid,
                         overlap_size=overlap_size,
                         destination_root=destination)
    patcher.create_patches(root_original=root_original,
                           root_ground_truth=root_ground_truth,
                           validation_dataset=validation_dataset,
                           test_dataset=testing_dataset)


if __name__ == '__main__':
    root_dir = os.path.dirname(os.path.abspath(__file__))
    path_config = os.path.join(root_dir, 'configs/create_patches.yaml')
    create_patches(path_config)
