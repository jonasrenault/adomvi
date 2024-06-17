from pathlib import Path
import shutil

def copy_dataset_structure(source_dir: Path, target_dir: Path):
    """
    Copy images and labels from the source directory to the target directory.

    Args:
        source_dir (Path): The source directory containing the dataset.
        target_dir (Path): The target directory to copy the dataset into.
    """
    
    # Define the mapping from source subdirectories to target subdirectories
    dir_mapping = {
        'test/images': 'images/test',
        'test/labels': 'labels/test',
        'train/images': 'images/train',
        'train/labels': 'labels/train',
        'valid/images': 'images/val',
        'valid/labels': 'labels/val'
    }

    # Create target subdirectories if they don't exist
    for target_subdir in dir_mapping.values():
        (target_dir / target_subdir).mkdir(parents=True, exist_ok=True)
    
    # Copy files from source to target
    for source_subdir, target_subdir in dir_mapping.items():
        source_path = source_dir / source_subdir
        target_path = target_dir / target_subdir
        for file in source_path.glob('*'):
            shutil.copy(file, target_path)
            print(f"Copied {file} to {target_path}")

