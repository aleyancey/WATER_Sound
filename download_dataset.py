import os
import logging
from huggingface_hub import snapshot_download
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def download_dataset(dataset_dir):
    """
    Download the Sound of Water dataset using snapshot_download
    """
    try:
        logger.info("Starting dataset download...")
        logger.info("This may take 5-10 minutes depending on your internet connection.")
        
        # Create necessary directories
        subdirs = ['annotations', 'assets', 'audios', 'splits', 'videos', 'youtube_samples']
        for subdir in subdirs:
            os.makedirs(os.path.join(dataset_dir, subdir), exist_ok=True)
        
        # Download the dataset
        snapshot_download(
            repo_id="bpiyush/sound-of-water",
            repo_type="dataset",
            local_dir=dataset_dir,
            local_dir_use_symlinks=False,
            max_workers=4  # Reduce concurrent downloads to avoid rate limiting
        )
        
        logger.info("Dataset download completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"Failed to download dataset: {str(e)}")
        return False

def main():
    # Set up the dataset directory
    dataset_dir = "data/raw/sound_of_water"
    os.makedirs(dataset_dir, exist_ok=True)
    
    # Download the dataset
    success = download_dataset(dataset_dir)
    
    if not success:
        logger.error("Dataset download failed")
        return
    
    # Print the dataset structure
    print("\nDataset structure:")
    for root, dirs, files in os.walk(dataset_dir):
        level = root.replace(dataset_dir, '').count(os.sep)
        indent = '│   ' * (level - 1) + '├── ' if level > 0 else ''
        print(f"{indent}{os.path.basename(root)}/")
        for file in files:
            if not file.startswith('.'):  # Skip hidden files
                indent = '│   ' * level + '├── '
                print(f"{indent}{file}")

if __name__ == "__main__":
    main() 