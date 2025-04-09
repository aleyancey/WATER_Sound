import os
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def fix_extensions(directory: str = "data/raw/rain_sounds"):
    """
    Fix incorrect file extensions in the given directory.
    
    Args:
        directory: Root directory to process
    """
    root_dir = Path(directory)
    
    # Count of files processed
    fixed_count = 0
    error_count = 0
    
    # Walk through all directories
    for root, _, files in os.walk(root_dir):
        for filename in files:
            if filename.endswith('.mp3') and ('.wav.' in filename or '.flac.' in filename or '.WAV.' in filename):
                old_path = Path(root) / filename
                
                # Get the correct extension
                if '.wav.' in filename.lower():
                    new_filename = filename.replace('.wav.mp3', '.wav').replace('.WAV.mp3', '.wav')
                elif '.flac.' in filename:
                    new_filename = filename.replace('.flac.mp3', '.flac')
                else:
                    continue
                    
                new_path = Path(root) / new_filename
                
                try:
                    old_path.rename(new_path)
                    logger.info(f"Fixed extension: {old_path} -> {new_path}")
                    fixed_count += 1
                except Exception as e:
                    logger.error(f"Failed to rename {old_path}: {str(e)}")
                    error_count += 1
    
    logger.info(f"Fixed {fixed_count} files")
    if error_count > 0:
        logger.warning(f"Failed to fix {error_count} files")

def main():
    """Main entry point for fixing file extensions."""
    try:
        fix_extensions()
    except Exception as e:
        logger.error(f"Failed to fix extensions: {str(e)}")
        raise

if __name__ == "__main__":
    main() 