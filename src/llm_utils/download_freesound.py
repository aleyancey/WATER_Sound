import os
import time
import random
from freesound import FreesoundClient
from dotenv import load_dotenv
from requests.exceptions import RequestException

# Load environment variables
load_dotenv()

# Initialize Freesound client
fs = FreesoundClient()
fs.set_token(os.getenv('FREESOUND_API_KEY'))

# Create directories
os.makedirs('data/raw/rain_sounds', exist_ok=True)
os.makedirs('data/raw/rain_sounds/metadata', exist_ok=True)

# Define surface types and their search terms
SURFACE_TYPES = {
    'metal': ['rain metal', 'rain tin', 'rain roof'],
    'glass': ['rain window', 'rain glass'],
    'concrete': ['rain concrete', 'rain pavement'],
    'wood': ['rain wood', 'rain deck'],
    'leaves': ['rain leaves', 'rain forest'],
    'water': ['rain water', 'rain puddle'],
    'grass': ['rain grass', 'rain lawn']
}

def retry_with_backoff(func, max_retries=5, initial_delay=1):
    """Retry a function with exponential backoff"""
    delay = initial_delay
    for attempt in range(max_retries):
        try:
            return func()
        except (FreesoundException, RequestException) as e:
            if attempt == max_retries - 1:
                raise
            print(f"Attempt {attempt + 1} failed: {str(e)}")
            print(f"Waiting {delay} seconds before retrying...")
            time.sleep(delay)
            delay *= 2  # Exponential backoff
            delay += random.uniform(0, 1)  # Add jitter

def download_sounds(surface_type, search_terms, max_sounds=3):
    """Download sounds for a specific surface type"""
    print(f"\nSearching for {surface_type} rain sounds...")
    
    # Create directory for this surface type
    surface_dir = f'data/raw/rain_sounds/{surface_type}'
    os.makedirs(surface_dir, exist_ok=True)
    
    downloaded_count = 0
    metadata = []
    
    for search_term in search_terms:
        if downloaded_count >= max_sounds:
            break
            
        def search():
            return fs.text_search(
                query=search_term,
                filter="duration:[30 TO 300] license:\"Attribution\"",
                sort="rating_desc",
                fields="id,name,url,previews,username,license,tags,duration,description"
            )
        
        try:
            results = retry_with_backoff(search)
            
            for sound in results:
                if downloaded_count >= max_sounds:
                    break
                    
                # Skip if sound already exists
                filename = f"{sound.name}.mp3"
                if os.path.exists(os.path.join(surface_dir, filename)):
                    print(f"Skipping {filename} - already exists")
                    continue
                
                try:
                    # Download the sound
                    print(f"Downloading: {sound.name}")
                    sound.retrieve_preview(surface_dir, filename)
                    
                    # Collect metadata
                    sound_metadata = {
                        'id': sound.id,
                        'name': sound.name,
                        'surface_type': surface_type,
                        'duration': sound.duration,
                        'tags': sound.tags,
                        'description': sound.description,
                        'license': sound.license,
                        'username': sound.username,
                        'search_term': search_term
                    }
                    metadata.append(sound_metadata)
                    
                    downloaded_count += 1
                    time.sleep(1)  # Respect API rate limits
                    
                except Exception as e:
                    print(f"Error downloading {sound.name}: {str(e)}")
                    continue
                    
        except Exception as e:
            print(f"Error searching for {search_term}: {str(e)}")
            continue
    
    # Save metadata for this surface type
    if metadata:
        import json
        metadata_file = f'data/raw/rain_sounds/metadata/{surface_type}_metadata.json'
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    return downloaded_count

def main():
    total_downloaded = 0
    
    for surface_type, search_terms in SURFACE_TYPES.items():
        try:
            downloaded = download_sounds(surface_type, search_terms)
            total_downloaded += downloaded
            print(f"Downloaded {downloaded} sounds for {surface_type}")
        except Exception as e:
            print(f"Error processing {surface_type}: {str(e)}")
            continue
    
    print(f"\nTotal sounds downloaded: {total_downloaded}")

if __name__ == "__main__":
    main() 