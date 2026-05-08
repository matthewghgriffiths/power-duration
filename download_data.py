import shutil
import zipfile
import requests

# Dropbox URL
url = "https://www.dropbox.com/scl/fi/auushkqzllkirm0l0eqqi/data.zip?rlkey=s39hk8r9njnyu89woggibg3j9&st=ho293uuq&dl=1"
# Output paths
zip_path = "data.zip"
extract_dir = "data"

def main():
    print(f"Downloading ZIP file to {zip_path}")
    with (
        requests.get(url, stream=True) as response, 
        open(zip_path, 'wb') as f
    ): 
        response.raise_for_status()
        shutil.copyfileobj(response.raw, f)
        
    print(f"Extracting {zip_path} to {extract_dir}")
    with zipfile.ZipFile(zip_path, 'r') as zipf:
        zipf.extractall(extract_dir)

if __name__ == '__main__':
    main()