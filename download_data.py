"""Download and prepare the Text8 corpus for training."""

import os
import requests
import zipfile

def download_text8(data_dir='data'):
    """Download the Text8 corpus."""
    os.makedirs(data_dir, exist_ok=True)
    
    url = 'http://mattmahoney.net/dc/text8.zip'
    zip_path = os.path.join(data_dir, 'text8.zip')
    text_path = os.path.join(data_dir, 'text8')
    
    if os.path.exists(text_path):
        print(f"Text8 corpus already exists at {text_path}")
        return text_path
    
    print(f"Downloading Text8 corpus from {url}...")
    response = requests.get(url)
    
    with open(zip_path, 'wb') as f:
        f.write(response.content)
    
    print("Extracting...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(data_dir)
    
    os.remove(zip_path)
    print(f"Text8 corpus downloaded and extracted to {text_path}")
    
    return text_path

if __name__ == '__main__':
    download_text8()
