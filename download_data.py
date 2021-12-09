# A simple downloader with progress bar

import requests
from tqdm import tqdm
import zipfile
from urllib.request import urlopen

url = "https://web.cs.dal.ca/~juanr/downloads/post_ocr_correction.zip"
block_size = 1024 #1 Kibibyte

filename = url.split("/")[-1]
print(f"Downloading {filename}...")
site = urlopen(url)
meta = site.info()
# Streaming, so we can iterate over the response.
response = requests.get(url, stream=True)
total_size_in_bytes = int(meta["Content-Length"])
progress_bar = tqdm(total = total_size_in_bytes, unit='iB', unit_scale=True)
with open(filename, 'wb') as file:
    for data in response.iter_content(block_size):
        progress_bar.update(len(data))
        file.write(data)
progress_bar.close()
print("Download complete")
print(f"Extracting {filename}...")
zip = zipfile.ZipFile(filename, "r")
zip.extractall()
zip.close()
print("Extracting complete")
