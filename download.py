"""
Download UHI data and model
"""

import os
import requests
import zipfile


def download_data():
    # Download zip archive in current directory
    if not os.path.isfile('sture2019a_data.zip'):
        url = 'http://folk.ntnu.no/oystestu/sture2019a/sture2019a_data.zip'
        r = requests.get(url, stream=True)
        sz = r.headers.get('content-length')

        with open('sture2019a_data.zip', 'wb') as f:
            if sz is None:
                f.write(r.content)
            else:
                sz = int(sz)
                print(f'Downloading data ({sz/2**20:.2f} MB)....')
                n_bytes = 0
                for data in r.iter_content(chunk_size=max(int(sz / 1024), 5 * 1024 * 1024)):
                    n_bytes += len(data)
                    f.write(data)
                    print(f'Downloading data {n_bytes/2**20:.2f} / {sz/2**20:.2f} MB')

                print('Download complete')

    # Extract zip archive to current directory
    print('Unzipping data in current directory..')
    with zipfile.ZipFile('sture2019a_data.zip', 'r') as z:
        z.extractall()

    print('Completed')


if __name__ == '__main__':
    download_data()




