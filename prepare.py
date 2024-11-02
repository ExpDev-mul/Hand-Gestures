# This script prepares the dataset and converts all of its contents into npz inside dataset.npz, to be later extracted.

import numpy as np
import pandas as pd
import os
from PIL import Image

main_dir = r"C:\Users\orpin\OneDrive\Desktop\RNN\Train\Images"

IMG_SIZE = 130

dataset = {}

print("Dumping all dataset into a an npz file...")

for label in os.listdir(main_dir):
    label_folder_path = os.path.join(main_dir, label) # Reference our unique label's folder path

    dataset[label] = []

    for img_file in os.listdir(label_folder_path):
        img_path = os.path.join(label_folder_path, img_file)

        with Image.open(img_path) as img:
            # After extracting the raw image, we should begin to process it
            img = img.convert('RGB') # Ensure color channels are aligned
            img = img.resize((IMG_SIZE, IMG_SIZE))

            dataset[label].append( np.array(img) )

    print(label)


print("Succesfully dumped everything at dataset.npz")

npz_data = {
    label: np.array(images) for label, images in dataset.items()
}

np.savez_compressed('dataset.npz', **npz_data)