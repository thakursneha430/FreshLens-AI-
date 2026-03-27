import os
import cv2
import numpy as np

import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
IMG_SIZE = 224

classes = ["fresh", "rotten"]

def process_images():
    for category in classes:
        path = os.path.join(RAW_DIR, category)
        save_path = os.path.join(PROCESSED_DIR, category)
        os.makedirs(save_path, exist_ok=True)

        for img in os.listdir(path):
            try:
                img_path = os.path.join(path, img)
                image = cv2.imread(img_path)
                image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
                cv2.imwrite(os.path.join(save_path, img), image)
            except:
                pass

if __name__ == "__main__":
    process_images()
    print("Dataset preprocessing completed ✅")