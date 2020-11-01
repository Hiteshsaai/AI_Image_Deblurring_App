import cv2
import os
import numpy as np
from tqdm import tqdm


## Creating the directory for the gaussin blurred images to reside
os.makedirs('../input_images/gaussian_blurred', exist_ok=True)

# source directory
src_dir = '../input_images/sharp'
images = os.listdir(src_dir)

# destination directory
dst_dir = '../input_images/gaussian_blurred'

for i, img in tqdm(enumerate(images), total=len(images)):
    img = cv2.imread(f"{src_dir}/{images[i]}", cv2.IMREAD_COLOR)
    # add gaussian blurring
    blur = cv2.GaussianBlur(img, (35, 35), 0)
    cv2.imwrite(f"{dst_dir}/{images[i]}", blur)

print('DONE')
