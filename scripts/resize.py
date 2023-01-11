import sys
import cv2
import numpy as np
import os

if not os.path.exists(r"scripts\out"):
    os.mkdir(r"scripts\out")
inputs = sys.argv[1:]

# read mask image
mask = cv2.imread(r'scripts\legendary_card_mask.png', 0)

# Get the black pixels of the mask image
_, mask = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY)
mask = cv2.resize(mask, (195,250))

for arg in inputs:
    if arg.endswith(".png"):
        # Load image as Numpy array in BGR order
        original_image = cv2.imread(arg)
        
        # Resize the image
        original_image = cv2.resize(original_image, (195,250))
        
        inverted_mask = cv2.bitwise_not(mask)
        
        # original_crop = cv2.bitwise_and(original_image, original_image, mask=mask)
        transparent_crop = cv2.bitwise_and(original_image, original_image, mask=inverted_mask)
        
        # Make a True/False mask of pixels whose BGR values sum to more than zero
        alpha = np.sum(transparent_crop, axis=-1) > 5

        # Convert True/False to 0/255 and change type to "uint8" to match "original_image"
        alpha = np.uint8(alpha * 255)

        alpha = cv2.merge([alpha]*3)

        result = original_image.copy()
        result[inverted_mask!=0] = alpha[inverted_mask!=0]

        # Save result
        path = rf"scripts\out\{os.path.basename(arg)}"
        print("saving image at :", path)
        cv2.imwrite(path, result)