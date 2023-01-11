import sys
import cv2
import numpy as np
import os

# Change these values if you want the card size to be different
WIDTH = 195
HEIGHT = 250
# 
BGR_THRESHOLD = 15

script_location = os.path.dirname(os.path.realpath(__file__))

if not os.path.exists(rf"{script_location}\out"):
    os.mkdir(rf"{script_location}\out")
inputs = sys.argv[1:]

# read mask image
mask = cv2.imread(rf'{script_location}\legendary_card_mask.png', 0)

# Get the black pixels of the mask image
_, mask = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY)
mask = cv2.resize(mask, (WIDTH,HEIGHT))
        
# Add an extra dimension in the last axis.
mask = np.expand_dims(mask, axis=-1)

# Inverts the mask
inverted_mask = cv2.bitwise_not(mask)

for arg in inputs:
    if arg.endswith(".png"):
        # Load image as Numpy array in BGR order
        original_image = cv2.imread(arg)
        
        # Resize the image
        original_image = cv2.resize(original_image, (WIDTH,HEIGHT))
        
        # Gets the inverted masked area of the original image
        transparent_crop = cv2.bitwise_and(original_image, original_image, mask=inverted_mask)
        
        original_area = np.sum(mask, axis=-1)
        
        # Make a True/False mask of pixels whose BGR values sum to more than BGR_THRESHOLD
        alpha = np.sum(transparent_crop, axis=-1) > BGR_THRESHOLD

        # If the pixel is not transparent then keep it
        original_and_alpha_area = (np.logical_or(original_area, alpha))
        
        # Convert True/False to 0/255 and change type to "uint8" to match "original_image"
        original_and_alpha_area = np.uint8(original_and_alpha_area * 255)

        # Add the alpha channel to the original image
        result = np.dstack((original_image, original_and_alpha_area))

        # Save result
        path = rf"{script_location}\out\{os.path.basename(arg)}"
        # print("saving image at :", path)
        cv2.imwrite(path, result)