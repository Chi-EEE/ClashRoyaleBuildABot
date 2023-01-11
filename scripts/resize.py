import sys
import cv2
import numpy as np
import os

if not os.path.exists("out"):
    os.mkdir("out")
inputs = sys.argv[1:]

# read mask image
mask = cv2.imread(r'scripts\legendary_card_mask.png', 0)

# Get the black pixels of the mask image
_, mask = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY)
mask = cv2.resize(mask, (195,250))

for arg in inputs:
    if arg.endswith(".png"):
        # Load image as Numpy array in BGR order
        na = cv2.imread(arg)
        
        # Resize the image
        na = cv2.resize(na, (195,250))
        
        inverted_mask = cv2.bitwise_not(mask)
        
        original_crop = cv2.bitwise_and(na, na, mask=mask)
        transparent_crop = cv2.bitwise_and(na, na, mask=inverted_mask)
        
        # Make a True/False mask of pixels whose BGR values sum to more than zero
        alpha = np.sum(transparent_crop, axis=-1) > 5

        # Convert True/False to 0/255 and change type to "uint8" to match "na"
        alpha = np.uint8(alpha * 255)

        # mask = circle[..., 3] != 0
        # background[mask] = circle[..., :3][mask]
        # Stack new alpha layer with existing image to go from BGR to BGRA, i.e. 3 channels to 4 channels
        # res = np.dstack((na, alpha))
        first_mask = original_crop[..., 3] != 0
        mask = alpha[..., 3] != 0

        first_mask[mask] = first_mask[..., :3][mask]

        # Save result
        path = "out/" + os.path.basename(arg)
        print("saving image at :", path)
        cv2.imwrite(path, first_mask)