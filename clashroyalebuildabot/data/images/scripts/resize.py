import sys
import cv2
import numpy as np
import os

for arg in sys.argv[1:]:
    if arg.endswith(".png"):
        # Load image as Numpy array in BGR order
        na = cv2.imread(sys.argv[1])
        
        # Resize the image
        cv2.resize(na, (195,250))
        
        # Make a True/False mask of pixels whose BGR values sum to more than zero
        alpha = np.sum(na, axis=-1) > 5

        # Convert True/False to 0/255 and change type to "uint8" to match "na"
        alpha = np.uint8(alpha * 255)

        # Stack new alpha layer with existing image to go from BGR to BGRA, i.e. 3 channels to 4 channels
        res = np.dstack((na, alpha))
        
        # Save result
        cv2.imwrite("out/" + os.path.basename(arg), res)
        print("out/" + arg)