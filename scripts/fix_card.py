import sys
import cv2
import os

import assets.CHANGE_CARD_SIZE_HERE as card_size

# To get card images: https://github.com/smlbiobot/cr-cardgen/tree/master/cardgen/card-src-236x300

script_location = os.path.dirname(os.path.realpath(__file__))

if not os.path.exists(rf"{script_location}\out"):
    os.mkdir(rf"{script_location}\out")
inputs = sys.argv[1:]

for arg in inputs:
    if arg.endswith(".png"):
        # Load image as Numpy array in BGR order
        image = cv2.imread(arg, cv2.IMREAD_UNCHANGED)

        # Resize the image
        image = cv2.resize(image, (card_size.WIDTH,card_size.HEIGHT))
        
        image = cv2.cvtColor(image,cv2.COLOR_BGR2BGRA) 
        
        # Save result
        path = rf"{script_location}\out\{os.path.basename(arg)}"
        # print("saving image at :", path)
        cv2.imwrite(path, image)