import sys
import cv2
import os

# To get card images: https://github.com/smlbiobot/cr-cardgen/tree/master/cardgen/ui_spells_out

# Change these values if you want the card size to be different
WIDTH = 195
HEIGHT = 250
# 

script_location = os.path.dirname(os.path.realpath(__file__))

if not os.path.exists(rf"{script_location}\out"):
    os.mkdir(rf"{script_location}\out")
inputs = sys.argv[1:]

# read mask image
mask = cv2.imread(rf'{script_location}\assets\champion_card_mask.png', 0)

_, mask = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY)
mask = cv2.resize(mask, (WIDTH,HEIGHT))

for arg in inputs:
    if arg.endswith(".png"):
        # Load image as Numpy array in BGR order
        original_image = cv2.imread(arg)
        
        # Resize the image
        original_image = cv2.resize(original_image, (WIDTH,HEIGHT))
        
        # Copy original image
        result = original_image.copy()
        
        # Convert the image to BGRA 
        result = cv2.cvtColor(result, cv2.COLOR_BGR2BGRA)
        
        # Insert the alpha channel
        result[:, :, 3] = mask

        # Save result
        path = rf"{script_location}\out\{os.path.basename(arg)}"
        # print("saving image at :", path)
        cv2.imwrite(path, result)