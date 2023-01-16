import subprocess

import numpy as np
import cv2

from clashroyalebuildabot.data.constants import SCREENSHOT_WIDTH, SCREENSHOT_HEIGHT


class Screen:
    def __init__(self):
        # Physical size: 720x1280 -> self.width = 720, self.height = 1280
        window_size = subprocess.check_output(['adb', 'shell', 'wm', 'size'])
        window_size = window_size.decode('ascii').replace('Physical size: ', '')
        self.width, self.height = [int(i) for i in window_size.split('x')]

    @staticmethod
    def click(x, y):
        """
        Click at the given (x, y) coordinate
        """
        subprocess.run(['adb', 'shell', 'input', 'tap', str(x), str(y)])

    def take_screenshot(self):
        """
        Take a screenshot of the emulator
        """
        screenshot_bytes = subprocess.run(['adb', 'exec-out', 'screencap'], check=True, capture_output=True).stdout
        # Convert screenshot_bytes to a numpy array
        screenshot = np.copy(np.frombuffer(screenshot_bytes[12:], np.uint8))
        # Resize array to the original image shape
        screenshot = np.resize(screenshot, (self.height, self.width, 4))
        # Remove the alpha channel
        screenshot = screenshot[:, :, :3]
        # Resize the image
        screenshot = cv2.resize(screenshot, (SCREENSHOT_WIDTH, SCREENSHOT_HEIGHT), interpolation = cv2.INTER_LINEAR)
        # Change color from BGR to RGB
        screenshot = cv2.cvtColor(screenshot, cv2.COLOR_BGR2RGB)
        return screenshot


def main():
    cls = Screen()
    screenshot = cls.take_screenshot()
    print(np.array(screenshot).shape)
    cv2.imwrite(img=screenshot, filename='screen.jpg')


if __name__ == '__main__':
    main()
