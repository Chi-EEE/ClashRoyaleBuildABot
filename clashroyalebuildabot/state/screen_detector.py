import os

import numpy as np
import cv2

from clashroyalebuildabot.data.constants import SCREEN_CONFIG, DATA_DIR


class ScreenDetector:
    def __init__(self, hash_size=8, threshold=20):
        self.hash_size = hash_size
        self.threshold = threshold

        self.screen_hashes = self._calculate_screen_hashes()

    def _calculate_screen_hashes(self):
        screen_hashes = np.zeros((len(SCREEN_CONFIG), self.hash_size * self.hash_size * 3), dtype=np.int32)
        for i, name in enumerate(SCREEN_CONFIG.keys()):
            path = os.path.join(f'{DATA_DIR}/images/screen', f'{name}.png')
            image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            hash_ = np.array(cv2.resize(image, (self.hash_size, self.hash_size), interpolation=cv2.INTER_LINEAR)).flatten()
            screen_hashes[i] = hash_
        return screen_hashes

    def run(self, image):
        crop_hashes = np.array([np.array(cv2.resize(image[v['bbox'][1]:v['bbox'][3], v['bbox'][0]:v['bbox'][2]].copy(), (self.hash_size, self.hash_size), interpolation=cv2.INTER_LINEAR))
                                .flatten()
                                for v in SCREEN_CONFIG.values()])
        hash_diffs = np.mean(np.abs(crop_hashes - self.screen_hashes), axis=1)

        # No crops match
        if min(hash_diffs) > self.threshold:
            screen = 'in_game'
        else:
            screen = list(SCREEN_CONFIG.keys())[np.argmin(hash_diffs)]

        return screen
