import unittest
import numpy as np
import cv2
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from engine.color_logic import analyze_anemia_eyelid, analyze_jaundice_sclera, extract_mean_color

class TestColorLogic(unittest.TestCase):

    def setUp(self):
        # Create a dummy BGR image (100x100x3)
        self.image = np.zeros((100, 100, 3), dtype=np.uint8)
        # Fill with a specific color
        self.image[:, :] = [100, 150, 200]  # BGR
        
        # Create a dummy mask (100x100)
        self.mask = np.zeros((100, 100), dtype=np.uint8)
        self.mask[25:75, 25:75] = 255

    def test_analyze_anemia_eyelid(self):
        a_val = analyze_anemia_eyelid(self.image, self.mask)
        self.assertIsNotNone(a_val)
        self.assertIsInstance(a_val, float)
        
        # Test with None image
        self.assertIsNone(analyze_anemia_eyelid(None, self.mask))

    def test_analyze_jaundice_sclera(self):
        b_val = analyze_jaundice_sclera(self.image, self.mask)
        self.assertIsNotNone(b_val)
        self.assertIsInstance(b_val, float)
        
        # Test with None image
        self.assertIsNone(analyze_jaundice_sclera(None, self.mask))

    def test_extract_mean_color_legacy(self):
        a_val = extract_mean_color(self.image, self.mask, 'a')
        b_val = extract_mean_color(self.image, self.mask, 'b')
        self.assertIsNotNone(a_val)
        self.assertIsNotNone(b_val)

if __name__ == '__main__':
    unittest.main()
