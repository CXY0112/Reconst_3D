import tkinter as tk
import unittest
from unittest.mock import patch
import numpy as np
from rgb_preview import RGBPreviewPanel


class RGBPreviewTests(unittest.TestCase):
    def test_bgr_colors_orientation_aspect_and_latest_frame(self):
        root = tk.Tk(); root.withdraw()
        try:
            panel = RGBPreviewPanel(root)
            image = np.zeros((20, 40, 3), np.uint8)
            image[:, :20] = (0, 0, 255)
            image[:, 20:] = (255, 0, 0)
            with patch.object(panel.frame, 'winfo_width', return_value=20), patch.object(panel.frame, 'winfo_height', return_value=20):
                panel.update(image)
                self.assertEqual((panel.photo.width(), panel.photo.height()), (20, 10))
                self.assertEqual(panel.photo.get(0, 0), (255, 0, 0))
                self.assertEqual(panel.photo.get(19, 0), (0, 0, 255))
                panel.update(np.full_like(image, (0, 255, 0)))
                self.assertEqual(panel.photo.get(0, 0), (0, 255, 0))
                panel.clear()
                self.assertIsNone(panel.image)
        finally:
            root.destroy()
