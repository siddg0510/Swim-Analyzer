import unittest
import numpy as np
import cv2

from src.vision.ego_motion import EgoMotionTracker

class TestEgoMotionTracker(unittest.TestCase):
    def test_ego_motion_tracking(self):
        # Create a synthetic image with some "background" shapes
        frame0 = np.zeros((480, 640, 3), dtype=np.uint8)
        # Draw some fixed rectangles to act as background features
        for x in range(50, 600, 100):
            for y in range(50, 400, 100):
                cv2.rectangle(frame0, (x, y), (x+20, y+20), (255, 255, 255), -1)
                
        # Initialize tracker
        tracker = EgoMotionTracker(frame0, lane_polygon=None)
        
        # Simulate camera panning to the right by shifting the background to the left by 10 pixels
        frame1 = np.zeros_like(frame0)
        for x in range(50, 600, 100):
            for y in range(50, 400, 100):
                cv2.rectangle(frame1, (x-10, y), (x+10, y+20), (255, 255, 255), -1)
                
        dx, dy = tracker.update(frame1)
        
        # Since background moved left (-10), camera moved right (+10).
        # Cumulative dx should be +10 (approximate due to optical flow)
        self.assertTrue(5 < dx < 15, f"Expected dx around 10, got {dx}")
        self.assertTrue(abs(dy) < 2, f"Expected dy around 0, got {dy}")
        
if __name__ == '__main__':
    unittest.main()
