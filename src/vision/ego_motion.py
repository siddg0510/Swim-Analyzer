import cv2
import numpy as np

class EgoMotionTracker:
    """
    Tracks camera movement (ego-motion) by following static background features.
    This allows the app to process videos shot by a moving cameraman walking
    alongside the pool.
    """
    def __init__(self, first_frame: np.ndarray, lane_polygon: list[tuple[float, float]] | None = None):
        self.prev_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
        self.cumulative_dx = 0.0
        self.cumulative_dy = 0.0
        
        # Mask out the lane polygon so we don't track splashing water or the swimmer
        self.mask = np.ones_like(self.prev_gray, dtype=np.uint8) * 255
        if lane_polygon and len(lane_polygon) >= 3:
            pts = np.array(lane_polygon, dtype=np.int32)
            cv2.fillPoly(self.mask, [pts], 0)
            
        # Also mask out the top 10% and bottom 10% of the screen as they often contain spectators/moving legs
        h, w = self.prev_gray.shape
        self.mask[0:int(h*0.1), :] = 0
        self.mask[int(h*0.9):, :] = 0

        # Extract initial features to track
        self.p0 = cv2.goodFeaturesToTrack(
            self.prev_gray, 
            mask=self.mask, 
            maxCorners=200, 
            qualityLevel=0.1, 
            minDistance=15, 
            blockSize=7
        )

    def update(self, frame: np.ndarray) -> tuple[float, float]:
        """
        Calculates the camera shift since the first frame.
        Returns: (cumulative_dx, cumulative_dy)
        """
        if self.p0 is None or len(self.p0) < 10:
            # Re-initialize features if we lost too many
            self.p0 = cv2.goodFeaturesToTrack(
                self.prev_gray, 
                mask=self.mask, 
                maxCorners=200, 
                qualityLevel=0.1, 
                minDistance=15, 
                blockSize=7
            )
            
        if self.p0 is None or len(self.p0) == 0:
            return self.cumulative_dx, self.cumulative_dy
            
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Calculate optical flow
        p1, st, err = cv2.calcOpticalFlowPyrLK(
            self.prev_gray, 
            gray, 
            self.p0, 
            None, 
            winSize=(21, 21), 
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
        )
        
        if p1 is not None and st is not None:
            good_new = p1[st == 1]
            good_old = self.p0[st == 1]
            
            if len(good_new) > 0:
                # Calculate median translation
                dxs = good_new[:, 0] - good_old[:, 0]
                dys = good_new[:, 1] - good_old[:, 1]
                
                dx = np.median(dxs)
                dy = np.median(dys)
                
                # If background moved left (-dx), camera moved right (+dx). 
                # We want to shift the swimmer's coordinates in the OPPOSITE direction 
                # of the background movement to keep them in the absolute coordinate space.
                # Actually, if background moves left (dx < 0), the camera moved right.
                # A static object at pixel x=100 moved to x=90 (dx=-10).
                # To map the new frame's pixels back to the original frame's coordinate space,
                # we need to subtract the background movement from the pixel coordinate.
                # x_original = x_new - dx.  E.g. 90 - (-10) = 100.
                self.cumulative_dx -= dx
                self.cumulative_dy -= dy
                
            self.p0 = good_new.reshape(-1, 1, 2)
            
        self.prev_gray = gray.copy()
        
        return self.cumulative_dx, self.cumulative_dy
