import cv2
import numpy as np
from datetime import datetime, timedelta

class MotionEventDetector:
    def __init__(self, threshold=30, min_area=500, history_size=10):
        """
        Initialize the motion and event detector.
        
        Args:
            threshold (int): Threshold for frame difference detection
            min_area (int): Minimum contour area to be considered as motion
            history_size (int): Number of frames to keep in history for event detection
        """
        self.threshold = threshold
        self.min_area = min_area
        self.history_size = history_size
        self.motion_history = []
        
    def compare_histograms(self, frame1, frame2):
        """Compare histograms of two frames to detect overall changes."""
        hist1 = cv2.calcHist([frame1], [0], None, [256], [0, 256])
        hist2 = cv2.calcHist([frame2], [0], None, [256], [0, 256])
        
        # Normalize histograms
        cv2.normalize(hist1, hist1, 0, 1, cv2.NORM_MINMAX)
        cv2.normalize(hist2, hist2, 0, 1, cv2.NORM_MINMAX)
        
        return cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
    
    def detect_motion(self, frame1, frame2):
        """Detect motion between two frames using frame differencing."""
        # Convert frames to grayscale
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        
        # Calculate absolute difference
        frame_diff = cv2.absdiff(gray1, gray2)
        
        # Threshold the difference
        _, thresh = cv2.threshold(frame_diff, self.threshold, 255, cv2.THRESH_BINARY)
        
        # Apply morphological operations to remove noise
        kernel = np.ones((5, 5), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        
        return thresh
    
    def detect_events(self, motion_mask):
        """Detect significant events based on motion intensity."""
        # Calculate the percentage of pixels showing motion
        motion_intensity = np.sum(motion_mask > 0) / (motion_mask.shape[0] * motion_mask.shape[1])
        self.motion_history.append(motion_intensity)
        
        # Keep only recent history
        if len(self.motion_history) > self.history_size:
            self.motion_history.pop(0)
        
        # Calculate average and standard deviation of recent motion
        if len(self.motion_history) >= 3:
            avg_motion = np.mean(self.motion_history[:-1])
            std_motion = np.std(self.motion_history[:-1])
            current_motion = self.motion_history[-1]
            
            # Detect significant deviation from recent history
            if current_motion > avg_motion + 2 * std_motion:
                return True, motion_intensity
        
        return False, motion_intensity

def process_video(video_path, output_path=None):
    """
    Process video file for motion and event detection.
    
    Args:
        video_path (str): Path to input video file
        output_path (str): Path to save output video (optional)
    """
    # Initialize video capture
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Error opening video file")
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Initialize video writer if output path is provided
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    
    # Initialize detector
    detector = MotionEventDetector()
    
    # Read first frame
    ret, prev_frame = cap.read()
    if not ret:
        return
    
    frame_count = 0
    start_time = datetime.now()
    events = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Calculate frame timestamp
        timestamp = start_time + timedelta(seconds=frame_count/fps)
        
        # Detect motion
        motion_mask = detector.detect_motion(prev_frame, frame)
        
        # Compare histograms
        hist_correlation = detector.compare_histograms(prev_frame, frame)
        
        # Detect events
        event_detected, motion_intensity = detector.detect_events(motion_mask)
        
        if event_detected:
            events.append({
                'frame': frame_count,
                'timestamp': timestamp,
                'intensity': motion_intensity,
                'correlation': hist_correlation
            })
            
            # Annotate frame with event information
            cv2.putText(frame, f"Event Detected! Intensity: {motion_intensity:.2f}",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # Visualize motion
        motion_overlay = cv2.cvtColor(motion_mask, cv2.COLOR_GRAY2BGR)
        frame_with_motion = cv2.addWeighted(frame, 1, motion_overlay, 0.3, 0)
        
        # Add timestamp
        cv2.putText(frame_with_motion, timestamp.strftime('%H:%M:%S.%f')[:-4],
                   (10, frame_height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Write frame if output path is provided
        if output_path:
            out.write(frame_with_motion)
        
        # Display frame
        cv2.imshow('Motion Detection', frame_with_motion)
        
        # Update previous frame
        prev_frame = frame.copy()
        frame_count += 1
        
        # Break on 'q' press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Clean up
    cap.release()
    if output_path:
        out.release()
    cv2.destroyAllWindows()
    
    return events

# Example usage
if __name__ == "__main__":
    video_path = "C:\\Users\\sraja\\Downloads\\2257010-uhd_3840_2160_24fps.mp4"
    output_path = "C:\\Users\\sraja\\Downloads\\output_video.avi"  # Optional output path
    
    try:
        events = process_video(video_path, output_path)
        
        # Print detected events
        print("\nDetected Events:")
        for event in events:
            print(f"Frame {event['frame']}: "
                  f"Time: {event['timestamp'].strftime('%H:%M:%S.%f')[:-4]}, "
                  f"Intensity: {event['intensity']:.2f}, "
                  f"Correlation: {event['correlation']:.2f}")
                  
    except Exception as e:
        print(f"Error processing video: {str(e)}")