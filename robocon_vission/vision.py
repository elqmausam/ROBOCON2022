import cv2
import numpy as np
import time
from collections import deque
import math

class AdvancedBallDetector:
    def __init__(self, buffer_size=10):
        """
        Advanced ball detector that distinguishes balls from other circular objects
        using multiple geometric and physical features
        """
        self.buffer_size = buffer_size
        self.position_buffer = deque(maxlen=buffer_size)
        self.velocity_buffer = deque(maxlen=5)
        self.last_known_position = None
        self.last_detection_time = time.time()
        
        # Multi-scale Hough parameters for different ball sizes
        self.hough_configs = [
            {'dp': 1, 'min_dist': 30, 'param1': 50, 'param2': 30, 'min_radius': 8, 'max_radius': 25},   # Small balls
            {'dp': 1, 'min_dist': 50, 'param1': 50, 'param2': 30, 'min_radius': 20, 'max_radius': 60},  # Medium balls
            {'dp': 1, 'min_dist': 80, 'param1': 50, 'param2': 30, 'min_radius': 50, 'max_radius': 120}, # Large balls
        ]
        
       
        self.ball_criteria = {
            'min_gradient_strength': 20,      
            'min_symmetry_score': 0.6,        
            'max_edge_irregularity': 0.3,     
            'min_texture_variance': 50,       # Balls have some texture variation
            'max_texture_variance': 180,      # But not too much (not text/patterns)
            'min_area_ratio': 0.6,            # Circle area vs actual contour area
            'bounce_detection': True,         
            'shadow_detection': True,         
        }
        
        # Background subtractor for motion analysis
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(detectShadows=True)
        
        # Previous frame for optical flow
        self.prev_gray = None
        
    def preprocess_frame(self, frame):
        """Enhanced preprocessing for better ball detection"""
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply CLAHE for better contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        
        # Multiple blur scales for different ball sizes
        blur_light = cv2.GaussianBlur(enhanced, (5, 5), 1)
        blur_medium = cv2.GaussianBlur(enhanced, (9, 9), 2)
        blur_heavy = cv2.GaussianBlur(enhanced, (15, 15), 3)
        
        return gray, enhanced, [blur_light, blur_medium, blur_heavy]
    
    def detect_multi_scale_circles(self, blurred_images):
        """Detect circles at multiple scales"""
        all_circles = []
        
        for i, (blur_img, config) in enumerate(zip(blurred_images, self.hough_configs)):
            circles = cv2.HoughCircles(
                blur_img,
                cv2.HOUGH_GRADIENT,
                dp=config['dp'],
                minDist=config['min_dist'],
                param1=config['param1'],
                param2=config['param2'],
                minRadius=config['min_radius'],
                maxRadius=config['max_radius']
            )
            
            if circles is not None:
                circles = np.round(circles[0, :]).astype("int")
                for (x, y, r) in circles:
                    all_circles.append((x, y, r, i))  # Include scale index
        
        return all_circles
    
    def calculate_gradient_strength(self, gray, x, y, r):
        """Calculate gradient strength around circle perimeter"""
        # Create mask for circle perimeter
        mask = np.zeros(gray.shape, dtype=np.uint8)
        cv2.circle(mask, (x, y), r, 255, 2)
        
        # Calculate gradients
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        # Get gradient strength at circle perimeter
        perimeter_gradients = gradient_magnitude[mask == 255]
        return np.mean(perimeter_gradients) if len(perimeter_gradients) > 0 else 0
    
    def calculate_symmetry_score(self, gray, x, y, r):
        """Calculate radial symmetry score"""
        if r < 5:
            return 0
        
        # Extract circular region
        mask = np.zeros(gray.shape, dtype=np.uint8)
        cv2.circle(mask, (x, y), r, 255, -1)
        
        # Create polar coordinate representation
        angles = np.linspace(0, 2*np.pi, 36)  # 36 radial lines
        symmetry_scores = []
        
        for angle in angles:
            # Get radial line
            radial_line = []
            for radius in range(1, r):
                px = int(x + radius * np.cos(angle))
                py = int(y + radius * np.sin(angle))
                if 0 <= px < gray.shape[1] and 0 <= py < gray.shape[0]:
                    radial_line.append(gray[py, px])
            
            if len(radial_line) > 4:
                # Compare with opposite radial line
                opposite_angle = angle + np.pi
                opposite_line = []
                for radius in range(1, r):
                    px = int(x + radius * np.cos(opposite_angle))
                    py = int(y + radius * np.sin(opposite_angle))
                    if 0 <= px < gray.shape[1] and 0 <= py < gray.shape[0]:
                        opposite_line.append(gray[py, px])
                
                if len(opposite_line) > 4:
                    # Calculate correlation between radial lines
                    min_len = min(len(radial_line), len(opposite_line))
                    correlation = np.corrcoef(
                        radial_line[:min_len], 
                        opposite_line[:min_len][::-1]  # Reverse for symmetry
                    )[0, 1]
                    if not np.isnan(correlation):
                        symmetry_scores.append(abs(correlation))
        
        return np.mean(symmetry_scores) if symmetry_scores else 0
    
    def calculate_edge_regularity(self, gray, x, y, r):
        """Calculate how regular/smooth the circle edge is"""
        # Get points on circle perimeter
        angles = np.linspace(0, 2*np.pi, 72)
        edge_points = []
        
        for angle in angles:
            # Sample points around the expected circle
            for delta_r in range(-2, 3):
                px = int(x + (r + delta_r) * np.cos(angle))
                py = int(y + (r + delta_r) * np.sin(angle))
                if 0 <= px < gray.shape[1] and 0 <= py < gray.shape[0]:
                    edge_points.append(gray[py, px])
        
        if len(edge_points) < 10:
            return 1.0  # Assume irregular if can't measure
        
        # Calculate variance in edge intensity (lower = more regular)
        edge_variance = np.var(edge_points)
        regularity = 1.0 / (1.0 + edge_variance / 100.0)  # Normalize
        return regularity
    
    def calculate_texture_features(self, gray, x, y, r):
        """Calculate texture variance within the circle"""
        # Create circular mask
        mask = np.zeros(gray.shape, dtype=np.uint8)
        cv2.circle(mask, (x, y), max(r-3, 1), 255, -1)
        
        # Get pixels within circle
        circle_pixels = gray[mask == 255]
        
        if len(circle_pixels) < 10:
            return 0, 0
        
        texture_var = np.var(circle_pixels)
        texture_mean = np.mean(circle_pixels)
        
        return texture_var, texture_mean
    
    def detect_shadow_or_ground_contact(self, frame, x, y, r):
        """Detect if object has shadow or ground contact (ball-like behavior)"""
        # Look for darker region below the circle (shadow)
        shadow_region_y = min(y + r + 5, frame.shape[0] - 10)
        shadow_region = frame[shadow_region_y:shadow_region_y + 10, 
                            max(0, x - r):min(frame.shape[1], x + r)]
        
        if shadow_region.size == 0:
            return False
        
        # Convert to grayscale for shadow detection
        if len(shadow_region.shape) == 3:
            shadow_gray = cv2.cvtColor(shadow_region, cv2.COLOR_BGR2GRAY)
        else:
            shadow_gray = shadow_region
        
        # Check if region below is darker (shadow indication)
        circle_region = frame[max(0, y - r):min(frame.shape[0], y + r),
                            max(0, x - r):min(frame.shape[1], x + r)]
        
        if circle_region.size == 0:
            return False
        
        if len(circle_region.shape) == 3:
            circle_gray = cv2.cvtColor(circle_region, cv2.COLOR_BGR2GRAY)
        else:
            circle_gray = circle_region
        
        shadow_brightness = np.mean(shadow_gray)
        circle_brightness = np.mean(circle_gray)
        
        # Shadow should be darker than the ball
        return shadow_brightness < circle_brightness * 0.85
    
    def analyze_motion_physics(self, x, y, current_time):
        """Analyze motion to validate ball-like physics"""
        if len(self.position_buffer) < 3:
            return True  # Not enough data yet
        
        # Calculate velocity
        prev_pos = self.position_buffer[-1]
        dt = current_time - self.last_detection_time
        
        if dt > 0:
            vx = (x - prev_pos[0]) / dt
            vy = (y - prev_pos[1]) / dt
            velocity = np.sqrt(vx**2 + vy**2)
            
            # Add to velocity buffer
            self.velocity_buffer.append((vx, vy, velocity))
            
            # Check for realistic motion patterns
            if len(self.velocity_buffer) >= 3:
                velocities = [v[2] for v in self.velocity_buffer]
                
                # Balls shouldn't have sudden extreme velocity changes
                velocity_changes = np.diff(velocities)
                max_change = np.max(np.abs(velocity_changes))
                
                # Realistic physics: velocity shouldn't change too dramatically
                return max_change < 200  # pixels per second change limit
        
        return True
    
    def classify_ball_candidate(self, frame, gray, x, y, r):
        """Comprehensive ball classification using multiple features"""
        features = {}
        
        # Calculate all features
        features['gradient_strength'] = self.calculate_gradient_strength(gray, x, y, r)
        features['symmetry_score'] = self.calculate_symmetry_score(gray, x, y, r)
        features['edge_regularity'] = self.calculate_edge_regularity(gray, x, y, r)
        features['texture_var'], features['texture_mean'] = self.calculate_texture_features(gray, x, y, r)
        
        # Physics-based validation
        current_time = time.time()
        features['physics_valid'] = self.analyze_motion_physics(x, y, current_time)
        
        # Shadow/ground contact detection
        features['has_shadow'] = self.detect_shadow_or_ground_contact(frame, x, y, r)
        
        # Calculate ball probability score
        score = 0
        
        # Gradient strength (balls have strong circular gradients)
        if features['gradient_strength'] >= self.ball_criteria['min_gradient_strength']:
            score += 1.5
        
        # Symmetry (balls are radially symmetric)
        if features['symmetry_score'] >= self.ball_criteria['min_symmetry_score']:
            score += 2.0
        
        # Edge regularity (ball edges are smooth)
        if features['edge_regularity'] >= (1 - self.ball_criteria['max_edge_irregularity']):
            score += 1.5
        
        # Texture variance (reasonable texture, not flat or too busy)
        if (self.ball_criteria['min_texture_variance'] <= features['texture_var'] <= 
            self.ball_criteria['max_texture_variance']):
            score += 1.0
        
        # Physics validation
        if features['physics_valid']:
            score += 1.0
        
        # Shadow presence (balls on ground cast shadows)
        if features['has_shadow']:
            score += 0.5
        
        # Size reasonableness (balls have reasonable size ranges)
        if 10 <= r <= 100:  # Reasonable ball size range
            score += 0.5
        
        return score, features
    
    def detect_balls(self, frame):
        """Main detection method with advanced ball classification"""
        gray, enhanced, blurred_images = self.preprocess_frame(frame)
        
        # Detect circles at multiple scales
        circle_candidates = self.detect_multi_scale_circles(blurred_images)
        
        # Classify each candidate
        ball_detections = []
        for (x, y, r, scale_idx) in circle_candidates:
            # Ensure circle is within frame bounds
            if (r < x < frame.shape[1] - r and r < y < frame.shape[0] - r):
                score, features = self.classify_ball_candidate(frame, gray, x, y, r)
                
                # Threshold for ball classification (adjust based on requirements)
                if score >= 4.0:  # Minimum score to be considered a ball
                    ball_detections.append({
                        'position': (x, y, r),
                        'score': score,
                        'features': features,
                        'scale': scale_idx
                    })
        
        # Sort by score and remove overlapping detections
        ball_detections.sort(key=lambda x: x['score'], reverse=True)
        filtered_balls = self.remove_overlapping_detections(ball_detections)
        
        # Track the best ball
        tracked_ball = self.track_best_ball(filtered_balls)
        
        # Update previous frame for optical flow
        self.prev_gray = gray.copy()
        self.last_detection_time = time.time()
        
        return tracked_ball, filtered_balls
    
    def remove_overlapping_detections(self, detections, overlap_threshold=0.5):
        """Remove overlapping detections keeping the highest scoring ones"""
        if len(detections) <= 1:
            return detections
        
        filtered = []
        for i, det1 in enumerate(detections):
            x1, y1, r1 = det1['position']
            is_overlap = False
            
            for j, det2 in enumerate(filtered):
                x2, y2, r2 = det2['position']
                distance = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
                overlap = distance < (r1 + r2) * overlap_threshold
                
                if overlap:
                    is_overlap = True
                    break
            
            if not is_overlap:
                filtered.append(det1)
        
        return filtered
    
    def track_best_ball(self, detections):
        """Track the most likely ball using temporal consistency"""
        if not detections:
            return None
        
        # If we have a previous position, find the closest high-scoring detection
        if self.last_known_position is not None:
            min_distance = float('inf')
            best_detection = None
            
            for detection in detections:
                x, y, r = detection['position']
                distance = np.sqrt((x - self.last_known_position[0])**2 + 
                                 (y - self.last_known_position[1])**2)
                
                # Weight by both distance and score
                weighted_score = detection['score'] / (1 + distance / 50)
                
                if distance < 150 and weighted_score > min_distance:  # Max jump distance
                    min_distance = weighted_score
                    best_detection = detection
            
            if best_detection is not None:
                current_pos = best_detection['position']
            else:
                current_pos = detections[0]['position']  # Take highest scoring
        else:
            current_pos = detections[0]['position']  # Take highest scoring
        
        # Add to tracking buffer
        self.position_buffer.append(current_pos)
        self.last_known_position = (current_pos[0], current_pos[1])
        
       
        if len(self.position_buffer) > 3:
            positions = np.array(list(self.position_buffer))
            smoothed = np.mean(positions[-3:], axis=0)
            return (int(smoothed[0]), int(smoothed[1]), int(smoothed[2]))
        
        return current_pos

def main():
    """Main function for advanced real-time ball detection"""
   
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open camera")
        return
    
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
   
    detector = AdvancedBallDetector()
    
   
    fps_counter = 0
    start_time = time.time()
    
    print("Advanced Ball Detection Started!")
    print("This system distinguishes balls from other circular objects")
    print("Press 'q' to quit")
    print("Press 'd' to toggle debug information")
    
    show_debug = False
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        tracked_ball, all_detections = detector.detect_balls(frame)
        
        
        result_frame = frame.copy()
        
       
        for detection in all_detections:
            x, y, r = detection['position']
            score = detection['score']
            
           
            if score >= 6.0:
                color = (0, 255, 0)  
            elif score >= 5.0:
                color = (0, 255, 255) 
            else:
                color = (0, 165, 255)  
            
            cv2.circle(result_frame, (x, y), r, color, 2)
            cv2.circle(result_frame, (x, y), 2, color, 3)
            
            # Show score
            cv2.putText(result_frame, f"{score:.1f}", (x-15, y-r-5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            
            if show_debug:
                features = detection['features']
                debug_text = [
                    f"Grad: {features['gradient_strength']:.1f}",
                    f"Sym: {features['symmetry_score']:.2f}",
                    f"Reg: {features['edge_regularity']:.2f}",
                    f"Tex: {features['texture_var']:.0f}",
                    f"Phys: {'Y' if features['physics_valid'] else 'N'}",
                    f"Shad: {'Y' if features['has_shadow'] else 'N'}"
                ]
                
                for i, text in enumerate(debug_text):
                    cv2.putText(result_frame, text, (x + r + 5, y - 30 + i*15), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        
        if tracked_ball is not None:
            x, y, r = tracked_ball
            cv2.circle(result_frame, (x, y), r, (255, 0, 0), 4)  
            cv2.putText(result_frame, "TRACKED BALL", (x-50, y+r+20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        
       
        fps_counter += 1
        if fps_counter % 30 == 0:
            end_time = time.time()
            fps = 30 / (end_time - start_time)
            start_time = end_time
        
        cv2.putText(result_frame, f"FPS: {fps:.1f}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(result_frame, f"Balls detected: {len(all_detections)}", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        if show_debug:
            cv2.putText(result_frame, "Debug Mode ON", (10, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        
        cv2.imshow('Advanced Ball Detection', result_frame)
        
       
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('d'):
            show_debug = not show_debug
            print(f"Debug mode: {'ON' if show_debug else 'OFF'}")
    
   
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
