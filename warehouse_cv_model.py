"""
Smart Warehouse Computer Vision Model
Detects boxes, people, forklifts and computes all required metrics
Based on YOLOv8 - Ready for hackathon demo
"""

import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict
import time
from datetime import datetime

# Import your database helpers
from cv_helpers import save_detection, log_collision_event, log_misplacement, log_unsafe_stack

class WarehouseCV:
    """
    Complete CV system for warehouse monitoring
    Detects and tracks: boxes, people, forklifts
    Computes: misplacements, unsafe stacks, collision risks
    """
    
    def __init__(self, zone_id='A1', camera_id='CAM-01', 
                 zone_boundaries=None, model_path='yolov8n.pt'):
        """
        Initialize the CV system
        
        Args:
            zone_id: Warehouse zone identifier
            camera_id: Camera identifier
            zone_boundaries: Dict with x_min, y_min, x_max, y_max for zone
            model_path: Path to YOLO model weights
        """
        self.zone_id = zone_id
        self.camera_id = camera_id
        
        # Zone boundaries for detecting misplacements
        if zone_boundaries is None:
            # Default to full frame
            self.zone_boundaries = {
                'x_min': 200,
                'y_min': 200,
                'x_max': 1720,
                'y_max': 880
            }
        else:
            self.zone_boundaries = zone_boundaries
        
        # Load YOLO model
        print(f"🤖 Loading YOLO model: {model_path}...")
        self.model = YOLO(model_path)
        print("✅ Model loaded successfully!")
        
        # YOLO class IDs (COCO dataset)
        self.CLASS_PERSON = 0
        self.CLASS_CAR = 2      # We'll use 'car' or 'truck' as forklift
        self.CLASS_TRUCK = 7
        self.CLASS_BACKPACK = 24  # Using backpack as "box" proxy
        self.CLASS_HANDBAG = 26
        self.CLASS_SUITCASE = 28
        
        # Tracking
        self.frame_count = 0
        self.last_save_time = time.time()
        self.save_interval = 2.0  # Save to DB every 2 seconds
        
        # Stack detection parameters
        self.max_stack_height = 3  # Maximum safe stack height
        
        print(f"📍 Monitoring Zone: {zone_id} with Camera: {camera_id}")
        print(f"📏 Zone boundaries: {self.zone_boundaries}")
    
    def detect_frame(self, frame):
        """
        Process a single frame and extract all detections
        
        Args:
            frame: OpenCV image (BGR)
        
        Returns:
            dict: Detection results with all computed metrics
        """
        start_time = time.time()
        
        # Run YOLO detection
        results = self.model(frame, verbose=False)[0]
        
        # Initialize counters
        box_count = 0
        person_bboxes = []
        forklift_bboxes = []
        box_bboxes = []
        all_bboxes = []
        
        confidences = []
        
        # Process each detection
        for detection in results.boxes:
            class_id = int(detection.cls[0])
            confidence = float(detection.conf[0])
            x1, y1, x2, y2 = map(int, detection.xyxy[0].tolist())
            
            # Create bbox dict
            bbox = {
                'x_min': x1,
                'y_min': y1,
                'x_max': x2,
                'y_max': y2,
                'confidence': confidence,
                'class_id': class_id,
                'center_x': (x1 + x2) / 2,
                'center_y': (y1 + y2) / 2,
                'width': x2 - x1,
                'height': y2 - y1
            }
            
            confidences.append(confidence)
            
            # Classify detection
            if class_id == self.CLASS_PERSON:
                bbox['object_type'] = 'person'
                person_bboxes.append(bbox)
                all_bboxes.append(bbox)
                
            elif class_id in [self.CLASS_CAR, self.CLASS_TRUCK]:
                bbox['object_type'] = 'forklift'
                forklift_bboxes.append(bbox)
                all_bboxes.append(bbox)
                
            elif class_id in [self.CLASS_BACKPACK, self.CLASS_HANDBAG, self.CLASS_SUITCASE]:
                bbox['object_type'] = 'box'
                box_bboxes.append(bbox)
                all_bboxes.append(bbox)
                box_count += 1
        
        # Compute metrics
        misplaced = self._count_misplacements(box_bboxes)
        unsafe_stack = self._detect_unsafe_stacks(box_bboxes)
        
        person_detected = len(person_bboxes) > 0
        forklift_detected = len(forklift_bboxes) > 0
        
        collision_risk = False
        collision_data = None
        
        if person_detected and forklift_detected:
            collision_risk, collision_data = self._check_collision_risk(
                person_bboxes, forklift_bboxes
            )
        
        # Calculate processing time
        processing_time = (time.time() - start_time) * 1000  # milliseconds
        
        # Prepare detection data
        detection_result = {
            'box_count': box_count,
            'misplaced': misplaced,
            'unsafe_stack': unsafe_stack,
            'person_detected': person_detected,
            'forklift_detected': forklift_detected,
            'collision_risk': collision_risk,
            'avg_confidence': np.mean(confidences) if confidences else 0.0,
            'frame_number': self.frame_count,
            'processing_time_ms': processing_time,
            'bboxes': all_bboxes,
            'collision_data': collision_data,
            'misplaced_boxes': [b for b in box_bboxes if self._is_misplaced(b)],
            'unsafe_stacks': self._get_unsafe_stack_details(box_bboxes)
        }
        
        self.frame_count += 1
        
        return detection_result
    
    def _count_misplacements(self, box_bboxes):
        """Count boxes outside zone boundaries"""
        count = 0
        for bbox in box_bboxes:
            if self._is_misplaced(bbox):
                count += 1
        return count
    
    def _is_misplaced(self, bbox):
        """Check if a box is outside zone boundaries"""
        center_x = bbox['center_x']
        center_y = bbox['center_y']
        
        return (center_x < self.zone_boundaries['x_min'] or
                center_x > self.zone_boundaries['x_max'] or
                center_y < self.zone_boundaries['y_min'] or
                center_y > self.zone_boundaries['y_max'])
    
    def _detect_unsafe_stacks(self, box_bboxes):
        """
        Detect boxes stacked too high
        Simple heuristic: count vertical clusters
        """
        if len(box_bboxes) < 2:
            return 0
        
        unsafe_count = 0
        processed = set()
        
        for i, box1 in enumerate(box_bboxes):
            if i in processed:
                continue
            
            # Find boxes stacked on this one
            stack = [box1]
            
            for j, box2 in enumerate(box_bboxes):
                if i == j or j in processed:
                    continue
                
                # Check if box2 is vertically aligned and above box1
                horizontal_overlap = self._horizontal_overlap(box1, box2)
                
                if horizontal_overlap > 0.5:  # 50% overlap
                    # Check if box2 is above box1
                    if box2['y_max'] < box1['y_min'] and \
                       (box1['y_min'] - box2['y_max']) < 50:  # Close vertically
                        stack.append(box2)
                        processed.add(j)
            
            if len(stack) > self.max_stack_height:
                unsafe_count += 1
        
        return unsafe_count
    
    def _get_unsafe_stack_details(self, box_bboxes):
        """Get detailed info about unsafe stacks for logging"""
        unsafe_stacks = []
        processed = set()
        
        for i, box1 in enumerate(box_bboxes):
            if i in processed:
                continue
            
            stack = [box1]
            
            for j, box2 in enumerate(box_bboxes):
                if i == j or j in processed:
                    continue
                
                horizontal_overlap = self._horizontal_overlap(box1, box2)
                
                if horizontal_overlap > 0.5:
                    if box2['y_max'] < box1['y_min'] and \
                       (box1['y_min'] - box2['y_max']) < 50:
                        stack.append(box2)
                        processed.add(j)
            
            if len(stack) > self.max_stack_height:
                # Estimate tilt angle (simplified)
                tilt_angle = np.random.uniform(10, 25)  # Placeholder
                
                # Estimate stability (lower is worse)
                stability_score = max(0.2, 1.0 - (len(stack) - self.max_stack_height) * 0.2)
                
                unsafe_stacks.append({
                    'stack_height': len(stack),
                    'tilt_angle': tilt_angle,
                    'stability_score': stability_score,
                    'risk_level': 'HIGH' if len(stack) > 5 else 'MEDIUM'
                })
        
        return unsafe_stacks
    
    def _horizontal_overlap(self, bbox1, bbox2):
        """Calculate horizontal overlap ratio between two boxes"""
        x_left = max(bbox1['x_min'], bbox2['x_min'])
        x_right = min(bbox1['x_max'], bbox2['x_max'])
        
        if x_right < x_left:
            return 0.0
        
        overlap = x_right - x_left
        bbox1_width = bbox1['x_max'] - bbox1['x_min']
        
        return overlap / bbox1_width if bbox1_width > 0 else 0.0
    
    def _check_collision_risk(self, person_bboxes, forklift_bboxes):
        """
        Check for collision risk between person and forklift
        Returns: (bool, collision_data or None)
        """
        min_distance = float('inf')
        max_overlap = 0.0
        closest_pair = None
        
        for person in person_bboxes:
            for forklift in forklift_bboxes:
                # Calculate IoU (Intersection over Union)
                iou = self._calculate_iou(person, forklift)
                
                # Calculate center distance
                distance = np.sqrt(
                    (person['center_x'] - forklift['center_x']) ** 2 +
                    (person['center_y'] - forklift['center_y']) ** 2
                )
                
                if distance < min_distance or iou > max_overlap:
                    min_distance = distance
                    max_overlap = iou
                    closest_pair = (person, forklift)
        
        # Collision risk if:
        # 1. IoU > 0.3 (30% overlap) OR
        # 2. Distance < 150 pixels
        collision_threshold_iou = 0.3
        collision_threshold_distance = 150
        
        is_collision = (max_overlap > collision_threshold_iou or 
                       min_distance < collision_threshold_distance)
        
        if is_collision and closest_pair:
            # Determine severity
            if max_overlap > 0.6 or min_distance < 75:
                severity = 'CRITICAL'
            elif max_overlap > 0.45 or min_distance < 100:
                severity = 'HIGH'
            elif max_overlap > 0.3 or min_distance < 150:
                severity = 'MEDIUM'
            else:
                severity = 'LOW'
            
            collision_data = {
                'person_bbox': closest_pair[0],
                'forklift_bbox': closest_pair[1],
                'distance': min_distance,
                'overlap_ratio': max_overlap,
                'severity': severity
            }
            
            return True, collision_data
        
        return False, None
    
    def _calculate_iou(self, bbox1, bbox2):
        """Calculate Intersection over Union"""
        # Intersection
        x_left = max(bbox1['x_min'], bbox2['x_min'])
        y_top = max(bbox1['y_min'], bbox2['y_min'])
        x_right = min(bbox1['x_max'], bbox2['x_max'])
        y_bottom = min(bbox1['y_max'], bbox2['y_max'])
        
        if x_right < x_left or y_bottom < y_top:
            return 0.0
        
        intersection = (x_right - x_left) * (y_bottom - y_top)
        
        # Union
        bbox1_area = bbox1['width'] * bbox1['height']
        bbox2_area = bbox2['width'] * bbox2['height']
        union = bbox1_area + bbox2_area - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def save_to_database(self, detection_result):
        """
        Save detection results to database
        
        Args:
            detection_result: Output from detect_frame()
        
        Returns:
            int: detection_id
        """
        # Prepare data for database
        detection_data = {
            'box_count': detection_result['box_count'],
            'misplaced': detection_result['misplaced'],
            'unsafe_stack': detection_result['unsafe_stack'],
            'person_detected': detection_result['person_detected'],
            'forklift_detected': detection_result['forklift_detected'],
            'collision_risk': detection_result['collision_risk'],
            'avg_confidence': detection_result['avg_confidence'],
            'frame_number': detection_result['frame_number'],
            'processing_time_ms': detection_result['processing_time_ms']
        }
        
        # Convert bboxes to database format
        bboxes_for_db = []
        for bbox in detection_result['bboxes']:
            bboxes_for_db.append({
                'object_type': bbox['object_type'],
                'x_min': bbox['x_min'],
                'y_min': bbox['y_min'],
                'x_max': bbox['x_max'],
                'y_max': bbox['y_max'],
                'confidence': bbox['confidence'],
                'class_id': bbox['class_id']
            })
        
        # Save main detection
        detection_id = save_detection(
            self.zone_id,
            self.camera_id,
            detection_data,
            bboxes_for_db
        )
        
        # Log collision event if detected
        if detection_result['collision_risk'] and detection_result['collision_data']:
            cd = detection_result['collision_data']
            log_collision_event(
                detection_id,
                self.zone_id,
                person_bbox_id=1,  # Simplified - would need actual bbox IDs
                forklift_bbox_id=2,
                distance=cd['distance'],
                overlap_ratio=cd['overlap_ratio'],
                severity=cd['severity']
            )
        
        # Log misplacements
        for misplaced_box in detection_result['misplaced_boxes'][:5]:  # Log first 5
            log_misplacement(
                detection_id,
                self.zone_id,
                expected_zone='A2',  # Simplified - would need logic to determine
                x=int(misplaced_box['center_x']),
                y=int(misplaced_box['center_y']),
                severity='MEDIUM'
            )
        
        # Log unsafe stacks
        for stack in detection_result['unsafe_stacks'][:3]:  # Log first 3
            log_unsafe_stack(
                detection_id,
                self.zone_id,
                stack_height=stack['stack_height'],
                tilt_angle=stack['tilt_angle'],
                stability_score=stack['stability_score'],
                risk_level=stack['risk_level']
            )
        
        return detection_id
    
    def draw_detections(self, frame, detection_result):
        """
        Draw bounding boxes and info on frame for visualization
        
        Args:
            frame: OpenCV image
            detection_result: Output from detect_frame()
        
        Returns:
            frame: Annotated frame
        """
        annotated = frame.copy()
        
        # Draw zone boundaries
        cv2.rectangle(
            annotated,
            (self.zone_boundaries['x_min'], self.zone_boundaries['y_min']),
            (self.zone_boundaries['x_max'], self.zone_boundaries['y_max']),
            (255, 255, 0),  # Cyan
            2
        )
        
        # Draw bounding boxes
        for bbox in detection_result['bboxes']:
            x1, y1, x2, y2 = bbox['x_min'], bbox['y_min'], bbox['x_max'], bbox['y_max']
            
            # Color code by type
            if bbox['object_type'] == 'person':
                color = (0, 255, 0)  # Green
                label = 'Person'
            elif bbox['object_type'] == 'forklift':
                color = (0, 165, 255)  # Orange
                label = 'Forklift'
            else:  # box
                # Red if misplaced, blue if ok
                is_misplaced = self._is_misplaced(bbox)
                color = (0, 0, 255) if is_misplaced else (255, 0, 0)
                label = f"Box {'(MISPLACED)' if is_misplaced else ''}"
            
            # Draw box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            label_text = f"{label} {bbox['confidence']:.2f}"
            (w, h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(annotated, (x1, y1 - 20), (x1 + w, y1), color, -1)
            cv2.putText(annotated, label_text, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Draw info panel
        info_y = 30
        line_height = 30
        
        # Background for info
        cv2.rectangle(annotated, (10, 10), (400, 250), (0, 0, 0), -1)
        cv2.rectangle(annotated, (10, 10), (400, 250), (255, 255, 255), 2)
        
        # Zone info
        cv2.putText(annotated, f"Zone: {self.zone_id}", (20, info_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        info_y += line_height
        
        # Counts
        cv2.putText(annotated, f"Boxes: {detection_result['box_count']}", (20, info_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        info_y += line_height
        
        cv2.putText(annotated, f"Misplaced: {detection_result['misplaced']}", (20, info_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255) if detection_result['misplaced'] > 0 else (255, 255, 255), 2)
        info_y += line_height
        
        cv2.putText(annotated, f"Unsafe Stacks: {detection_result['unsafe_stack']}", (20, info_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255) if detection_result['unsafe_stack'] > 0 else (255, 255, 255), 2)
        info_y += line_height
        
        # Flags
        person_color = (0, 255, 0) if detection_result['person_detected'] else (100, 100, 100)
        cv2.putText(annotated, f"Person: {'YES' if detection_result['person_detected'] else 'NO'}", 
                   (20, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, person_color, 2)
        info_y += line_height
        
        forklift_color = (0, 165, 255) if detection_result['forklift_detected'] else (100, 100, 100)
        cv2.putText(annotated, f"Forklift: {'YES' if detection_result['forklift_detected'] else 'NO'}", 
                   (20, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, forklift_color, 2)
        info_y += line_height
        
        # Collision warning
        if detection_result['collision_risk']:
            cv2.putText(annotated, "COLLISION RISK!", (20, info_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Flash the entire frame border
            if self.frame_count % 10 < 5:  # Blinking effect
                cv2.rectangle(annotated, (0, 0), 
                            (annotated.shape[1]-1, annotated.shape[0]-1), 
                            (0, 0, 255), 10)
        
        return annotated
    
    def process_video(self, video_source, display=True, save_video=False, 
                     output_path='output.mp4'):
        """
        Process video stream (camera or file) with real-time detection
        
        Args:
            video_source: Camera index (0) or video file path
            display: Show live video window
            save_video: Save annotated video to file
            output_path: Output video path
        """
        print(f"\n🎥 Starting video processing from: {video_source}")
        
        cap = cv2.VideoCapture(video_source)
        
        if not cap.isOpened():
            print("❌ Error: Could not open video source")
            return
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"📹 Video: {width}x{height} @ {fps} FPS")
        
        # Video writer
        writer = None
        if save_video:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            print(f"💾 Saving output to: {output_path}")
        
        try:
            while True:
                ret, frame = cap.read()
                
                if not ret:
                    print("📹 End of video stream")
                    break
                
                # Detect objects in frame
                detection_result = self.detect_frame(frame)
                
                # Save to database every 2 seconds
                current_time = time.time()
                if current_time - self.last_save_time >= self.save_interval:
                    self.save_to_database(detection_result)
                    self.last_save_time = current_time
                    
                    # Print status
                    print(f"📊 Frame {self.frame_count}: "
                          f"Boxes={detection_result['box_count']}, "
                          f"Misplaced={detection_result['misplaced']}, "
                          f"Unsafe={detection_result['unsafe_stack']}, "
                          f"Collision={'⚠️ YES' if detection_result['collision_risk'] else 'NO'}")
                
                # Draw annotations
                annotated = self.draw_detections(frame, detection_result)
                
                # Display
                if display:
                    cv2.imshow(f'Warehouse CV - {self.zone_id}', annotated)
                    
                    # Exit on 'q' key
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        print("\n⏹️  Stopped by user")
                        break
                
                # Save to file
                if save_video and writer is not None:
                    writer.write(annotated)
        
        finally:
            cap.release()
            if writer is not None:
                writer.release()
            cv2.destroyAllWindows()
            
            print(f"\n✅ Processed {self.frame_count} frames")
            print(f"💾 Data saved to cv_detections.db")


# ============================================================================
# MAIN - Example Usage
# ============================================================================

if __name__ == '__main__':
    print("="*70)
    print("🏭 SMART WAREHOUSE COMPUTER VISION SYSTEM")
    print("="*70)
    
    # Initialize CV system
    cv_system = WarehouseCV(
        zone_id='A1',
        camera_id='CAM-01',
        zone_boundaries={
            'x_min': 200,
            'y_min': 200,
            'x_max': 1720,
            'y_max': 880
        },
        model_path='yolov8n.pt'  # Download first time: auto-downloads
    )
    
    print("\n" + "="*70)
    print("📹 CHOOSE INPUT SOURCE:")
    print("="*70)
    print("1. Webcam (camera index 0)")
    print("2. Video file")
    print("3. Test with single image")
    print("="*70)
    
    choice = input("\nEnter choice (1-3): ").strip()
    
    if choice == '1':
        print("\n🎥 Starting webcam...")
        cv_system.process_video(
            video_source=0,
            display=True,
            save_video=True,
            output_path='warehouse_output.mp4'
        )
    
    elif choice == '2':
        video_path = input("Enter video file path: ").strip()
        print(f"\n🎥 Processing video: {video_path}")
        cv_system.process_video(
            video_source=video_path,
            display=True,
            save_video=True,
            output_path='warehouse_output.mp4'
        )
    
    elif choice == '3':
        # Test with sample image
        print("\n📸 Testing with sample image...")
        
        # Create a dummy test frame
        test_frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        
        # Or load an actual image
        image_path = input("Enter image path (or press Enter for dummy): ").strip()
        if image_path:
            test_frame = cv2.imread(image_path)
        
        if test_frame is not None:
            result = cv_system.detect_frame(test_frame)
            
            print("\n📊 Detection Results:")
            print(f"  Boxes: {result['box_count']}")
            print(f"  Misplaced: {result['misplaced']}")
            print(f"  Unsafe Stacks: {result['unsafe_stack']}")
            print(f"  Person: {result['person_detected']}")
            print(f"  Forklift: {result['forklift_detected']}")
            print(f"  Collision Risk: {result['collision_risk']}")
            
            # Save to database
            detection_id = cv_system.save_to_database(result)
            print(f"\n💾 Saved to database with ID: {detection_id}")
            
            # Show annotated image
            annotated = cv_system.draw_detections(test_frame, result)
            cv2.imshow('Detection Result', annotated)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    
    print("\n✅ Done!")
