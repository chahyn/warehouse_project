"""
CV Detection Database - Helper Functions
Ready-to-use functions for storing and retrieving CV detection data
"""

import sqlite3
from datetime import datetime
from typing import Dict, List, Tuple, Optional

DB_PATH = 'cv_detections.db'

# ============================================================================
# CORE CV DETECTION FUNCTIONS
# ============================================================================

def save_detection(zone_id: str, camera_id: str, detection_data: Dict, 
                   bboxes: List[Dict] = None) -> int:
    """
    Save a complete CV detection result to database
    
    Args:
        zone_id: Zone identifier (e.g., 'A1')
        camera_id: Camera identifier (e.g., 'CAM-01')
        detection_data: {
            'box_count': int,
            'misplaced': int,
            'unsafe_stack': int,
            'person_detected': bool,
            'forklift_detected': bool,
            'collision_risk': bool,
            'avg_confidence': float (optional),
            'frame_number': int (optional),
            'processing_time_ms': float (optional)
        }
        bboxes: List of bounding boxes (optional) [{
            'object_type': str ('box', 'person', 'forklift'),
            'x_min': int, 'y_min': int, 'x_max': int, 'y_max': int,
            'confidence': float,
            'class_id': int,
            'tracking_id': int (optional)
        }]
    
    Returns:
        int: detection_id of inserted record
    
    Example:
        >>> detection = {
        ...     'box_count': 760,
        ...     'misplaced': 10,
        ...     'unsafe_stack': 5,
        ...     'person_detected': True,
        ...     'forklift_detected': True,
        ...     'collision_risk': True,
        ...     'avg_confidence': 0.92
        ... }
        >>> bboxes = [
        ...     {'object_type': 'box', 'x_min': 100, 'y_min': 100, 
        ...      'x_max': 200, 'y_max': 200, 'confidence': 0.95, 'class_id': 24},
        ...     {'object_type': 'person', 'x_min': 500, 'y_min': 300,
        ...      'x_max': 650, 'y_max': 650, 'confidence': 0.98, 'class_id': 0}
        ... ]
        >>> detection_id = save_detection('A1', 'CAM-01', detection, bboxes)
    """
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Insert main detection record
        cursor.execute('''
            INSERT INTO CVDetections 
            (zone_id, camera_id, box_count, misplaced, unsafe_stack,
             person_detected, forklift_detected, collision_risk,
             avg_detection_confidence, frame_number, processing_time_ms)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            zone_id,
            camera_id,
            detection_data.get('box_count', 0),
            detection_data.get('misplaced', 0),
            detection_data.get('unsafe_stack', 0),
            detection_data.get('person_detected', False),
            detection_data.get('forklift_detected', False),
            detection_data.get('collision_risk', False),
            detection_data.get('avg_confidence', None),
            detection_data.get('frame_number', None),
            detection_data.get('processing_time_ms', None)
        ))
        
        detection_id = cursor.lastrowid
        
        # Insert bounding boxes if provided
        if bboxes:
            for bbox in bboxes:
                cursor.execute('''
                    INSERT INTO BoundingBoxes
                    (detection_id, object_type, x_min, y_min, x_max, y_max,
                     confidence, class_id, tracking_id)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    detection_id,
                    bbox['object_type'],
                    bbox['x_min'],
                    bbox['y_min'],
                    bbox['x_max'],
                    bbox['y_max'],
                    bbox.get('confidence', None),
                    bbox.get('class_id', None),
                    bbox.get('tracking_id', None)
                ))
        
        conn.commit()
        conn.close()
        
        print(f"✅ Saved detection {detection_id} for zone {zone_id}")
        return detection_id
        
    except Exception as e:
        print(f"❌ Error saving detection: {e}")
        return -1


def log_collision_event(detection_id: int, zone_id: str, 
                       person_bbox_id: int, forklift_bbox_id: int,
                       distance: float, overlap_ratio: float,
                       severity: str = 'MEDIUM') -> bool:
    """
    Log a collision risk event
    
    Args:
        detection_id: Parent detection ID
        zone_id: Zone where collision detected
        person_bbox_id: ID of person bounding box
        forklift_bbox_id: ID of forklift bounding box
        distance: Distance in pixels
        overlap_ratio: IoU ratio (0.0 to 1.0)
        severity: 'LOW', 'MEDIUM', 'HIGH', 'CRITICAL'
    
    Returns:
        bool: Success status
    """
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO CollisionEvents
            (detection_id, zone_id, person_bbox_id, forklift_bbox_id,
             distance_pixels, overlap_ratio, severity)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (detection_id, zone_id, person_bbox_id, forklift_bbox_id,
              distance, overlap_ratio, severity))
        
        conn.commit()
        conn.close()
        
        print(f"⚠️  Logged {severity} collision event in {zone_id}")
        return True
        
    except Exception as e:
        print(f"❌ Error logging collision: {e}")
        return False


def log_misplacement(detection_id: int, zone_id: str, expected_zone: str,
                    bbox_id: int = None, x: int = None, y: int = None,
                    severity: str = 'MEDIUM') -> bool:
    """
    Log a misplaced item
    
    Args:
        detection_id: Parent detection ID
        zone_id: Current zone
        expected_zone: Where item should be
        bbox_id: Bounding box ID (optional)
        x, y: Position coordinates (optional)
        severity: 'LOW', 'MEDIUM', 'HIGH'
    
    Returns:
        bool: Success status
    """
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO MisplacementLog
            (detection_id, zone_id, expected_zone, bbox_id,
             actual_position_x, actual_position_y, severity)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (detection_id, zone_id, expected_zone, bbox_id, x, y, severity))
        
        conn.commit()
        conn.close()
        
        print(f"📍 Logged misplacement: {zone_id} → {expected_zone}")
        return True
        
    except Exception as e:
        print(f"❌ Error logging misplacement: {e}")
        return False


def log_unsafe_stack(detection_id: int, zone_id: str,
                    stack_height: int, tilt_angle: float = None,
                    stability_score: float = None,
                    risk_level: str = 'MEDIUM') -> bool:
    """
    Log an unsafe stack incident
    
    Args:
        detection_id: Parent detection ID
        zone_id: Zone identifier
        stack_height: Number of boxes in stack
        tilt_angle: Tilt in degrees (optional)
        stability_score: 0.0 to 1.0, lower is worse (optional)
        risk_level: 'LOW', 'MEDIUM', 'HIGH'
    
    Returns:
        bool: Success status
    """
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO UnsafeStackIncidents
            (detection_id, zone_id, stack_height, tilt_angle,
             stability_score, risk_level)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (detection_id, zone_id, stack_height, tilt_angle,
              stability_score, risk_level))
        
        conn.commit()
        conn.close()
        
        print(f"⚡ Logged unsafe stack in {zone_id}: height={stack_height}")
        return True
        
    except Exception as e:
        print(f"❌ Error logging unsafe stack: {e}")
        return False


# ============================================================================
# DATA RETRIEVAL FUNCTIONS
# ============================================================================

def get_latest_detections() -> List[Dict]:
    """
    Get latest detection for each zone
    
    Returns:
        List of dicts with detection data
    """
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM LatestDetections")
        
        columns = [desc[0] for desc in cursor.description]
        results = [dict(zip(columns, row)) for row in cursor.fetchall()]
        
        conn.close()
        return results
        
    except Exception as e:
        print(f"❌ Error fetching latest detections: {e}")
        return []


def get_active_collision_risks() -> List[Dict]:
    """
    Get all unacknowledged collision risks
    
    Returns:
        List of active collision events
    """
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM ActiveCollisionRisks")
        
        columns = [desc[0] for desc in cursor.description]
        results = [dict(zip(columns, row)) for row in cursor.fetchall()]
        
        conn.close()
        return results
        
    except Exception as e:
        print(f"❌ Error fetching collision risks: {e}")
        return []


def get_unresolved_misplacements() -> List[Dict]:
    """
    Get all items that haven't been corrected
    
    Returns:
        List of misplacement records
    """
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM UnresolvedMisplacements")
        
        columns = [desc[0] for desc in cursor.description]
        results = [dict(zip(columns, row)) for row in cursor.fetchall()]
        
        conn.close()
        return results
        
    except Exception as e:
        print(f"❌ Error fetching misplacements: {e}")
        return []


def get_active_unsafe_stacks() -> List[Dict]:
    """
    Get all unresolved unsafe stack incidents
    
    Returns:
        List of unsafe stack records
    """
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM ActiveUnsafeStacks")
        
        columns = [desc[0] for desc in cursor.description]
        results = [dict(zip(columns, row)) for row in cursor.fetchall()]
        
        conn.close()
        return results
        
    except Exception as e:
        print(f"❌ Error fetching unsafe stacks: {e}")
        return []


def get_hourly_summary(zone_id: str = None, hours: int = 24) -> List[Dict]:
    """
    Get hourly detection statistics
    
    Args:
        zone_id: Filter by zone (optional)
        hours: Last N hours (default 24)
    
    Returns:
        List of hourly statistics
    """
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        if zone_id:
            cursor.execute(f"""
                SELECT * FROM HourlyDetectionSummary
                WHERE zone_id = ?
                AND datetime(date || ' ' || printf('%02d', hour) || ':00:00') 
                    >= datetime('now', '-{hours} hours')
                ORDER BY date DESC, hour DESC
            """, (zone_id,))
        else:
            cursor.execute(f"""
                SELECT * FROM HourlyDetectionSummary
                WHERE datetime(date || ' ' || printf('%02d', hour) || ':00:00') 
                    >= datetime('now', '-{hours} hours')
                ORDER BY date DESC, hour DESC
            """)
        
        columns = [desc[0] for desc in cursor.description]
        results = [dict(zip(columns, row)) for row in cursor.fetchall()]
        
        conn.close()
        return results
        
    except Exception as e:
        print(f"❌ Error fetching hourly summary: {e}")
        return []


def get_camera_status() -> List[Dict]:
    """
    Get current status of all cameras
    
    Returns:
        List of camera status records
    """
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM CameraStatus ORDER BY camera_id")
        
        columns = [desc[0] for desc in cursor.description]
        results = [dict(zip(columns, row)) for row in cursor.fetchall()]
        
        conn.close()
        return results
        
    except Exception as e:
        print(f"❌ Error fetching camera status: {e}")
        return []


def get_detection_history(zone_id: str, hours: int = 1) -> List[Dict]:
    """
    Get detection history for a zone
    
    Args:
        zone_id: Zone identifier
        hours: Last N hours
    
    Returns:
        List of detection records
    """
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute(f"""
            SELECT 
                detection_id,
                timestamp,
                box_count,
                misplaced,
                unsafe_stack,
                person_detected,
                forklift_detected,
                collision_risk,
                avg_detection_confidence
            FROM CVDetections
            WHERE zone_id = ?
            AND timestamp >= datetime('now', '-{hours} hours')
            ORDER BY timestamp DESC
        """, (zone_id,))
        
        columns = [desc[0] for desc in cursor.description]
        results = [dict(zip(columns, row)) for row in cursor.fetchall()]
        
        conn.close()
        return results
        
    except Exception as e:
        print(f"❌ Error fetching detection history: {e}")
        return []


# ============================================================================
# ANALYTICS FUNCTIONS
# ============================================================================

def get_zone_analytics(zone_id: str) -> Dict:
    """
    Get comprehensive analytics for a zone
    
    Args:
        zone_id: Zone identifier
    
    Returns:
        Dict with analytics data
    """
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Get latest detection
        cursor.execute("""
            SELECT box_count, misplaced, unsafe_stack, collision_risk
            FROM CVDetections
            WHERE zone_id = ?
            ORDER BY timestamp DESC
            LIMIT 1
        """, (zone_id,))
        
        latest = cursor.fetchone()
        
        # Get 24h statistics
        cursor.execute("""
            SELECT 
                COUNT(*) as total_detections,
                AVG(box_count) as avg_boxes,
                MAX(box_count) as max_boxes,
                MIN(box_count) as min_boxes,
                SUM(misplaced) as total_misplaced,
                SUM(unsafe_stack) as total_unsafe,
                SUM(CASE WHEN collision_risk = 1 THEN 1 ELSE 0 END) as collision_count
            FROM CVDetections
            WHERE zone_id = ?
            AND timestamp >= datetime('now', '-24 hours')
        """, (zone_id,))
        
        stats_24h = cursor.fetchone()
        
        conn.close()
        
        return {
            'zone_id': zone_id,
            'current': {
                'box_count': latest[0] if latest else 0,
                'misplaced': latest[1] if latest else 0,
                'unsafe_stack': latest[2] if latest else 0,
                'collision_risk': bool(latest[3]) if latest else False
            },
            'last_24h': {
                'total_detections': stats_24h[0],
                'avg_boxes': round(stats_24h[1], 2) if stats_24h[1] else 0,
                'max_boxes': stats_24h[2] or 0,
                'min_boxes': stats_24h[3] or 0,
                'total_misplaced': stats_24h[4] or 0,
                'total_unsafe': stats_24h[5] or 0,
                'collision_events': stats_24h[6] or 0
            }
        }
        
    except Exception as e:
        print(f"❌ Error getting zone analytics: {e}")
        return {}


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def acknowledge_collision(event_id: int) -> bool:
    """Mark a collision event as acknowledged"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute("""
            UPDATE CollisionEvents 
            SET acknowledged = 1 
            WHERE event_id = ?
        """, (event_id,))
        
        conn.commit()
        conn.close()
        
        print(f"✅ Acknowledged collision event {event_id}")
        return True
        
    except Exception as e:
        print(f"❌ Error acknowledging collision: {e}")
        return False


def mark_misplacement_corrected(misplacement_id: int) -> bool:
    """Mark a misplacement as corrected"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute("""
            UPDATE MisplacementLog 
            SET corrected = 1 
            WHERE misplacement_id = ?
        """, (misplacement_id,))
        
        conn.commit()
        conn.close()
        
        print(f"✅ Marked misplacement {misplacement_id} as corrected")
        return True
        
    except Exception as e:
        print(f"❌ Error updating misplacement: {e}")
        return False


def resolve_unsafe_stack(incident_id: int) -> bool:
    """Mark an unsafe stack as resolved"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute("""
            UPDATE UnsafeStackIncidents 
            SET resolved = 1 
            WHERE incident_id = ?
        """, (incident_id,))
        
        conn.commit()
        conn.close()
        
        print(f"✅ Resolved unsafe stack incident {incident_id}")
        return True
        
    except Exception as e:
        print(f"❌ Error resolving incident: {e}")
        return False


# ============================================================================
# DEMO/TEST FUNCTIONS
# ============================================================================

def simulate_yolo_detection_cycle(zone_id: str = 'A1'):
    """
    Simulate a complete YOLO detection cycle
    Demonstrates the full workflow
    """
    print(f"\n🎥 Simulating YOLO detection for {zone_id}...")
    
    # Simulated YOLO output
    detection_data = {
        'box_count': 760,
        'misplaced': 10,
        'unsafe_stack': 5,
        'person_detected': True,
        'forklift_detected': True,
        'collision_risk': True,
        'avg_confidence': 0.92,
        'frame_number': 12345,
        'processing_time_ms': 28.5
    }
    
    # Simulated bounding boxes
    bboxes = [
        {'object_type': 'person', 'x_min': 500, 'y_min': 300,
         'x_max': 650, 'y_max': 650, 'confidence': 0.98, 'class_id': 0},
        {'object_type': 'forklift', 'x_min': 550, 'y_min': 350,
         'x_max': 950, 'y_max': 650, 'confidence': 0.87, 'class_id': 2},
        {'object_type': 'box', 'x_min': 100, 'y_min': 100,
         'x_max': 200, 'y_max': 200, 'confidence': 0.95, 'class_id': 24}
    ]
    
    # Save detection
    detection_id = save_detection(zone_id, f'CAM-0{zone_id[1]}', 
                                  detection_data, bboxes)
    
    if detection_id > 0:
        # Log events
        if detection_data['collision_risk']:
            log_collision_event(detection_id, zone_id, 1, 2, 
                              distance=75.5, overlap_ratio=0.45, 
                              severity='HIGH')
        
        if detection_data['misplaced'] > 0:
            log_misplacement(detection_id, zone_id, 'A2',
                           x=150, y=150, severity='MEDIUM')
        
        if detection_data['unsafe_stack'] > 0:
            log_unsafe_stack(detection_id, zone_id,
                           stack_height=4, tilt_angle=18.5,
                           stability_score=0.35, risk_level='HIGH')
    
    print("✅ Detection cycle complete!\n")
    return detection_id


def print_dashboard_preview():
    """Print a text-based dashboard preview"""
    print("\n" + "="*70)
    print("📊 CV DETECTION DASHBOARD PREVIEW")
    print("="*70)
    
    # Latest detections
    print("\n📷 Latest Detections:")
    detections = get_latest_detections()
    for d in detections:
        icon = "⚠️ " if d['collision_risk'] else "✅"
        print(f"  {icon} {d['zone_id']} ({d['zone_name']}): "
              f"{d['box_count']} boxes | Misplaced: {d['misplaced']} | "
              f"Unsafe: {d['unsafe_stack']}")
    
    # Active collision risks
    print("\n🚨 Active Collision Risks:")
    collisions = get_active_collision_risks()
    for c in collisions[:5]:
        print(f"  🔴 {c['zone_id']}: {c['severity']} "
              f"(overlap: {c['overlap_ratio']:.2f}, {c['hours_ago']}h ago)")
    
    # Unresolved issues
    print("\n📍 Unresolved Issues:")
    misplacements = get_unresolved_misplacements()
    print(f"  - Misplacements: {len(misplacements)}")
    unsafe = get_active_unsafe_stacks()
    print(f"  - Unsafe Stacks: {len(unsafe)}")
    
    # Camera status
    print("\n📷 Camera Status:")
    cameras = get_camera_status()
    for cam in cameras:
        icon = "🟢" if cam['status'] == 'ONLINE' else "🟡"
        print(f"  {icon} {cam['camera_id']}: {cam['status']} @ {cam['fps']} FPS")
    
    print("\n" + "="*70)


# ============================================================================
# MAIN - For testing
# ============================================================================

if __name__ == '__main__':
    print("🧪 Testing CV Detection Database Functions\n")
    
    # Test 1: Simulate detection
    simulate_yolo_detection_cycle('A1')
    
    # Test 2: Get analytics
    print("📊 Zone Analytics for A1:")
    analytics = get_zone_analytics('A1')
    print(f"  Current boxes: {analytics['current']['box_count']}")
    print(f"  24h average: {analytics['last_24h']['avg_boxes']}")
    print(f"  Collision events (24h): {analytics['last_24h']['collision_events']}")
    
    # Test 3: Dashboard preview
    print_dashboard_preview()
    
    print("\n✅ All tests complete!")
