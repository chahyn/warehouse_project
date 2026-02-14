"""
CV Detection Database Generator
Creates a database to store all Computer Vision detection results
with realistic time-series data for demo and analysis
"""

import sqlite3
import random
from datetime import datetime, timedelta
import json

def create_cv_database(conn):
    """Create tables specifically for CV detection storage"""
    cursor = conn.cursor()
    
    cursor.execute('PRAGMA foreign_keys = ON;')
    
    # ========================================================================
    # TABLE 1: Zone Configuration (Camera Setup)
    # ========================================================================
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS ZoneConfig (
        zone_id TEXT PRIMARY KEY,
        zone_name TEXT NOT NULL,
        camera_id TEXT NOT NULL,
        component_type TEXT,
        x_min INTEGER,
        y_min INTEGER,
        x_max INTEGER,
        y_max INTEGER,
        max_stack_height INTEGER DEFAULT 3,
        area_sqm REAL,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
    );
    ''')
    
    # ========================================================================
    # TABLE 2: Real-Time CV Detections (Main storage for YOLO output)
    # ========================================================================
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS CVDetections (
        detection_id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
        zone_id TEXT NOT NULL,
        camera_id TEXT NOT NULL,
        
        -- Box/Inventory Counts
        box_count INTEGER DEFAULT 0,
        misplaced INTEGER DEFAULT 0,
        unsafe_stack INTEGER DEFAULT 0,
        
        -- Object Detection Flags
        person_detected BOOLEAN DEFAULT 0,
        forklift_detected BOOLEAN DEFAULT 0,
        
        -- Computed Risk
        collision_risk BOOLEAN DEFAULT 0,
        
        -- Confidence Scores (optional but useful)
        avg_detection_confidence REAL,
        
        -- Processing metadata
        frame_number INTEGER,
        processing_time_ms REAL,
        
        FOREIGN KEY (zone_id) REFERENCES ZoneConfig(zone_id)
    );
    ''')
    
    # ========================================================================
    # TABLE 3: Bounding Box Details (Detailed object tracking)
    # ========================================================================
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS BoundingBoxes (
        bbox_id INTEGER PRIMARY KEY AUTOINCREMENT,
        detection_id INTEGER NOT NULL,
        object_type TEXT NOT NULL,
        x_min INTEGER NOT NULL,
        y_min INTEGER NOT NULL,
        x_max INTEGER NOT NULL,
        y_max INTEGER NOT NULL,
        confidence REAL,
        class_id INTEGER,
        tracking_id INTEGER,
        
        FOREIGN KEY (detection_id) REFERENCES CVDetections(detection_id)
    );
    ''')
    
    # ========================================================================
    # TABLE 4: Collision Events (Safety tracking)
    # ========================================================================
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS CollisionEvents (
        event_id INTEGER PRIMARY KEY AUTOINCREMENT,
        detection_id INTEGER NOT NULL,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
        zone_id TEXT NOT NULL,
        
        person_bbox_id INTEGER,
        forklift_bbox_id INTEGER,
        
        distance_pixels REAL,
        overlap_ratio REAL,
        severity TEXT CHECK(severity IN ('LOW', 'MEDIUM', 'HIGH', 'CRITICAL')),
        
        alert_sent BOOLEAN DEFAULT 0,
        acknowledged BOOLEAN DEFAULT 0,
        
        FOREIGN KEY (detection_id) REFERENCES CVDetections(detection_id),
        FOREIGN KEY (zone_id) REFERENCES ZoneConfig(zone_id)
    );
    ''')
    
    # ========================================================================
    # TABLE 5: Misplacement Log
    # ========================================================================
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS MisplacementLog (
        misplacement_id INTEGER PRIMARY KEY AUTOINCREMENT,
        detection_id INTEGER NOT NULL,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
        zone_id TEXT NOT NULL,
        
        bbox_id INTEGER,
        expected_zone TEXT,
        actual_position_x INTEGER,
        actual_position_y INTEGER,
        
        severity TEXT DEFAULT 'MEDIUM',
        corrected BOOLEAN DEFAULT 0,
        
        FOREIGN KEY (detection_id) REFERENCES CVDetections(detection_id),
        FOREIGN KEY (bbox_id) REFERENCES BoundingBoxes(bbox_id)
    );
    ''')
    
    # ========================================================================
    # TABLE 6: Unsafe Stack Incidents
    # ========================================================================
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS UnsafeStackIncidents (
        incident_id INTEGER PRIMARY KEY AUTOINCREMENT,
        detection_id INTEGER NOT NULL,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
        zone_id TEXT NOT NULL,
        
        stack_height INTEGER,
        tilt_angle REAL,
        stability_score REAL,
        
        risk_level TEXT CHECK(risk_level IN ('LOW', 'MEDIUM', 'HIGH')),
        resolved BOOLEAN DEFAULT 0,
        
        FOREIGN KEY (detection_id) REFERENCES CVDetections(detection_id)
    );
    ''')
    
    # ========================================================================
    # TABLE 7: Detection Statistics (Aggregated metrics)
    # ========================================================================
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS DetectionStatistics (
        stat_id INTEGER PRIMARY KEY AUTOINCREMENT,
        zone_id TEXT NOT NULL,
        date DATE NOT NULL,
        hour INTEGER NOT NULL,
        
        total_detections INTEGER DEFAULT 0,
        avg_box_count REAL,
        max_box_count INTEGER,
        min_box_count INTEGER,
        
        total_misplacements INTEGER DEFAULT 0,
        total_unsafe_stacks INTEGER DEFAULT 0,
        total_collision_events INTEGER DEFAULT 0,
        
        person_detection_rate REAL,
        forklift_detection_rate REAL,
        
        UNIQUE(zone_id, date, hour),
        FOREIGN KEY (zone_id) REFERENCES ZoneConfig(zone_id)
    );
    ''')
    
    # ========================================================================
    # TABLE 8: Camera Health/Status
    # ========================================================================
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS CameraStatus (
        status_id INTEGER PRIMARY KEY AUTOINCREMENT,
        camera_id TEXT NOT NULL,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
        
        status TEXT CHECK(status IN ('ONLINE', 'OFFLINE', 'DEGRADED', 'ERROR')),
        fps REAL,
        resolution_width INTEGER,
        resolution_height INTEGER,
        
        last_detection_time DATETIME,
        error_message TEXT
    );
    ''')
    
    conn.commit()
    print("✅ CV detection database tables created successfully!")


def populate_cv_database(conn):
    """Populate database with realistic random detection data"""
    cursor = conn.cursor()
    
    # ========================================================================
    # POPULATE: Zone Configuration
    # ========================================================================
    zones = [
        ('A1', 'Piston Storage', 'CAM-01', 'piston', 0, 0, 1920, 1080, 3, 150.5),
        ('A2', 'Bolt Storage', 'CAM-02', 'bolt', 0, 0, 1920, 1080, 4, 80.0),
        ('A3', 'Gasket Storage', 'CAM-03', 'gasket', 0, 0, 1920, 1080, 3, 60.0),
        ('B1', 'Gear Storage', 'CAM-04', 'gear', 0, 0, 1920, 1080, 3, 120.0),
        ('B2', 'Bearing Storage', 'CAM-05', 'bearing', 0, 0, 1920, 1080, 4, 90.0),
        ('C1', 'Brake Components', 'CAM-06', 'brake_pad', 0, 0, 1920, 1080, 3, 100.0),
    ]
    
    cursor.executemany('''
        INSERT INTO ZoneConfig 
        (zone_id, zone_name, camera_id, component_type, x_min, y_min, x_max, y_max, 
         max_stack_height, area_sqm)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', zones)
    
    conn.commit()
    print("✅ Zone configuration populated")
    
    # ========================================================================
    # POPULATE: Real-Time CV Detections (Last 48 hours)
    # ========================================================================
    print("📊 Generating CV detection history (48 hours)...")
    
    zone_ids = [z[0] for z in zones]
    camera_ids = [z[2] for z in zones]
    
    detections = []
    bounding_boxes = []
    collision_events = []
    misplacement_log = []
    unsafe_stack_incidents = []
    
    detection_id_counter = 1
    bbox_id_counter = 1
    
    # Generate detections for last 48 hours, every 2 seconds
    start_time = datetime.now() - timedelta(hours=48)
    
    for hour in range(48):
        for minute in range(0, 60, 2):  # Every 2 minutes for demo (not every 2 seconds)
            timestamp = start_time + timedelta(hours=hour, minutes=minute)
            
            # Each zone gets a detection
            for zone_id, camera_id in zip(zone_ids, camera_ids):
                
                # Simulate realistic patterns
                is_work_hours = 8 <= timestamp.hour <= 17
                is_peak_hours = 10 <= timestamp.hour <= 12 or 14 <= timestamp.hour <= 16
                
                # Box count varies by time
                base_count = 800
                variation = random.randint(-100, 100)
                
                # Gradual depletion over 48 hours (simulate consumption)
                depletion = int((hour / 48) * 200)
                
                box_count = max(base_count + variation - depletion, 500)
                
                # Misplaced boxes (2-15)
                misplaced = random.randint(2, 15) if random.random() < 0.7 else 0
                
                # Unsafe stacks (0-8)
                unsafe_stack = random.randint(0, 8) if random.random() < 0.4 else 0
                
                # Person detection (more likely during work hours)
                person_detected = random.random() < (0.8 if is_work_hours else 0.1)
                
                # Forklift detection (less frequent)
                forklift_detected = random.random() < (0.3 if is_work_hours else 0.05)
                
                # Collision risk (only if both person and forklift)
                collision_risk = person_detected and forklift_detected and random.random() < 0.4
                
                # Average confidence
                avg_confidence = round(random.uniform(0.75, 0.98), 2)
                
                # Processing time
                processing_time = round(random.uniform(15, 45), 2)
                
                # Frame number
                frame_number = hour * 1800 + minute * 30 + random.randint(0, 29)
                
                detections.append((
                    timestamp, zone_id, camera_id,
                    box_count, misplaced, unsafe_stack,
                    person_detected, forklift_detected, collision_risk,
                    avg_confidence, frame_number, processing_time
                ))
                
                # ============================================================
                # Generate Bounding Boxes for this detection
                # ============================================================
                current_detection_id = detection_id_counter
                
                # Boxes
                for _ in range(min(box_count, 20)):  # Store max 20 boxes per frame
                    x_min = random.randint(100, 1500)
                    y_min = random.randint(100, 800)
                    width = random.randint(50, 150)
                    height = random.randint(50, 150)
                    
                    bounding_boxes.append((
                        current_detection_id,
                        'box',
                        x_min, y_min, x_min + width, y_min + height,
                        round(random.uniform(0.7, 0.99), 2),
                        24,  # YOLO class ID for backpack/box
                        random.randint(1, 1000)
                    ))
                    bbox_id_counter += 1
                
                # Person
                person_bbox_id = None
                if person_detected:
                    x_min = random.randint(200, 1400)
                    y_min = random.randint(200, 700)
                    
                    person_bbox_id = bbox_id_counter
                    bounding_boxes.append((
                        current_detection_id,
                        'person',
                        x_min, y_min, x_min + 150, y_min + 350,
                        round(random.uniform(0.85, 0.99), 2),
                        0,  # YOLO class ID for person
                        random.randint(1, 100)
                    ))
                    bbox_id_counter += 1
                
                # Forklift
                forklift_bbox_id = None
                if forklift_detected:
                    x_min = random.randint(300, 1300)
                    y_min = random.randint(300, 600)
                    
                    forklift_bbox_id = bbox_id_counter
                    bounding_boxes.append((
                        current_detection_id,
                        'forklift',
                        x_min, y_min, x_min + 400, y_min + 300,
                        round(random.uniform(0.75, 0.95), 2),
                        2,  # YOLO class ID for truck
                        random.randint(1, 50)
                    ))
                    bbox_id_counter += 1
                
                # ============================================================
                # Log Collision Event
                # ============================================================
                if collision_risk and person_bbox_id and forklift_bbox_id:
                    distance = round(random.uniform(50, 200), 2)
                    overlap = round(random.uniform(0.3, 0.7), 2)
                    
                    if overlap > 0.6:
                        severity = 'CRITICAL'
                    elif overlap > 0.45:
                        severity = 'HIGH'
                    elif overlap > 0.3:
                        severity = 'MEDIUM'
                    else:
                        severity = 'LOW'
                    
                    collision_events.append((
                        current_detection_id, timestamp, zone_id,
                        person_bbox_id, forklift_bbox_id,
                        distance, overlap, severity,
                        random.choice([True, False]),  # alert_sent
                        random.choice([True, False])   # acknowledged
                    ))
                
                # ============================================================
                # Log Misplacements
                # ============================================================
                if misplaced > 0:
                    for _ in range(min(misplaced, 5)):  # Log up to 5
                        expected_zone = random.choice([z for z in zone_ids if z != zone_id])
                        
                        misplacement_log.append((
                            current_detection_id, timestamp, zone_id,
                            random.randint(1, bbox_id_counter),
                            expected_zone,
                            random.randint(0, 1920),
                            random.randint(0, 1080),
                            random.choice(['LOW', 'MEDIUM', 'HIGH']),
                            random.choice([True, False])  # corrected
                        ))
                
                # ============================================================
                # Log Unsafe Stacks
                # ============================================================
                if unsafe_stack > 0:
                    for _ in range(min(unsafe_stack, 3)):
                        stack_height = random.randint(4, 7)
                        tilt_angle = round(random.uniform(5, 35), 2)
                        stability = round(random.uniform(0.2, 0.6), 2)
                        
                        if stability < 0.3:
                            risk = 'HIGH'
                        elif stability < 0.5:
                            risk = 'MEDIUM'
                        else:
                            risk = 'LOW'
                        
                        unsafe_stack_incidents.append((
                            current_detection_id, timestamp, zone_id,
                            stack_height, tilt_angle, stability,
                            risk, random.choice([True, False])
                        ))
                
                detection_id_counter += 1
    
    # Insert all detections
    print(f"📝 Inserting {len(detections)} detection records...")
    cursor.executemany('''
        INSERT INTO CVDetections 
        (timestamp, zone_id, camera_id, box_count, misplaced, unsafe_stack,
         person_detected, forklift_detected, collision_risk,
         avg_detection_confidence, frame_number, processing_time_ms)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', detections)
    
    print(f"📦 Inserting {len(bounding_boxes)} bounding box records...")
    cursor.executemany('''
        INSERT INTO BoundingBoxes
        (detection_id, object_type, x_min, y_min, x_max, y_max, confidence, class_id, tracking_id)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', bounding_boxes)
    
    print(f"⚠️  Inserting {len(collision_events)} collision events...")
    cursor.executemany('''
        INSERT INTO CollisionEvents
        (detection_id, timestamp, zone_id, person_bbox_id, forklift_bbox_id,
         distance_pixels, overlap_ratio, severity, alert_sent, acknowledged)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', collision_events)
    
    print(f"📍 Inserting {len(misplacement_log)} misplacement logs...")
    cursor.executemany('''
        INSERT INTO MisplacementLog
        (detection_id, timestamp, zone_id, bbox_id, expected_zone,
         actual_position_x, actual_position_y, severity, corrected)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', misplacement_log)
    
    print(f"⚡ Inserting {len(unsafe_stack_incidents)} unsafe stack incidents...")
    cursor.executemany('''
        INSERT INTO UnsafeStackIncidents
        (detection_id, timestamp, zone_id, stack_height, tilt_angle,
         stability_score, risk_level, resolved)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    ''', unsafe_stack_incidents)
    
    conn.commit()
    
    # ========================================================================
    # POPULATE: Camera Status
    # ========================================================================
    camera_statuses = [
        ('CAM-01', 'ONLINE', 30.0, 1920, 1080, datetime.now() - timedelta(minutes=2), None),
        ('CAM-02', 'ONLINE', 29.5, 1920, 1080, datetime.now() - timedelta(minutes=2), None),
        ('CAM-03', 'ONLINE', 30.0, 1920, 1080, datetime.now() - timedelta(minutes=2), None),
        ('CAM-04', 'DEGRADED', 15.2, 1920, 1080, datetime.now() - timedelta(minutes=5), 'Low light conditions'),
        ('CAM-05', 'ONLINE', 28.8, 1920, 1080, datetime.now() - timedelta(minutes=2), None),
        ('CAM-06', 'ONLINE', 30.0, 1920, 1080, datetime.now() - timedelta(minutes=2), None),
    ]
    
    cursor.executemany('''
        INSERT INTO CameraStatus
        (camera_id, status, fps, resolution_width, resolution_height, last_detection_time, error_message)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', camera_statuses)
    
    conn.commit()
    print("✅ Camera status populated")


def create_useful_views(conn):
    """Create SQL views for easy data access"""
    cursor = conn.cursor()
    
    # View 1: Latest detection per zone
    cursor.execute('''
    CREATE VIEW IF NOT EXISTS LatestDetections AS
    SELECT 
        z.zone_id,
        z.zone_name,
        z.camera_id,
        d.timestamp,
        d.box_count,
        d.misplaced,
        d.unsafe_stack,
        d.person_detected,
        d.forklift_detected,
        d.collision_risk,
        d.avg_detection_confidence
    FROM ZoneConfig z
    LEFT JOIN CVDetections d ON z.zone_id = d.zone_id
    WHERE d.detection_id IN (
        SELECT MAX(detection_id) 
        FROM CVDetections 
        GROUP BY zone_id
    );
    ''')
    
    # View 2: Active collision risks
    cursor.execute('''
    CREATE VIEW IF NOT EXISTS ActiveCollisionRisks AS
    SELECT 
        c.event_id,
        c.zone_id,
        c.timestamp,
        c.severity,
        c.overlap_ratio,
        c.distance_pixels,
        c.alert_sent,
        c.acknowledged,
        ROUND((julianday('now') - julianday(c.timestamp)) * 24, 1) as hours_ago
    FROM CollisionEvents c
    WHERE c.acknowledged = 0
    ORDER BY c.severity DESC, c.timestamp DESC;
    ''')
    
    # View 3: Unresolved misplacements
    cursor.execute('''
    CREATE VIEW IF NOT EXISTS UnresolvedMisplacements AS
    SELECT 
        m.misplacement_id,
        m.zone_id,
        m.expected_zone,
        m.timestamp,
        m.severity,
        ROUND((julianday('now') - julianday(m.timestamp)) * 24, 1) as hours_ago
    FROM MisplacementLog m
    WHERE m.corrected = 0
    ORDER BY m.severity DESC, m.timestamp DESC;
    ''')
    
    # View 4: Active unsafe stacks
    cursor.execute('''
    CREATE VIEW IF NOT EXISTS ActiveUnsafeStacks AS
    SELECT 
        u.incident_id,
        u.zone_id,
        u.timestamp,
        u.stack_height,
        u.tilt_angle,
        u.stability_score,
        u.risk_level,
        ROUND((julianday('now') - julianday(u.timestamp)) * 24, 1) as hours_ago
    FROM UnsafeStackIncidents u
    WHERE u.resolved = 0
    ORDER BY u.risk_level DESC, u.timestamp DESC;
    ''')
    
    # View 5: Hourly detection summary
    cursor.execute('''
    CREATE VIEW IF NOT EXISTS HourlyDetectionSummary AS
    SELECT 
        zone_id,
        DATE(timestamp) as date,
        strftime('%H', timestamp) as hour,
        COUNT(*) as detection_count,
        ROUND(AVG(box_count), 2) as avg_boxes,
        SUM(misplaced) as total_misplaced,
        SUM(unsafe_stack) as total_unsafe,
        SUM(CASE WHEN collision_risk = 1 THEN 1 ELSE 0 END) as collision_events,
        ROUND(AVG(avg_detection_confidence), 3) as avg_confidence,
        ROUND(AVG(processing_time_ms), 2) as avg_processing_time
    FROM CVDetections
    GROUP BY zone_id, DATE(timestamp), strftime('%H', timestamp)
    ORDER BY date DESC, hour DESC;
    ''')
    
    conn.commit()
    print("✅ Useful views created")


def print_database_summary(conn):
    """Print summary statistics"""
    cursor = conn.cursor()
    
    print("\n" + "="*70)
    print("📊 CV DETECTION DATABASE SUMMARY")
    print("="*70)
    
    # Total detections
    cursor.execute("SELECT COUNT(*) FROM CVDetections")
    total_detections = cursor.fetchone()[0]
    print(f"\n🎥 Total Detection Records: {total_detections:,}")
    
    # Total bounding boxes
    cursor.execute("SELECT COUNT(*) FROM BoundingBoxes")
    total_boxes = cursor.fetchone()[0]
    print(f"📦 Total Bounding Boxes: {total_boxes:,}")
    
    # Collision events
    cursor.execute("SELECT COUNT(*) FROM CollisionEvents")
    total_collisions = cursor.fetchone()[0]
    print(f"⚠️  Total Collision Events: {total_collisions}")
    
    cursor.execute("SELECT COUNT(*) FROM CollisionEvents WHERE acknowledged=0")
    active_collisions = cursor.fetchone()[0]
    print(f"🔴 Active Collision Alerts: {active_collisions}")
    
    # Misplacements
    cursor.execute("SELECT COUNT(*) FROM MisplacementLog WHERE corrected=0")
    active_misplacements = cursor.fetchone()[0]
    print(f"📍 Unresolved Misplacements: {active_misplacements}")
    
    # Unsafe stacks
    cursor.execute("SELECT COUNT(*) FROM UnsafeStackIncidents WHERE resolved=0")
    active_unsafe = cursor.fetchone()[0]
    print(f"⚡ Active Unsafe Stacks: {active_unsafe}")
    
    print("\n" + "="*70)
    print("📷 CAMERA STATUS")
    print("="*70)
    
    cursor.execute("""
        SELECT camera_id, status, fps, 
               strftime('%H:%M:%S', last_detection_time) as last_seen
        FROM CameraStatus
        ORDER BY camera_id
    """)
    
    for row in cursor.fetchall():
        status_icon = "🟢" if row[1] == "ONLINE" else "🟡" if row[1] == "DEGRADED" else "🔴"
        print(f"{status_icon} {row[0]}: {row[1]} | {row[2]} FPS | Last: {row[3]}")
    
    print("\n" + "="*70)
    print("📊 LATEST DETECTION PER ZONE")
    print("="*70)
    
    cursor.execute("SELECT * FROM LatestDetections ORDER BY zone_id")
    
    for row in cursor.fetchall():
        collision_icon = "⚠️ " if row[9] else "✅"
        print(f"{collision_icon} {row[0]} ({row[1]}): {row[4]} boxes | "
              f"Misplaced: {row[5]} | Unsafe: {row[6]}")
    
    print("\n" + "="*70)
    print("🚨 ACTIVE SAFETY ALERTS")
    print("="*70)
    
    # Critical collisions
    cursor.execute("""
        SELECT zone_id, severity, hours_ago 
        FROM ActiveCollisionRisks 
        WHERE severity IN ('CRITICAL', 'HIGH')
        LIMIT 5
    """)
    
    for row in cursor.fetchall():
        print(f"🔴 {row[0]}: {row[1]} collision risk ({row[2]}h ago)")
    
    # High-risk unsafe stacks
    cursor.execute("""
        SELECT zone_id, stack_height, risk_level, hours_ago
        FROM ActiveUnsafeStacks
        WHERE risk_level = 'HIGH'
        LIMIT 3
    """)
    
    for row in cursor.fetchall():
        print(f"⚡ {row[0]}: Stack height {row[1]} - {row[2]} risk ({row[3]}h ago)")
    
    print("\n" + "="*70)


def main():
    """Main execution"""
    db_name = 'cv_detections.db'
    
    print("🔧 Creating CV Detection Database...")
    conn = sqlite3.connect(db_name)
    
    # Create tables
    create_cv_database(conn)
    
    # Populate with data
    populate_cv_database(conn)
    
    # Create views
    create_useful_views(conn)
    
    # Print summary
    print_database_summary(conn)
    
    conn.close()
    
    print("\n" + "="*70)
    print(f"✅ Database '{db_name}' created successfully!")
    print("="*70)
    print("\n💡 NEXT STEPS:")
    print("1. Use this database to store real-time YOLO detections")
    print("2. Query LatestDetections view for dashboard")
    print("3. Monitor ActiveCollisionRisks for safety alerts")
    print("4. Track HourlyDetectionSummary for analytics")
    print("\n🚀 Ready for CV integration!")


if __name__ == '__main__':
    main()
