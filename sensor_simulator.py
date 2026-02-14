"""
Multi-Sensor Simulation System
Generates realistic data from Load Cells, Ultrasonic, and Proximity sensors
Implements sensor fusion logic for robust warehouse monitoring
"""

import sqlite3
import random
import numpy as np
from datetime import datetime, timedelta
import time

class SensorSimulator:
    """
    Simulates multiple sensor types for warehouse monitoring:
    - Load Cells (weight-based counting)
    - Ultrasonic Sensors (level detection)
    - Proximity Sensors (forklift safety)
    """
    
    def __init__(self, db_path='cv_detections.db'):
        self.db_path = db_path
        self.initialize_sensor_tables()
        
        # Sensor specifications
        self.sensor_specs = {
            'load_cell': {
                'unit_weight_kg': 2.5,  # Average weight per box in kg
                'accuracy': 0.98,        # 98% accurate
                'noise_range': 0.02      # ±2% noise
            },
            'ultrasonic': {
                'max_distance_cm': 300,  # 3 meters max range
                'min_distance_cm': 10,
                'accuracy': 0.95,
                'noise_range': 0.03      # ±3% noise
            },
            'proximity': {
                'detection_range_m': 5,  # Detects up to 5 meters
                'accuracy': 0.99,
                'false_positive_rate': 0.01
            }
        }
    
    def initialize_sensor_tables(self):
        """Create database tables for sensor data"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Table 1: Load Cell Readings (Weight-based counting)
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS LoadCellReadings (
            reading_id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            zone_id TEXT NOT NULL,
            sensor_id TEXT NOT NULL,
            
            total_weight_kg REAL NOT NULL,
            estimated_unit_count INTEGER,
            weight_per_unit_kg REAL,
            
            confidence REAL,
            calibration_status TEXT,
            
            FOREIGN KEY (zone_id) REFERENCES ZoneConfig(zone_id)
        );
        ''')
        
        # Table 2: Ultrasonic Level Readings
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS UltrasonicReadings (
            reading_id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            zone_id TEXT NOT NULL,
            sensor_id TEXT NOT NULL,
            
            distance_cm REAL NOT NULL,
            fill_level_percent REAL,
            stack_height_cm REAL,
            
            max_capacity INTEGER,
            current_capacity_estimate INTEGER,
            
            signal_quality REAL,
            
            FOREIGN KEY (zone_id) REFERENCES ZoneConfig(zone_id)
        );
        ''')
        
        # Table 3: Proximity Sensor Readings (Forklift detection)
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS ProximityReadings (
            reading_id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            zone_id TEXT NOT NULL,
            sensor_id TEXT NOT NULL,
            
            object_detected BOOLEAN NOT NULL,
            distance_m REAL,
            object_type TEXT,
            
            velocity_estimate REAL,
            confidence REAL,
            
            FOREIGN KEY (zone_id) REFERENCES ZoneConfig(zone_id)
        );
        ''')
        
        # Table 4: Sensor Fusion Results
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS SensorFusionData (
            fusion_id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            zone_id TEXT NOT NULL,
            
            -- Individual sensor estimates
            camera_count INTEGER,
            load_cell_count INTEGER,
            ultrasonic_level_percent REAL,
            
            -- Fusion results
            fused_count INTEGER,
            confidence_score REAL,
            anomaly_detected BOOLEAN,
            anomaly_reason TEXT,
            
            -- Agreement metrics
            sensor_agreement_percent REAL,
            fusion_method TEXT,
            
            FOREIGN KEY (zone_id) REFERENCES ZoneConfig(zone_id)
        );
        ''')
        
        # Table 5: Environmental Conditions (affects sensor accuracy)
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS EnvironmentalConditions (
            condition_id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            zone_id TEXT NOT NULL,
            
            lighting_level_lux REAL,
            temperature_celsius REAL,
            humidity_percent REAL,
            
            dust_level TEXT,
            vibration_level TEXT,
            
            camera_visibility_score REAL,
            
            FOREIGN KEY (zone_id) REFERENCES ZoneConfig(zone_id)
        );
        ''')
        
        conn.commit()
        conn.close()
        print("✅ Sensor tables created successfully!")
    
    def simulate_load_cell(self, zone_id, actual_box_count):
        """
        Simulate load cell weight sensor
        
        Args:
            zone_id: Warehouse zone
            actual_box_count: Ground truth from CV
        
        Returns:
            dict: Load cell reading
        """
        specs = self.sensor_specs['load_cell']
        
        # Calculate expected weight
        expected_weight = actual_box_count * specs['unit_weight_kg']
        
        # Add realistic noise
        noise = np.random.normal(0, specs['noise_range'])
        measured_weight = expected_weight * (1 + noise)
        measured_weight = max(0, measured_weight)  # Can't be negative
        
        # Estimate count from weight
        if measured_weight > 0:
            estimated_count = int(round(measured_weight / specs['unit_weight_kg']))
        else:
            estimated_count = 0
        
        # Confidence based on calibration
        confidence = specs['accuracy'] - abs(noise) * 0.1
        confidence = max(0.5, min(1.0, confidence))
        
        # Calibration status
        if abs(estimated_count - actual_box_count) <= 2:
            calibration = 'GOOD'
        elif abs(estimated_count - actual_box_count) <= 5:
            calibration = 'FAIR'
        else:
            calibration = 'NEEDS_CALIBRATION'
        
        return {
            'zone_id': zone_id,
            'sensor_id': f'LC-{zone_id}',
            'total_weight_kg': round(measured_weight, 2),
            'estimated_unit_count': estimated_count,
            'weight_per_unit_kg': specs['unit_weight_kg'],
            'confidence': round(confidence, 3),
            'calibration_status': calibration
        }
    
    def simulate_ultrasonic(self, zone_id, actual_box_count, max_capacity=1000):
        """
        Simulate ultrasonic level sensor
        
        Args:
            zone_id: Warehouse zone
            actual_box_count: Ground truth
            max_capacity: Maximum zone capacity
        
        Returns:
            dict: Ultrasonic reading
        """
        specs = self.sensor_specs['ultrasonic']
        
        # Calculate fill level (0-100%)
        fill_level = (actual_box_count / max_capacity) * 100
        fill_level = min(100, max(0, fill_level))
        
        # Stack height estimation (boxes stack vertically)
        # Assume: 20cm per box layer, 5 boxes wide
        layers = actual_box_count / 5
        stack_height_cm = layers * 20
        
        # Distance from sensor to top of stack
        # Sensor is at ceiling (300cm from ground)
        distance_from_sensor = specs['max_distance_cm'] - stack_height_cm
        
        # Add noise
        noise = np.random.normal(0, specs['noise_range'])
        measured_distance = distance_from_sensor * (1 + noise)
        measured_distance = max(specs['min_distance_cm'], 
                               min(specs['max_distance_cm'], measured_distance))
        
        # Recalculate estimates from measured distance
        measured_stack_height = specs['max_distance_cm'] - measured_distance
        measured_layers = measured_stack_height / 20
        estimated_capacity = int(measured_layers * 5)
        
        # Signal quality (degrades with distance and dust)
        signal_quality = specs['accuracy'] - (measured_distance / specs['max_distance_cm']) * 0.1
        signal_quality = max(0.6, min(1.0, signal_quality))
        
        return {
            'zone_id': zone_id,
            'sensor_id': f'US-{zone_id}',
            'distance_cm': round(measured_distance, 1),
            'fill_level_percent': round(fill_level, 1),
            'stack_height_cm': round(stack_height_cm, 1),
            'max_capacity': max_capacity,
            'current_capacity_estimate': estimated_capacity,
            'signal_quality': round(signal_quality, 3)
        }
    
    def simulate_proximity(self, zone_id, person_detected, forklift_detected):
        """
        Simulate proximity sensor for forklift detection
        
        Args:
            zone_id: Warehouse zone
            person_detected: Is person present (from CV)
            forklift_detected: Is forklift present (from CV)
        
        Returns:
            dict: Proximity sensor reading
        """
        specs = self.sensor_specs['proximity']
        
        # Determine if object detected (with false positive/negative rate)
        actual_object = person_detected or forklift_detected
        
        # Apply accuracy
        if actual_object:
            # True positive or false negative
            detected = random.random() < specs['accuracy']
        else:
            # True negative or false positive
            detected = random.random() < specs['false_positive_rate']
        
        # If detected, estimate distance and type
        if detected:
            if forklift_detected:
                object_type = 'FORKLIFT'
                distance = random.uniform(1.0, 4.0)  # Forklifts are larger
                velocity = random.uniform(0.5, 2.0)  # Moving slowly in warehouse
            elif person_detected:
                object_type = 'PERSON'
                distance = random.uniform(0.5, 3.0)
                velocity = random.uniform(0.3, 1.5)  # Walking speed
            else:
                object_type = 'UNKNOWN'
                distance = random.uniform(2.0, 5.0)
                velocity = 0.0
            
            confidence = specs['accuracy']
        else:
            object_type = None
            distance = None
            velocity = None
            confidence = 1.0 - specs['false_positive_rate']
        
        return {
            'zone_id': zone_id,
            'sensor_id': f'PROX-{zone_id}',
            'object_detected': detected,
            'distance_m': round(distance, 2) if distance else None,
            'object_type': object_type,
            'velocity_estimate': round(velocity, 2) if velocity else None,
            'confidence': round(confidence, 3)
        }
    
    def simulate_environmental_conditions(self, zone_id):
        """
        Simulate environmental conditions that affect sensor accuracy
        
        Returns:
            dict: Environmental data
        """
        # Time of day affects lighting
        hour = datetime.now().hour
        
        # Lighting (lux): Poor at night, good during day
        if 6 <= hour <= 18:
            lighting = random.uniform(300, 800)  # Good lighting
            camera_visibility = random.uniform(0.85, 0.98)
        else:
            lighting = random.uniform(50, 200)   # Poor lighting
            camera_visibility = random.uniform(0.60, 0.80)
        
        # Temperature (affects some sensors)
        temperature = random.uniform(18, 28)
        
        # Humidity
        humidity = random.uniform(30, 70)
        
        # Dust level (categorical)
        dust_level = random.choice(['LOW', 'LOW', 'LOW', 'MEDIUM', 'HIGH'])
        
        # Vibration from forklifts
        vibration_level = random.choice(['LOW', 'LOW', 'MEDIUM'])
        
        # Dust and poor lighting reduce camera visibility
        if dust_level == 'HIGH':
            camera_visibility *= 0.8
        elif dust_level == 'MEDIUM':
            camera_visibility *= 0.9
        
        return {
            'zone_id': zone_id,
            'lighting_level_lux': round(lighting, 1),
            'temperature_celsius': round(temperature, 1),
            'humidity_percent': round(humidity, 1),
            'dust_level': dust_level,
            'vibration_level': vibration_level,
            'camera_visibility_score': round(camera_visibility, 3)
        }
    
    def sensor_fusion(self, zone_id, camera_count, load_cell_count, 
                     ultrasonic_count, environmental_score):
        """
        Fuse multiple sensor readings to get robust estimate
        
        Implements weighted fusion based on environmental conditions
        """
        # Weight sensors based on reliability
        # In poor conditions, weight and ultrasonic are more reliable
        camera_weight = environmental_score  # Camera affected by environment
        load_cell_weight = 0.98              # Very reliable
        ultrasonic_weight = 0.95             # Quite reliable
        
        # Normalize weights
        total_weight = camera_weight + load_cell_weight + ultrasonic_weight
        camera_weight /= total_weight
        load_cell_weight /= total_weight
        ultrasonic_weight /= total_weight
        
        # Weighted average
        fused_count = (
            camera_count * camera_weight +
            load_cell_count * load_cell_weight +
            ultrasonic_count * ultrasonic_weight
        )
        fused_count = int(round(fused_count))
        
        # Calculate agreement between sensors
        sensor_values = [camera_count, load_cell_count, ultrasonic_count]
        mean_value = np.mean(sensor_values)
        
        if mean_value > 0:
            agreement = 100 * (1 - np.std(sensor_values) / mean_value)
            agreement = max(0, min(100, agreement))
        else:
            agreement = 100
        
        # Detect anomalies
        anomaly_detected = False
        anomaly_reason = None
        
        # Check if sensors disagree significantly (>5%)
        if agreement < 95:
            max_diff = max(sensor_values) - min(sensor_values)
            if max_diff > mean_value * 0.05:
                anomaly_detected = True
                anomaly_reason = f"Sensor disagreement: {max_diff} units ({100*max_diff/mean_value:.1f}%)"
        
        # Confidence score based on agreement
        confidence = agreement / 100
        
        return {
            'zone_id': zone_id,
            'camera_count': camera_count,
            'load_cell_count': load_cell_count,
            'ultrasonic_level_percent': ultrasonic_count,
            'fused_count': fused_count,
            'confidence_score': round(confidence, 3),
            'anomaly_detected': anomaly_detected,
            'anomaly_reason': anomaly_reason,
            'sensor_agreement_percent': round(agreement, 1),
            'fusion_method': 'WEIGHTED_AVERAGE'
        }
    
    def save_sensor_readings(self, load_cell_data, ultrasonic_data, 
                           proximity_data, environmental_data, fusion_data):
        """Save all sensor readings to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # Save load cell
            cursor.execute('''
                INSERT INTO LoadCellReadings 
                (zone_id, sensor_id, total_weight_kg, estimated_unit_count, 
                 weight_per_unit_kg, confidence, calibration_status)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                load_cell_data['zone_id'],
                load_cell_data['sensor_id'],
                load_cell_data['total_weight_kg'],
                load_cell_data['estimated_unit_count'],
                load_cell_data['weight_per_unit_kg'],
                load_cell_data['confidence'],
                load_cell_data['calibration_status']
            ))
            
            # Save ultrasonic
            cursor.execute('''
                INSERT INTO UltrasonicReadings 
                (zone_id, sensor_id, distance_cm, fill_level_percent, 
                 stack_height_cm, max_capacity, current_capacity_estimate, signal_quality)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                ultrasonic_data['zone_id'],
                ultrasonic_data['sensor_id'],
                ultrasonic_data['distance_cm'],
                ultrasonic_data['fill_level_percent'],
                ultrasonic_data['stack_height_cm'],
                ultrasonic_data['max_capacity'],
                ultrasonic_data['current_capacity_estimate'],
                ultrasonic_data['signal_quality']
            ))
            
            # Save proximity
            cursor.execute('''
                INSERT INTO ProximityReadings 
                (zone_id, sensor_id, object_detected, distance_m, 
                 object_type, velocity_estimate, confidence)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                proximity_data['zone_id'],
                proximity_data['sensor_id'],
                proximity_data['object_detected'],
                proximity_data['distance_m'],
                proximity_data['object_type'],
                proximity_data['velocity_estimate'],
                proximity_data['confidence']
            ))
            
            # Save environmental
            cursor.execute('''
                INSERT INTO EnvironmentalConditions 
                (zone_id, lighting_level_lux, temperature_celsius, 
                 humidity_percent, dust_level, vibration_level, camera_visibility_score)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                environmental_data['zone_id'],
                environmental_data['lighting_level_lux'],
                environmental_data['temperature_celsius'],
                environmental_data['humidity_percent'],
                environmental_data['dust_level'],
                environmental_data['vibration_level'],
                environmental_data['camera_visibility_score']
            ))
            
            # Save fusion
            cursor.execute('''
                INSERT INTO SensorFusionData 
                (zone_id, camera_count, load_cell_count, ultrasonic_level_percent,
                 fused_count, confidence_score, anomaly_detected, anomaly_reason,
                 sensor_agreement_percent, fusion_method)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                fusion_data['zone_id'],
                fusion_data['camera_count'],
                fusion_data['load_cell_count'],
                fusion_data['ultrasonic_level_percent'],
                fusion_data['fused_count'],
                fusion_data['confidence_score'],
                fusion_data['anomaly_detected'],
                fusion_data['anomaly_reason'],
                fusion_data['sensor_agreement_percent'],
                fusion_data['fusion_method']
            ))
            
            conn.commit()
            return True
            
        except Exception as e:
            print(f"❌ Error saving sensor data: {e}")
            conn.rollback()
            return False
        finally:
            conn.close()
    
    def simulate_all_sensors(self, zone_id, camera_count, 
                            person_detected, forklift_detected):
        """
        Simulate all sensors for a zone and perform fusion
        
        Args:
            zone_id: Zone identifier
            camera_count: Box count from CV
            person_detected: Person present (from CV)
            forklift_detected: Forklift present (from CV)
        """
        # Simulate environmental conditions
        env_data = self.simulate_environmental_conditions(zone_id)
        
        # Simulate load cell
        load_cell = self.simulate_load_cell(zone_id, camera_count)
        
        # Simulate ultrasonic
        ultrasonic = self.simulate_ultrasonic(zone_id, camera_count)
        
        # Simulate proximity
        proximity = self.simulate_proximity(zone_id, person_detected, forklift_detected)
        
        # Perform sensor fusion
        fusion = self.sensor_fusion(
            zone_id,
            camera_count,
            load_cell['estimated_unit_count'],
            ultrasonic['current_capacity_estimate'],
            env_data['camera_visibility_score']
        )
        
        # Save all data
        self.save_sensor_readings(load_cell, ultrasonic, proximity, env_data, fusion)
        
        return {
            'load_cell': load_cell,
            'ultrasonic': ultrasonic,
            'proximity': proximity,
            'environmental': env_data,
            'fusion': fusion
        }


# ============================================================================
# MAIN - Generate Sensor Data
# ============================================================================

def main():
    """Generate realistic sensor data for all zones"""
    print("="*70)
    print("🎛️  MULTI-SENSOR SIMULATION SYSTEM")
    print("="*70)
    
    simulator = SensorSimulator()
    
    # Get latest CV detections from database
    conn = sqlite3.connect('cv_detections.db')
    cursor = conn.cursor()
    
    cursor.execute("SELECT * FROM LatestDetections")
    zones = cursor.fetchall()
    conn.close()
    
    if not zones:
        print("⚠️ No CV detection data found. Run warehouse_cv_model.py first!")
        return
    
    print(f"\n📊 Generating sensor data for {len(zones)} zones...")
    print("="*70)
    
    for zone in zones:
        zone_id = zone[0]
        box_count = zone[4]
        person_detected = zone[7]
        forklift_detected = zone[8]
        
        print(f"\n🎯 Processing Zone: {zone_id}")
        
        # Simulate all sensors
        result = simulator.simulate_all_sensors(
            zone_id, 
            box_count,
            person_detected,
            forklift_detected
        )
        
        # Display results
        print(f"\n📷 Camera Count: {box_count}")
        print(f"⚖️  Load Cell: {result['load_cell']['estimated_unit_count']} units "
              f"({result['load_cell']['total_weight_kg']}kg)")
        print(f"📡 Ultrasonic: {result['ultrasonic']['current_capacity_estimate']} units "
              f"({result['ultrasonic']['fill_level_percent']:.1f}% full)")
        print(f"🔮 Fused Count: {result['fusion']['fused_count']} "
              f"(confidence: {result['fusion']['confidence_score']:.1%})")
        print(f"📊 Sensor Agreement: {result['fusion']['sensor_agreement_percent']:.1f}%")
        
        if result['fusion']['anomaly_detected']:
            print(f"⚠️  ANOMALY: {result['fusion']['anomaly_reason']}")
        
        if result['proximity']['object_detected']:
            print(f"🚨 Proximity Alert: {result['proximity']['object_type']} detected "
                  f"at {result['proximity']['distance_m']}m")
        
        print(f"🌡️  Environment: {result['environmental']['lighting_level_lux']:.0f} lux, "
              f"Camera visibility: {result['environmental']['camera_visibility_score']:.1%}")
    
    print("\n" + "="*70)
    print("✅ Sensor simulation complete!")
    print("="*70)
    print("\n💡 Data saved to:")
    print("  • LoadCellReadings")
    print("  • UltrasonicReadings")
    print("  • ProximityReadings")
    print("  • EnvironmentalConditions")
    print("  • SensorFusionData")
    print("\n🚀 Ready for dashboard integration!")


if __name__ == '__main__':
    main()
