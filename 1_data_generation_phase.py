"""
CARLA Data Generation Phase

This module collects training data from CARLA simulator by recording vehicle sensor data,
control inputs, and camera images during autonomous driving.
"""

import carla
import os
import csv
import math
import time
from typing import List, Optional


class CarlaDataCollector:
    """Handles data collection from CARLA simulation."""
    
    # Configuration constants
    DISK_PATH = 'G:/carla_data_T4_cropped/'
    OUTPUT_FOLDER = 'CCW151'
    SAVE_INTERVAL = 0.5  # seconds between saves
    CAMERA_FOV = 90
    CAMERA_WIDTH = 640
    CAMERA_HEIGHT = 480
    VEHICLE_MODEL = 'vehicle.lincoln.mkz_2020'
    SPAWN_POINT_INDEX = 151
    SPAWN_OFFSET_X = 50
    SPAWN_OFFSET_Y = 0
    CAMERA_OFFSET_X = 1.2
    CAMERA_OFFSET_Z = 1.5
    SPECTATOR_OFFSET_X = -6
    SPECTATOR_OFFSET_Z = 2
    UPDATE_INTERVAL = 0.1  # seconds
    
    # Weather settings
    WEATHER_FOG_DENSITY = 0
    WEATHER_CLOUDINESS = 100
    WEATHER_PRECIPITATION = 0
    WEATHER_PRECIPITATION_DEPOSIT = 0
    
    # CARLA connection settings
    CARLA_HOST = 'localhost'
    CARLA_PORT = 2000
    CARLA_TIMEOUT = 5.0
    
    # CSV headers
    CSV_HEADERS = [
        'Frame id', 'Time(s)', 
        'Velocity (x)', 'Velocity(y)', 'Velocity(z)', 'Velocity',
        'Throttle', 'Steer', 'Brake', 'Handbrake', 'Reverse', 
        'Manual Gear Shift', 'Gear', 'Image Path'
    ]
    
    def __init__(self):
        """Initialize the data collector."""
        self.frame_id = 0
        self.starting_time = 0
        self.last_save_time = time.time()
        self.csv_filename = f'{self.OUTPUT_FOLDER}_data.csv'
        
        # CARLA objects (initialized later)
        self.client: Optional[carla.Client] = None
        self.world: Optional[carla.World] = None
        self.vehicle: Optional[carla.Vehicle] = None
        self.camera: Optional[carla.Sensor] = None
        
        # Create output directory
        self.output_path = os.path.join(self.DISK_PATH, self.OUTPUT_FOLDER)
        os.makedirs(self.output_path, exist_ok=True)
    
    def connect_to_carla(self) -> None:
        """Establish connection to CARLA server."""
        self.client = carla.Client(self.CARLA_HOST, self.CARLA_PORT)
        self.client.set_timeout(self.CARLA_TIMEOUT)
        self.world = self.client.get_world()
        print(f"Connected to CARLA server at {self.CARLA_HOST}:{self.CARLA_PORT}")
    
    def setup_weather(self) -> None:
        """Configure weather conditions."""
        weather = self.world.get_weather()
        weather.fog_density = self.WEATHER_FOG_DENSITY
        weather.cloudiness = self.WEATHER_CLOUDINESS
        self.world.set_weather(weather)
    
    def spawn_vehicle(self) -> None:
        """Spawn the vehicle at the designated spawn point."""
        blueprint_library = self.world.get_blueprint_library()
        vehicle_bp = blueprint_library.filter(self.VEHICLE_MODEL)[0]
        
        world_map = self.world.get_map()
        spawn_points = world_map.get_spawn_points()
        spawn_point = spawn_points[self.SPAWN_POINT_INDEX]
        
        # Apply offset to spawn point
        shifted_spawn_point = carla.Transform(
            spawn_point.transform(carla.Location(x=self.SPAWN_OFFSET_X, y=self.SPAWN_OFFSET_Y)),
            spawn_point.rotation
        )
        
        self.vehicle = self.world.spawn_actor(vehicle_bp, shifted_spawn_point)
        print(f"Vehicle spawned at spawn point {self.SPAWN_POINT_INDEX}")
    
    def attach_camera(self) -> None:
        """Attach RGB camera to the vehicle."""
        blueprint_library = self.world.get_blueprint_library()
        camera_bp = blueprint_library.find('sensor.camera.rgb')
        
        # Configure camera
        camera_bp.set_attribute("image_size_x", str(self.CAMERA_WIDTH))
        camera_bp.set_attribute("image_size_y", str(self.CAMERA_HEIGHT))
        camera_bp.set_attribute("fov", str(self.CAMERA_FOV))
        
        camera_transform = carla.Transform(
            carla.Location(x=self.CAMERA_OFFSET_X, z=self.CAMERA_OFFSET_Z)
        )
        
        self.camera = self.world.spawn_actor(camera_bp, camera_transform, attach_to=self.vehicle)
        print("Camera attached to vehicle")
    
    def initialize_csv(self) -> None:
        """Initialize CSV file with headers."""
        with open(self.csv_filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(self.CSV_HEADERS)
    
    def save_data_to_csv(self, data: List) -> None:
        """Append data row to CSV file."""
        with open(self.csv_filename, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(data)
    
    def calculate_velocity_magnitude(self, velocity: carla.Vector3D) -> float:
        """Calculate velocity magnitude from vector components."""
        return math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
    
    def on_image_captured(self, image: carla.Image) -> None:
        """Callback function executed when camera captures an image."""
        current_time = time.time()
        
        # Apply save interval throttling
        if current_time - self.last_save_time < self.SAVE_INTERVAL:
            return
        
        # Get vehicle velocity
        velocity = self.vehicle.get_velocity()
        velocity_magnitude = self.calculate_velocity_magnitude(velocity)
        
        # Get vehicle control data
        control = self.vehicle.get_control()
        
        # Save image to disk
        save_image_path = os.path.join(self.output_path, f'{self.frame_id}.png')
        image.save_to_disk(save_image_path)
        
        # Prepare data row
        data = [
            self.frame_id,
            current_time - self.starting_time,
            velocity.x,
            velocity.y,
            velocity.z,
            velocity_magnitude,
            control.throttle,
            control.steer,
            control.brake,
            control.hand_brake,
            control.reverse,
            control.manual_gear_shift,
            control.gear,
            save_image_path
        ]
        
        self.save_data_to_csv(data)
        self.frame_id += 1
        self.last_save_time = current_time
    
    def update_spectator_position(self) -> None:
        """Move spectator camera to follow the vehicle."""
        spectator = self.world.get_spectator()
        transform = carla.Transform(
            self.vehicle.get_transform().transform(
                carla.Location(x=self.SPECTATOR_OFFSET_X, z=self.SPECTATOR_OFFSET_Z)
            ),
            self.vehicle.get_transform().rotation
        )
        spectator.set_transform(transform)
    
    def cleanup(self) -> None:
        """Clean up spawned actors."""
        print("Cleaning up...")
        if self.camera:
            self.camera.stop()
        
        for actor in self.world.get_actors().filter('*vehicle*'):
            actor.destroy()
        for sensor in self.world.get_actors().filter('*sensor*'):
            sensor.destroy()
        
        print("Cleanup complete")
    
    def run(self) -> None:
        """Main data collection loop."""
        try:
            # Setup phase
            self.connect_to_carla()
            self.setup_weather()
            self.spawn_vehicle()
            self.attach_camera()
            self.initialize_csv()
            
            # Start recording
            self.starting_time = time.time()
            time.sleep(0.5)  # Initial delay
            
            self.camera.listen(lambda image: self.on_image_captured(image))
            self.vehicle.set_autopilot(True)
            
            print("Data collection started. Press Ctrl+C to stop.")
            
            # Main loop
            while True:
                self.update_spectator_position()
                time.sleep(self.UPDATE_INTERVAL)
        
        except KeyboardInterrupt:
            print("\nData collection stopped by user")
        
        except Exception as e:
            print(f"Error during data collection: {e}")
        
        finally:
            self.cleanup()


def main():
    """Entry point for the data generation phase."""
    collector = CarlaDataCollector()
    collector.run()


if __name__ == '__main__':
    main()