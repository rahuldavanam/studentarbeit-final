"""
Deployment Phase

This module deploys a trained steering prediction model in CARLA simulator
for autonomous vehicle control. The model processes camera images in real-time
and generates steering commands to control the vehicle.
"""

import os
from pathlib import Path
from typing import Optional, Tuple, List, Dict
import carla
import numpy as np
import torch
from torch import nn
from torchvision import transforms
import torchvision.models as models
from PIL import Image
import pandas as pd
import cv2


def region_selection(image: np.ndarray) -> np.ndarray:
    """
    Apply region of interest mask to focus on the road area.
    
    Args:
        image: Input image as numpy array
        
    Returns:
        Masked image with only the region of interest visible
    """
    # Create mask with same dimensions as input image
    mask = np.zeros_like(image)
    
    # Determine mask color based on image channels
    if len(image.shape) > 2:
        channel_count = image.shape[2]
        mask_color = (255,) * channel_count
    else:
        mask_color = 255
    
    # Define trapezoidal region of interest
    rows, cols = image.shape[:2]
    bottom_left = [cols * 0.0, rows * 1.0]
    top_left = [cols * 0.4, rows * 0.49]
    bottom_right = [cols * 1.0, rows * 1.0]
    top_right = [cols * 0.6, rows * 0.49]
    
    vertices = np.array(
        [[bottom_left, top_left, top_right, bottom_right]], 
        dtype=np.int32
    )
    
    # Fill polygon and apply mask
    cv2.fillPoly(mask, vertices, mask_color)
    masked_image = cv2.bitwise_and(image, mask)
    
    return masked_image


class CarlaDeployment:
    """Handles model deployment for autonomous driving in CARLA."""
    
    # Configuration constants
    CARLA_HOST = 'localhost'
    CARLA_PORT = 2000
    CARLA_TIMEOUT = 5.0
    
    # Camera settings
    CAMERA_WIDTH = 640
    CAMERA_HEIGHT = 480
    CAMERA_FOV = 90
    CAMERA_OFFSET_X = 1.2
    CAMERA_OFFSET_Z = 1.5
    
    # Spectator settings
    SPECTATOR_OFFSET_X = -6
    SPECTATOR_OFFSET_Z = 2
    
    # Model settings
    IMAGE_SIZE = (224, 224)
    MODEL_DIR = "carla_models"
    DEFAULT_MODEL_NAME = "model_checkpoint_v6.pth"
    
    # Vehicle settings
    VEHICLE_MODEL = 'vehicle.lincoln.mkz_2020'
    DEFAULT_SPAWN_POINT_INDEX = 1
    SPAWN_OFFSET_X = 0
    
    # Control settings
    DEFAULT_THROTTLE = 0.4
    
    # Synchronous mode settings
    FIXED_DELTA_SECONDS = 0.1
    
    # Weather settings
    WEATHER_CLOUDINESS = 100
    WEATHER_FOG_DENSITY = 0
    WEATHER_PRECIPITATION = 40
    
    # Logging settings
    LOG_INTERVAL = 1.0  # Log every 1 simulated second
    DEFAULT_LOG_FILENAME = "steering_trace.csv"
    
    def __init__(self, model_name: str = DEFAULT_MODEL_NAME):
        """
        Initialize the deployment system.
        
        Args:
            model_name: Name of the trained model file to load
        """
        self.model_name = model_name
        
        # Initialize device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Initialize components (to be set up later)
        self.client: Optional[carla.Client] = None
        self.world: Optional[carla.World] = None
        self.vehicle: Optional[carla.Vehicle] = None
        self.camera: Optional[carla.Sensor] = None
        self.model: Optional[nn.Module] = None
        self.preprocess: Optional[transforms.Compose] = None
        self.original_settings: Optional[carla.WorldSettings] = None
        
        # Logging state
        self.steer_log: List[Dict] = []
        self.last_log_time: Optional[float] = None
        self.starting_time: Optional[float] = None
    
    def setup_transforms(self) -> None:
        """Initialize image preprocessing transforms."""
        self.preprocess = transforms.Compose([
            transforms.Resize(self.IMAGE_SIZE),
            transforms.ToTensor()
        ])
    
    def load_model(self) -> None:
        """Load the trained steering prediction model."""
        # Create model architecture
        self.model = torchvision.models.resnet18(weights=None)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, 1)
        
        # Load trained weights
        model_path = Path(self.MODEL_DIR) / self.model_name
        print(f"Loading model from: {model_path}")
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        
        # Set to evaluation mode
        self.model.to(self.device)
        self.model.eval()
        print("Model loaded successfully")
    
    def connect_to_carla(self) -> None:
        """Establish connection to CARLA server."""
        self.client = carla.Client(self.CARLA_HOST, self.CARLA_PORT)
        self.client.set_timeout(self.CARLA_TIMEOUT)
        self.world = self.client.get_world()
        print(f"Connected to CARLA server at {self.CARLA_HOST}:{self.CARLA_PORT}")
    
    def setup_synchronous_mode(self) -> None:
        """Enable synchronous mode for deterministic simulation."""
        settings = self.world.get_settings()
        self.original_settings = settings
        
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = self.FIXED_DELTA_SECONDS
        self.world.apply_settings(settings)
        print("Synchronous mode enabled")
    
    def setup_weather(
        self,
        cloudiness: Optional[float] = None,
        fog_density: Optional[float] = None,
        precipitation: Optional[float] = None
    ) -> None:
        """
        Configure weather conditions.
        
        Args:
            cloudiness: Cloud coverage (0-100)
            fog_density: Fog density (0-100)
            precipitation: Rain intensity (0-100)
        """
        weather = self.world.get_weather()
        weather.cloudiness = cloudiness if cloudiness is not None else self.WEATHER_CLOUDINESS
        weather.fog_density = fog_density if fog_density is not None else self.WEATHER_FOG_DENSITY
        weather.precipitation = precipitation if precipitation is not None else self.WEATHER_PRECIPITATION
        self.world.set_weather(weather)
        print(f"Weather set: cloudiness={weather.cloudiness}, "
              f"fog={weather.fog_density}, precipitation={weather.precipitation}")
    
    def spawn_vehicle(self, spawn_point_index: Optional[int] = None) -> None:
        """
        Spawn the vehicle in the simulation.
        
        Args:
            spawn_point_index: Index of spawn point to use
        """
        blueprint_library = self.world.get_blueprint_library()
        vehicle_bp = blueprint_library.filter(self.VEHICLE_MODEL)[0]
        
        spawn_points = self.world.get_map().get_spawn_points()
        spawn_idx = spawn_point_index if spawn_point_index is not None else self.DEFAULT_SPAWN_POINT_INDEX
        spawn_point = spawn_points[spawn_idx]
        
        # Apply offset
        shifted_spawn_point = carla.Transform(
            spawn_point.transform(carla.Location(x=self.SPAWN_OFFSET_X)),
            spawn_point.rotation
        )
        
        self.vehicle = self.world.spawn_actor(vehicle_bp, shifted_spawn_point)
        print(f"Vehicle spawned at spawn point {spawn_idx}")
    
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
    
    def predict_control(self, image: np.ndarray) -> Tuple[float, float]:
        """
        Predict steering angle from camera image.
        
        Args:
            image: RGB image from camera
            
        Returns:
            Tuple of (throttle, steering_angle)
        """
        # Apply region of interest
        cropped_image = region_selection(image)
        
        # Convert to PIL Image
        pil_image = Image.fromarray(cropped_image)
        
        # Preprocess and predict
        input_tensor = self.preprocess(pil_image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            output = self.model(input_tensor)
            steer = output.cpu().numpy()[0, 0]
        
        throttle = self.DEFAULT_THROTTLE
        
        return throttle, steer
    
    def process_image(self, carla_image: carla.Image) -> None:
        """
        Process camera frame and control vehicle.
        
        Args:
            carla_image: Image from CARLA camera sensor
        """
        # Convert CARLA image to numpy array
        carla_image.convert(carla.ColorConverter.Raw)
        array_raw = np.frombuffer(carla_image.raw_data, dtype=np.uint8)
        array_bgra = np.reshape(array_raw, (carla_image.height, carla_image.width, 4))
        array_bgr = array_bgra[:, :, :3]
        array_rgb = array_bgr[:, :, ::-1]  # Convert BGR to RGB
        
        # Predict control
        throttle, steer = self.predict_control(array_rgb)
        
        # Apply control to vehicle
        control = carla.VehicleControl(throttle=float(throttle), steer=float(steer))
        self.vehicle.apply_control(control)
        
        # Initialize starting time
        if self.starting_time is None:
            self.starting_time = carla_image.timestamp
        
        # Log data at specified intervals
        current_time = carla_image.timestamp
        if self.last_log_time is None or (current_time - self.last_log_time) >= self.LOG_INTERVAL:
            location = self.vehicle.get_location()
            self.steer_log.append({
                "timestamp": current_time - self.starting_time,
                "steer": float(steer),
                "throttle": throttle,
                "x": location.x,
                "y": location.y,
                "z": location.z
            })
            self.last_log_time = current_time
    
    def update_spectator(self) -> None:
        """Move spectator camera to follow the vehicle."""
        spectator = self.world.get_spectator()
        transform = carla.Transform(
            self.vehicle.get_transform().transform(
                carla.Location(x=self.SPECTATOR_OFFSET_X, z=self.SPECTATOR_OFFSET_Z)
            ),
            self.vehicle.get_transform().rotation
        )
        spectator.set_transform(transform)
    
    def save_log(self, filename: str = DEFAULT_LOG_FILENAME) -> None:
        """
        Save steering log to CSV file.
        
        Args:
            filename: Output CSV filename
        """
        if self.steer_log:
            df = pd.DataFrame(self.steer_log)
            df.to_csv(filename, index=False)
            print(f"Steering log saved to: {filename}")
        else:
            print("No log data to save")
    
    def cleanup(self) -> None:
        """Clean up spawned actors and restore settings."""
        print("\nCleaning up...")
        
        # Stop camera
        if self.camera:
            self.camera.stop()
        
        # Destroy actors
        if self.world:
            for actor in self.world.get_actors().filter('*vehicle*'):
                actor.destroy()
            for sensor in self.world.get_actors().filter('*sensor*'):
                sensor.destroy()
        
        # Restore original settings
        if self.world and self.original_settings:
            self.world.apply_settings(self.original_settings)
            self.world.set_weather(carla.WeatherParameters.Default)
        
        print("Cleanup complete")
    
    def run(
        self,
        duration: Optional[float] = None,
        log_filename: str = DEFAULT_LOG_FILENAME,
        spawn_point_index: Optional[int] = None
    ) -> None:
        """
        Execute the autonomous driving deployment.
        
        Args:
            duration: Optional duration in seconds (None for infinite)
            log_filename: Filename for steering log CSV
            spawn_point_index: Spawn point index (None for default)
        """
        try:
            # Setup all components
            print("="*60)
            print("Initializing Autonomous Driving System")
            print("="*60)
            
            self.setup_transforms()
            self.load_model()
            self.connect_to_carla()
            self.setup_synchronous_mode()
            self.setup_weather()
            self.spawn_vehicle(spawn_point_index=spawn_point_index)
            self.attach_camera()
            
            # Clear logging state
            self.steer_log.clear()
            self.last_log_time = None
            self.starting_time = None
            
            # Start camera callback
            self.camera.listen(lambda image: self.process_image(image))
            
            print("\n" + "="*60)
            print("Autonomous driving started. Press Ctrl+C to stop.")
            print("="*60 + "\n")
            
            # Main control loop
            start_time = None
            while True:
                # Track elapsed time if duration is specified
                if duration is not None:
                    if start_time is None:
                        start_time = self.world.get_snapshot().timestamp.elapsed_seconds
                    
                    elapsed = self.world.get_snapshot().timestamp.elapsed_seconds - start_time
                    if elapsed >= duration:
                        print(f"\nDuration of {duration}s reached. Stopping...")
                        break
                
                # Update spectator and tick world
                self.update_spectator()
                self.world.tick()
        
        except KeyboardInterrupt:
            print("\n\nStopped by user (Ctrl+C)")
        
        except Exception as e:
            print(f"\nError during deployment: {e}")
            import traceback
            traceback.print_exc()
        
        finally:
            self.cleanup()
            self.save_log(filename=log_filename)
            
            # Print summary
            if self.steer_log:
                print(f"\nDeployment Summary:")
                print(f"  Total logged samples: {len(self.steer_log)}")
                print(f"  Duration: {self.steer_log[-1]['timestamp']:.1f}s")
                steers = [log['steer'] for log in self.steer_log]
                print(f"  Steering range: [{min(steers):.3f}, {max(steers):.3f}]")


def main():
    """Entry point for the deployment phase."""
    deployment = CarlaDeployment(model_name="model_checkpoint_v6.pth")
    deployment.run(
        duration=None,  # Run indefinitely until Ctrl+C
        log_filename="steering_trace.csv",
        spawn_point_index=1
    )


if __name__ == '__main__':
    main()
