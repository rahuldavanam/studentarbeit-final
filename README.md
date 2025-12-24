# Autonomous Driving System for CARLA Simulator

A complete end-to-end autonomous driving pipeline that collects training data from CARLA simulator, processes images, trains a deep learning model, and deploys it for real-time steering control.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [System Requirements](#system-requirements)
- [Installation](#installation)
- [Pipeline Workflow](#pipeline-workflow)
- [Module Documentation](#module-documentation)
    - [1. Data Generation Phase](#1-data-generation-phase)
    - [2. Image Processing Phase](#2-image-processing-phase)
    - [3. Training Phase](#3-training-phase)
    - [4. Deployment Phase](#4-deployment-phase)
- [Dataset Format](#dataset-format)
- [Configuration](#configuration)
- [Usage Examples](#usage-examples)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [Troubleshooting](#troubleshooting)
- [License](#license)
- [Contributing](#contributing)
- [Contact](#contact)

## Overview

This project implements an autonomous driving system that learns to steer a vehicle by observing human (autopilot) driving behavior in the CARLA simulator. The system consists of four main phases:

1. **Data Generation**: Collect driving data from CARLA autopilot.
2. **Image Processing**: Apply region of interest masking to focus on relevant road areas.
3. **Training**: Train a ResNet-18 model to predict steering angles from camera images.
4. **Deployment**: Deploy the trained model for real-time autonomous driving.

## Features

- Automated data collection from CARLA simulator.
- Region of interest (ROI) image preprocessing.
- Transfer learning with ResNet18 for steering prediction.
- Real-time autonomous vehicle control.
- Comprehensive logging and monitoring.
- Configurable weather and environmental conditions.
- Synchronous mode support for deterministic simulation.

## System Requirements

- **CARLA Simulator**: Version 0.9.14
- **Python**: 3.7+
- **GPU**: NVIDIA GPU with CUDA support (recommended)
- **RAM**: 8GB minimum, 16GB recommended
- **Storage**: 50GB+ for dataset storage

### Python Dependencies

```
torch>=1.9.0
torchvision>=0.10.0
opencv-python>=4.5.0
pandas>=1.3.0
numpy>=1.21.0
Pillow>=8.3.0
carla>=0.9.13
```

## Installation

1. **Install CARLA Simulator**
     ```bash
     # Download from https://github.com/carla-simulator/carla/releases
     # Extract and run CarlaUE4.exe
     ```

2. **Clone this repository**
     ```bash
     git clone https://github.com/yourusername/studentarbeit-final.git
     cd studentarbeit-final
     ```

3. **Install Python dependencies**
     ```bash
     pip install -r requirements.txt
     ```

4. **Install CARLA Python API**
     ```bash
     # Copy the .egg file from CARLA installation
     easy_install path/to/carla/PythonAPI/carla/dist/carla-0.9.13-py3.7-win-amd64.egg
     ```

## Pipeline Workflow

```
┌─────────────────────┐
│  1. Data Generation │
│   (CARLA Autopilot) │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ 2. Image Processing │
│   (ROI Masking)     │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│   3. Model Training │
│   (ResNet18)        │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│   4. Deployment     │
│ (Autonomous Drive)  │
└─────────────────────┘
```

## Module Documentation

### 1. Data Generation Phase

**File**: `1_data_generation_phase.py`

This module connects to CARLA simulator and collects training data by recording the autopilot's driving behavior.

#### Key Components

- **CarlaDataCollector**: Main class handling data collection
    - Spawns vehicle at specified location
    - Attaches RGB camera sensor
    - Records vehicle telemetry and control inputs
    - Saves synchronized camera images and CSV data

#### Configuration Options

```python
DISK_PATH = 'G:/carla_data_T4_cropped/'      # Example output directory
OUTPUT_FOLDER = 'CCW151'                     # Example dataset subfolder
SAVE_INTERVAL = 0.5                          # Seconds between saves
SPAWN_POINT_INDEX = 151                      # Starting location
CAMERA_WIDTH = 640                           # Image width
CAMERA_HEIGHT = 480                          # Image height
```

#### Collected Data

For each frame, the system records:
- **Timestamp**: Simulation time
- **Velocity**: Vehicle speed (x, y, z components and magnitude)
- **Control Inputs**: Throttle, steering, brake, gear
- **Camera Image**: RGB image from vehicle perspective

#### Usage

```python
from data_generation_phase import CarlaDataCollector

collector = CarlaDataCollector()
collector.run()  # Press Ctrl+C to stop
```

#### Output Structure

```
G:/carla_data_T4_cropped/CCW151/
├── 0.png
├── 1.png
├── 2.png
├── ...
└── CCW151_data.csv
```

---

### 2. Image Processing Phase

**File**: `2_image_processing_phase.py`

This module processes raw camera images by applying region of interest (ROI) masking to focus on the relevant road area.

#### Key Components

- **ImageProcessor**: Handles batch image processing
    - Applies trapezoidal ROI mask
    - Filters out irrelevant background
    - Prepares images for training

#### ROI Configuration

The ROI is defined as a trapezoid covering the road ahead:

```python
ROI_BOTTOM_LEFT_X = 0.0    # Left edge at bottom
ROI_BOTTOM_LEFT_Y = 1.0    # Bottom of image
ROI_TOP_LEFT_X = 0.4       # Narrows at horizon
ROI_TOP_LEFT_Y = 0.49      # Horizon level
ROI_BOTTOM_RIGHT_X = 1.0   # Right edge at bottom
ROI_BOTTOM_RIGHT_Y = 1.0
ROI_TOP_RIGHT_X = 0.6      # Narrows at horizon
ROI_TOP_RIGHT_Y = 0.49
```

#### Usage

```python
from image_processing_phase import ImageProcessor

processor = ImageProcessor(folder_name="CW12")
processor.run()
```

#### Processing Steps

1. Load raw image from source directory
2. Create trapezoidal mask
3. Apply mask to isolate road region
4. Save processed image to destination directory

---

### 3. Training Phase

**File**: `3_training_phase.py`

This module trains a deep learning model to predict steering angles from camera images using transfer learning with ResNet18.

#### Key Components

- **CarlaCustomDataset**: PyTorch Dataset for loading CARLA data
    - Reads images and steering labels from CSV
    - Applies image transformations
    - Validates data integrity

- **ModelTrainer**: Handles complete training pipeline
    - Loads pre-trained ResNet18
    - Fine-tunes last layers
    - Implements learning rate scheduling
    - Tracks training/validation metrics

#### Model Architecture

```
ResNet18 (Pre-trained on ImageNet)
├── Conv layers (frozen)
├── Layer 1-3 (frozen)
├── Layer 4 (trainable, LR=1e-5)
└── FC layer (trainable, LR=1e-4)
        └── Output: 1 (steering angle)
```

#### Training Configuration

```python
IMAGE_SIZE = (224, 224)           # Input image size
TRAIN_BATCH_SIZE = 64             # Training batch size
DEFAULT_EPOCHS = 30               # Number of epochs
LAYER4_LR = 1e-5                  # Learning rate for layer4
FC_LR = 1e-4                      # Learning rate for FC layer
SCHEDULER_PATIENCE = 5            # LR scheduler patience
```

#### Loss Function

- **Mean Squared Error (MSE)**: Regression loss for continuous steering values

#### Usage

```python
from training_phase import ModelTrainer

trainer = ModelTrainer(
        train_csv_path="train.csv",
        test_csv_path="test.csv",
        model_save_dir="carla_models",
        model_name="model_checkpoint.pth"
)

trainer.run(epochs=30, save_model=True)
```

#### Training Output

```
Epoch [ 1/30] | Train Loss: 0.0234 | Test Loss: 0.0198 | LR: 0.000100
Epoch [ 2/30] | Train Loss: 0.0187 | Test Loss: 0.0165 | LR: 0.000100
...
Model saved to: carla_models/model_checkpoint.pth
```

---

### 4. Deployment Phase

**File**: `4_deployment_phase.py`

This module deploys the trained model for real-time autonomous driving in CARLA simulator.

#### Key Components

- **CarlaDeployment**: Handles model deployment
    - Loads trained model
    - Processes camera frames in real-time
    - Generates steering predictions
    - Controls vehicle autonomously
    - Logs driving behavior

#### Deployment Pipeline

```
Camera Frame → ROI Masking → Preprocessing → Model Inference → Vehicle Control
```

#### Configuration

```python
DEFAULT_THROTTLE = 0.4            # Constant throttle
FIXED_DELTA_SECONDS = 0.1         # Synchronous mode timestep
LOG_INTERVAL = 1.0                # Logging frequency (seconds)
IMAGE_SIZE = (224, 224)           # Model input size
```

#### Usage

```python
from deployment_phase import CarlaDeployment

deployment = CarlaDeployment(model_name="model_checkpoint_v6.pth")
deployment.run(
        duration=None,  # Run until Ctrl+C
        log_filename="steering_trace.csv",
        spawn_point_index=1
)
```

#### Real-time Processing

1. **Image Capture**: Camera sensor provides RGB frames
2. **ROI Masking**: Apply region selection to focus on road
3. **Preprocessing**: Resize and normalize image
4. **Prediction**: Model predicts steering angle
5. **Control**: Apply predicted steering with constant throttle
6. **Logging**: Record steering commands and vehicle position

#### Output Log

The system generates `steering_trace.csv` with:
- Timestamp
- Predicted steering angle
- Throttle value
- Vehicle position (x, y, z)

---

## Dataset Format

### Training/Test CSV Files

**Files**: `train.csv`, `test.csv`

These CSV files contain the training and testing datasets with vehicle telemetry and image paths.

#### Column Descriptions

| Column | Type | Description |
|--------|------|-------------|
| Frame id | int | Sequential frame number |
| Time(s) | float | Simulation timestamp in seconds |
| Velocity (x) | float | Velocity in x-direction (m/s) |
| Velocity(y) | float | Velocity in y-direction (m/s) |
| Velocity(z) | float | Velocity in z-direction (m/s) |
| Velocity | float | Velocity magnitude (m/s) |
| Throttle | float | Throttle input [0.0-1.0] |
| Steer | float | **Steering angle (target variable)** |
| Brake | float | Brake input [0.0-1.0] |
| Handbrake | bool | Handbrake state |
| Reverse | bool | Reverse gear state |
| Manual Gear Shift | bool | Manual transmission mode |
| Gear | int | Current gear number |
| Image Path | string | Path to corresponding camera image |

#### Example Row

```csv
Frame id,Time(s),Velocity (x),Velocity(y),Velocity(z),Velocity,Throttle,Steer,Brake,Handbrake,Reverse,Manual Gear Shift,Gear,Image Path
5,3.09,8.70,-0.066,0.0001,8.70,0.0,0.0006,0.0,False,False,False,2,G:/carla_data_T4/CCW5/5.png
```

---

## Configuration

### Weather Settings

Customize weather conditions in data generation and deployment:

```python
weather.cloudiness = 100      # Cloud coverage (0-100)
weather.fog_density = 0       # Fog intensity (0-100)
weather.precipitation = 40    # Rain intensity (0-100)
```

### Camera Parameters

Adjust camera placement and properties:

```python
CAMERA_WIDTH = 640            # Resolution width
CAMERA_HEIGHT = 480           # Resolution height
CAMERA_FOV = 90               # Field of view (degrees)
CAMERA_OFFSET_X = 1.2         # Forward offset from vehicle center
CAMERA_OFFSET_Z = 1.5         # Height above ground
```

---

## Usage Examples

### Complete Pipeline Execution

#### 1. Start CARLA simulator 
```bash
./CarlaUE4.exe
```

#### 2. Generate training data
```bash
python 1_data_generation_phase.py
```

#### 3. Process images 
```bash
python 2_image_processing_phase.py
```

#### 4. Train model 
```bash
python 3_training_phase.py
```

#### 5. Deploy trained model to test autonomous vehicle 
```bash
python 4_deployment_phase.py
```

### Custom Training Configuration

```python
trainer = ModelTrainer(
        train_csv_path="custom_train.csv",
        test_csv_path="custom_test.csv",
        model_save_dir="my_models",
        model_name="steering_model.pth"
)

trainer.run(epochs=50, save_model=True)
```

### Deployment with Different Spawn Points

```python
deployment = CarlaDeployment(model_name="model_checkpoint.pth")
deployment.run(
        duration=300,  # 5 minutes
        log_filename="drive_log_spawn_25.csv",
        spawn_point_index=25
)
```

---

## Model Architecture

### ResNet18 Transfer Learning

The system uses ResNet18 pre-trained on ImageNet with the following modifications:

1. **Frozen Layers**: Conv1, Layer1-3 (feature extraction)
2. **Trainable Layers**: Layer4 (fine-tuning) + FC (regression head)
3. **Output**: Single continuous value (steering angle)

### Why ResNet18?

- Proven feature extraction capability
- Efficient for real-time inference
- Transfer learning reduces training time
- Good balance between accuracy and speed

---

## Results

### Training Metrics

- **Training Loss**: Converges to ~0.01-0.02 MSE
- **Validation Loss**: Typically 0.015-0.025 MSE
- **Training Time**: ~2-3 hours on NVIDIA RTX 3060

### Deployment Performance

- **Inference Speed**: ~30-60 FPS
- **Control Latency**: <100ms
- **Autonomous Driving**: Successfully navigates trained routes

---

## Troubleshooting

### Common Issues

**Problem**: `Connection refused` error
```
Solution: Ensure CARLA server is running before executing scripts
```

**Problem**: Out of memory during training
```
Solution: Reduce TRAIN_BATCH_SIZE in training_phase.py
```

**Problem**: Camera images not saving
```
Solution: Check disk space and write permissions for output directory
```

**Problem**: Model predictions are erratic
```
Solution: Verify ROI masking is applied consistently in training and deployment
```

---

## License

This project is provided for educational and research purposes.

---

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository.
2. Create a feature branch (`git checkout -b feature/improvement`).
3. Commit changes (`git commit -am 'Add new feature'`).
4. Push to branch (`git push origin feature/improvement`).
5. Open a Pull Request.

---

## Contact

For questions or issues, please open an issue on GitHub.

