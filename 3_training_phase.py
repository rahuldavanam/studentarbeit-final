"""
Training Phase

This module trains a ResNet18 model for autonomous steering prediction
using CARLA simulator data. The model predicts steering angles from camera images.
"""

import os
from pathlib import Path
from typing import Dict, Tuple, List, Optional
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights
from PIL import Image
import pandas as pd


class CarlaCustomDataset(Dataset):
    """Custom Dataset for loading CARLA simulator data with images and steering labels."""
    
    def __init__(self, csv_path: str, transform: Optional[transforms.Compose] = None):
        """
        Initialize the dataset.
        
        Args:
            csv_path: Path to CSV file containing image paths and steering values
            transform: Optional transforms to apply to images
        """
        self.csv_path = csv_path
        self.data = pd.read_csv(csv_path)
        self.transform = transform
        
        # Pre-filter data to include only valid images
        self.valid_data = []
        print(f"Loading dataset from {csv_path}...")
        
        for _, row in self.data.iterrows():
            img_path = row['Image Path']
            label = row['Steer']
            
            if os.path.exists(img_path):
                try:
                    # Verify image can be loaded
                    with Image.open(img_path) as img:
                        img.convert("RGB")
                    self.valid_data.append((img_path, label))
                except Exception as e:
                    print(f"Skipping corrupted image {img_path}: {e}")
            else:
                print(f"Image not found: {img_path}")
        
        print(f"Loaded {len(self.valid_data)} valid samples")
    
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.valid_data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single sample from the dataset.
        
        Args:
            idx: Index of the sample to retrieve
            
        Returns:
            Dictionary containing 'image' and 'steer' tensors
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        # Load image and label
        img_path, steer = self.valid_data[idx]
        image = Image.open(img_path).convert('RGB')
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        return {'image': image, 'steer': steer}


class ModelTrainer:
    """Handles model training and evaluation for steering prediction."""
    
    # Configuration constants
    IMAGE_SIZE = (224, 224)
    TRAIN_BATCH_SIZE = 64
    TEST_BATCH_SIZE = 1
    NUM_WORKERS = 0
    MODEL_OUTPUT_DIM = 1  # Steering only
    LAYER4_LR = 1e-5
    FC_LR = 1e-4
    SCHEDULER_PATIENCE = 5
    SCHEDULER_FACTOR = 0.1
    DEFAULT_EPOCHS = 30
    
    def __init__(
        self,
        train_csv_path: str = "train.csv",
        test_csv_path: str = "test.csv",
        model_save_dir: str = "carla_models",
        model_name: str = "model_checkpoint.pth"
    ):
        """
        Initialize the model trainer.
        
        Args:
            train_csv_path: Path to training data CSV
            test_csv_path: Path to test data CSV
            model_save_dir: Directory to save trained models
            model_name: Name for the saved model file
        """
        self.train_csv_path = train_csv_path
        self.test_csv_path = test_csv_path
        self.model_save_dir = Path(model_save_dir)
        self.model_name = model_name
        
        # Initialize device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Training history
        self.train_loss_history: List[float] = []
        self.test_loss_history: List[float] = []
        
        # Initialize components (to be set up later)
        self.model: Optional[nn.Module] = None
        self.loss_fn: Optional[nn.Module] = None
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None
        self.train_dataloader: Optional[DataLoader] = None
        self.test_dataloader: Optional[DataLoader] = None
    
    def setup_data(self) -> None:
        """Create datasets and dataloaders."""
        # Define transforms
        transform = transforms.Compose([
            transforms.Resize(self.IMAGE_SIZE),
            transforms.ToTensor()
        ])
        
        # Create datasets
        train_dataset = CarlaCustomDataset(
            csv_path=self.train_csv_path,
            transform=transform
        )
        test_dataset = CarlaCustomDataset(
            csv_path=self.test_csv_path,
            transform=transform
        )
        
        # Create dataloaders
        self.train_dataloader = DataLoader(
            dataset=train_dataset,
            batch_size=self.TRAIN_BATCH_SIZE,
            num_workers=self.NUM_WORKERS,
            shuffle=True
        )
        
        self.test_dataloader = DataLoader(
            dataset=test_dataset,
            batch_size=self.TEST_BATCH_SIZE,
            num_workers=self.NUM_WORKERS,
            shuffle=False
        )
        
        print(f"Training batches: {len(self.train_dataloader)}")
        print(f"Test batches: {len(self.test_dataloader)}")
    
    def setup_model(self) -> None:
        """Initialize and configure the ResNet18 model."""
        # Load pre-trained ResNet18
        self.model = resnet18(weights=ResNet18_Weights.DEFAULT)
        
        # Modify final layer for regression
        self.model.fc = nn.Linear(512, self.MODEL_OUTPUT_DIM)
        
        # Freeze all layers initially
        for param in self.model.parameters():
            param.requires_grad = False
        
        # Unfreeze layer4 and fc for fine-tuning
        for param in self.model.layer4.parameters():
            param.requires_grad = True
        for param in self.model.fc.parameters():
            param.requires_grad = True
        
        # Move model to device
        self.model = self.model.to(self.device)
        
        # Count trainable parameters
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"Model parameters: {trainable_params:,} trainable / {total_params:,} total")
    
    def setup_training_components(self) -> None:
        """Initialize loss function, optimizer, and scheduler."""
        # Loss function for regression
        self.loss_fn = nn.MSELoss()
        
        # Optimizer with different learning rates for different layers
        self.optimizer = torch.optim.Adam([
            {'params': self.model.layer4.parameters(), 'lr': self.LAYER4_LR},
            {'params': self.model.fc.parameters(), 'lr': self.FC_LR}
        ])
        
        # Learning rate scheduler
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=self.SCHEDULER_FACTOR,
            patience=self.SCHEDULER_PATIENCE
        )
        
        print(f"Training components configured:")
        print(f"  Loss function: MSE")
        print(f"  Layer4 LR: {self.LAYER4_LR}")
        print(f"  FC LR: {self.FC_LR}")
        print(f"  Scheduler: ReduceLROnPlateau (patience={self.SCHEDULER_PATIENCE})")
    
    def train_epoch(self) -> float:
        """
        Execute one training epoch.
        
        Returns:
            Average training loss for the epoch
        """
        self.model.train()
        epoch_loss = 0.0
        
        for batch in self.train_dataloader:
            # Move data to device
            images = batch['image'].to(self.device)
            labels = batch['steer'].to(self.device).float()
            
            # Forward pass
            predictions = self.model(images).squeeze(dim=1)
            
            # Calculate loss
            loss = self.loss_fn(predictions, labels)
            epoch_loss += loss.item()
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        
        # Calculate average loss
        avg_loss = epoch_loss / len(self.train_dataloader)
        self.train_loss_history.append(avg_loss)
        
        return avg_loss
    
    def evaluate(self) -> float:
        """
        Evaluate the model on test data.
        
        Returns:
            Average test loss
        """
        self.model.eval()
        epoch_loss = 0.0
        
        with torch.inference_mode():
            for batch in self.test_dataloader:
                # Move data to device
                images = batch['image'].to(self.device)
                labels = batch['steer'].to(self.device).float()
                
                # Forward pass
                predictions = self.model(images).squeeze(dim=1)
                
                # Calculate loss
                loss = self.loss_fn(predictions, labels)
                epoch_loss += loss.item()
        
        # Calculate average loss
        avg_loss = epoch_loss / len(self.test_dataloader)
        self.test_loss_history.append(avg_loss)
        
        return avg_loss
    
    def train(self, epochs: int = DEFAULT_EPOCHS) -> None:
        """
        Execute the complete training loop.
        
        Args:
            epochs: Number of training epochs
        """
        print(f"\n{'='*60}")
        print(f"Starting training for {epochs} epochs")
        print(f"{'='*60}\n")
        
        for epoch in range(epochs):
            # Train and evaluate
            train_loss = self.train_epoch()
            test_loss = self.evaluate()
            
            # Adjust learning rate
            self.scheduler.step(test_loss)
            
            # Get current learning rate
            current_lr = self.scheduler.get_last_lr()[0]
            
            # Print progress
            print(
                f"Epoch [{epoch+1:2d}/{epochs}] | "
                f"Train Loss: {train_loss:.4f} | "
                f"Test Loss: {test_loss:.4f} | "
                f"LR: {current_lr:.6f}"
            )
        
        print(f"\n{'='*60}")
        print("Training complete!")
        print(f"{'='*60}\n")
    
    def save_model(self) -> None:
        """Save the trained model to disk."""
        # Create save directory
        self.model_save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model
        model_path = self.model_save_dir / self.model_name
        torch.save(self.model.state_dict(), model_path)
        
        print(f"Model saved to: {model_path}")
    
    def get_training_summary(self) -> Dict[str, float]:
        """
        Get summary statistics of the training process.
        
        Returns:
            Dictionary with training metrics
        """
        if not self.train_loss_history or not self.test_loss_history:
            return {}
        
        return {
            'final_train_loss': self.train_loss_history[-1],
            'final_test_loss': self.test_loss_history[-1],
            'best_train_loss': min(self.train_loss_history),
            'best_test_loss': min(self.test_loss_history),
            'total_epochs': len(self.train_loss_history)
        }
    
    def run(self, epochs: int = DEFAULT_EPOCHS, save_model: bool = True) -> None:
        """
        Execute the complete training pipeline.
        
        Args:
            epochs: Number of training epochs
            save_model: Whether to save the model after training
        """
        # Setup all components
        self.setup_data()
        self.setup_model()
        self.setup_training_components()
        
        # Train the model
        self.train(epochs=epochs)
        
        # Save model if requested
        if save_model:
            self.save_model()
        
        # Print summary
        summary = self.get_training_summary()
        print("\nTraining Summary:")
        print(f"  Final Train Loss: {summary['final_train_loss']:.4f}")
        print(f"  Final Test Loss: {summary['final_test_loss']:.4f}")
        print(f"  Best Train Loss: {summary['best_train_loss']:.4f}")
        print(f"  Best Test Loss: {summary['best_test_loss']:.4f}")


def main():
    """Entry point for the training phase."""
    trainer = ModelTrainer(
        train_csv_path="train.csv",
        test_csv_path="test.csv",
        model_save_dir="carla_models",
        model_name="model_checkpoint.pth"
    )
    
    trainer.run(epochs=30, save_model=True)


if __name__ == '__main__':
    main()
