"""
Image Processing Phase

This module processes CARLA-generated images by applying region of interest
masking to focus on the relevant road area for training.
"""

import os
from typing import Optional, Tuple
import cv2
import numpy as np


class ImageProcessor:
    """Handles batch processing of images with region of interest selection."""
    
    # Configuration constants
    SOURCE_BASE_PATH = 'G:/carla_data_T4_original/'
    DESTINATION_BASE_PATH = 'G:/carla_data_T4/'
    DEFAULT_FOLDER = 'CW12'
    
    # Region of interest parameters (as fractions of image dimensions)
    ROI_BOTTOM_LEFT_X = 0.0
    ROI_BOTTOM_LEFT_Y = 1.0
    ROI_TOP_LEFT_X = 0.4
    ROI_TOP_LEFT_Y = 0.49
    ROI_BOTTOM_RIGHT_X = 1.0
    ROI_BOTTOM_RIGHT_Y = 1.0
    ROI_TOP_RIGHT_X = 0.6
    ROI_TOP_RIGHT_Y = 0.49
    
    # Supported image formats
    SUPPORTED_FORMATS = ('.png', '.jpg', '.jpeg')
    
    def __init__(self, folder_name: str = DEFAULT_FOLDER):
        """
        Initialize the image processor.
        
        Args:
            folder_name: Name of the folder containing images to process
        """
        self.folder_name = folder_name
        self.source_path = os.path.join(self.SOURCE_BASE_PATH, folder_name)
        self.destination_path = os.path.join(self.DESTINATION_BASE_PATH, folder_name)
        self.processed_count = 0
        self.skipped_count = 0
        self.error_count = 0
    
    def apply_region_of_interest(self, image: np.ndarray) -> np.ndarray:
        """
        Apply region of interest mask to focus on the road area.
        
        This method creates a trapezoidal mask based on the camera placement
        to isolate the relevant road area in the image.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Masked image with only the region of interest visible
        """
        # Create mask with same dimensions as input image
        mask = np.zeros_like(image)
        
        # Determine mask color based on image channels
        if len(image.shape) > 2:
            # Multi-channel image (e.g., RGB)
            channel_count = image.shape[2]
            mask_color = (255,) * channel_count
        else:
            # Single-channel image (e.g., grayscale)
            mask_color = 255
        
        # Define trapezoidal region of interest
        rows, cols = image.shape[:2]
        bottom_left = [cols * self.ROI_BOTTOM_LEFT_X, rows * self.ROI_BOTTOM_LEFT_Y]
        top_left = [cols * self.ROI_TOP_LEFT_X, rows * self.ROI_TOP_LEFT_Y]
        bottom_right = [cols * self.ROI_BOTTOM_RIGHT_X, rows * self.ROI_BOTTOM_RIGHT_Y]
        top_right = [cols * self.ROI_TOP_RIGHT_X, rows * self.ROI_TOP_RIGHT_Y]
        
        vertices = np.array(
            [[bottom_left, top_left, top_right, bottom_right]], 
            dtype=np.int32
        )
        
        # Fill polygon and apply mask
        cv2.fillPoly(mask, vertices, mask_color)
        masked_image = cv2.bitwise_and(image, mask)
        
        return masked_image
    
    def process_single_image(self, filename: str) -> bool:
        """
        Process a single image file.
        
        Args:
            filename: Name of the image file to process
            
        Returns:
            True if processing was successful, False otherwise
        """
        input_path = os.path.join(self.source_path, filename)
        output_path = os.path.join(self.destination_path, filename)
        
        try:
            # Read image
            image = cv2.imread(input_path)
            if image is None:
                print(f"Warning: Could not read {filename}. Skipping...")
                return False
            
            # Apply region of interest
            processed_image = self.apply_region_of_interest(image)
            if processed_image is None:
                print(f"Warning: Processing failed for {filename}. Skipping...")
                return False
            
            # Save processed image
            success = cv2.imwrite(output_path, processed_image)
            if not success:
                print(f"Warning: Could not save {filename}. Skipping...")
                return False
            
            return True
            
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            return False
    
    def validate_paths(self) -> bool:
        """
        Validate that source directory exists.
        
        Returns:
            True if validation passes, False otherwise
        """
        if not os.path.exists(self.source_path):
            print(f"Error: Source path does not exist: {self.source_path}")
            return False
        
        if not os.path.isdir(self.source_path):
            print(f"Error: Source path is not a directory: {self.source_path}")
            return False
        
        return True
    
    def create_output_directory(self) -> None:
        """Create output directory if it doesn't exist."""
        os.makedirs(self.destination_path, exist_ok=True)
    
    def get_image_files(self) -> list:
        """
        Get list of valid image files in source directory.
        
        Returns:
            List of image filenames
        """
        try:
            all_files = os.listdir(self.source_path)
            image_files = [
                f for f in all_files 
                if f.lower().endswith(self.SUPPORTED_FORMATS)
            ]
            return image_files
        except Exception as e:
            print(f"Error listing directory: {e}")
            return []
    
    def process_batch(self) -> Tuple[int, int, int]:
        """
        Process all images in the source directory.
        
        Returns:
            Tuple of (processed_count, skipped_count, error_count)
        """
        # Validate and prepare
        if not self.validate_paths():
            return (0, 0, 0)
        
        self.create_output_directory()
        
        # Get image files
        image_files = self.get_image_files()
        if not image_files:
            print(f"No valid image files found in {self.source_path}")
            return (0, 0, 0)
        
        print(f"Found {len(image_files)} image(s) to process")
        
        # Process each image
        for filename in image_files:
            success = self.process_single_image(filename)
            if success:
                self.processed_count += 1
            else:
                self.skipped_count += 1
        
        return (self.processed_count, self.skipped_count, self.error_count)
    
    def run(self) -> None:
        """Execute the batch processing workflow."""
        print(f"Starting image processing for folder: {self.folder_name}")
        print(f"Source: {self.source_path}")
        print(f"Destination: {self.destination_path}")
        
        try:
            processed, skipped, errors = self.process_batch()
            
            print("\n" + "="*50)
            print("Processing Summary:")
            print(f"  Successfully processed: {processed}")
            print(f"  Skipped: {skipped}")
            print(f"  Total files: {processed + skipped}")
            print("="*50)
            
        except Exception as e:
            print(f"Fatal error during batch processing: {e}")


def main():
    """Entry point for the image processing phase."""
    processor = ImageProcessor(folder_name="CW12")
    processor.run()


if __name__ == '__main__':
    main()