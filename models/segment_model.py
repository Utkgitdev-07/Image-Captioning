from ultralytics import YOLO
import torch
import numpy as np
from config.config import Config

class SegmentationModel:
    def __init__(self, model_name=Config.YOLO_MODEL):
        """Initialize YOLOv8 model for segmentation"""
        self.model = YOLO(model_name)
        self.confidence_threshold = Config.CONFIDENCE_THRESHOLD
        
    def predict(self, image_path):
        """
        Perform segmentation on an image
        
        Args:
            image_path (str): Path to the input image
            
        Returns:
            dict: Dictionary containing:
                - boxes: Bounding boxes
                - masks: Segmentation masks
                - classes: Detected class names
                - scores: Confidence scores
        """
        # Run inference
        results = self.model(image_path)[0]
        
        # Get predictions above confidence threshold
        boxes = results.boxes.data.cpu().numpy()
        masks = results.masks.data.cpu().numpy() if results.masks is not None else None
        
        # Filter by confidence
        confident_idx = boxes[:, 4] > self.confidence_threshold
        boxes = boxes[confident_idx]
        masks = masks[confident_idx] if masks is not None else None
        
        # Get class names and scores
        class_ids = boxes[:, 5].astype(int)
        scores = boxes[:, 4]
        classes = [results.names[class_id] for class_id in class_ids]
        
        return {
            'boxes': boxes[:, :4],  # x1, y1, x2, y2
            'masks': masks,
            'classes': classes,
            'scores': scores
        }
    
    def visualize(self, image_path, save_path=None):
        """
        Visualize segmentation results
        
        Args:
            image_path (str): Path to input image
            save_path (str, optional): Path to save visualization
        """
        results = self.model(image_path)
        
        # Plot results
        fig = results[0].plot()
        
        if save_path:
            fig.save(save_path)
            
        return fig 