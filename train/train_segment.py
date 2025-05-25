from ultralytics import YOLO
import os
from config.config import Config

def train_yolo():
    """Train YOLOv8 model for segmentation"""
    # Create YOLO model
    model = YOLO('yolov8n-seg.yaml')  # Build a new model from scratch
    
    # Train the model on COCO dataset
    results = model.train(
        data='coco.yaml',  # COCO dataset config file
        epochs=100,
        imgsz=640,
        batch=16,
        name='yolov8_segmentation',
        project=Config.MODELS_DIR
    )
    
    # Validate the model
    results = model.val()
    
    # Export the model
    success = model.export(format='onnx')  # Export to ONNX format

if __name__ == '__main__':
    # Create necessary directories
    Config.setup()
    
    # Start training
    train_yolo()