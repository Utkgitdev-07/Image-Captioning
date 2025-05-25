import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    # Paths
    DATA_DIR = os.getenv('DATA_DIR', 'data')
    COCO_DIR = os.path.join(DATA_DIR, 'coco')
    MODELS_DIR = os.path.join(DATA_DIR, 'models')
    
    # Model parameters
    VOCAB_SIZE = 10000
    EMBED_DIM = 256
    HIDDEN_DIM = 512
    NUM_LAYERS = 2
    
    # Training parameters
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 50
    
    # Image preprocessing
    IMAGE_SIZE = (224, 224)
    MAX_CAPTION_LENGTH = 50
    
    # YOLOv8 configuration
    YOLO_MODEL = 'yolov8n-seg.pt'  # Can be 'yolov8n-seg.pt', 'yolov8s-seg.pt', etc.
    CONFIDENCE_THRESHOLD = 0.25
    
    # Flask configuration
    UPLOAD_FOLDER = os.path.join('app', 'static', 'uploads')
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
    
    @staticmethod
    def setup():
        """Create necessary directories if they don't exist"""
        dirs = [Config.DATA_DIR, Config.COCO_DIR, Config.MODELS_DIR, 
                Config.UPLOAD_FOLDER]
        for dir_path in dirs:
            os.makedirs(dir_path, exist_ok=True) 