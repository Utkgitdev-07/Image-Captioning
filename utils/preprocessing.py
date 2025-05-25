import torch
import torchvision.transforms as transforms
from PIL import Image
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from config.config import Config

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

class ImagePreprocessor:
    def __init__(self):
        self.transform = transforms.Compose([
            transforms.Resize(Config.IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
        ])
    
    def preprocess(self, image_path):
        """
        Preprocess an image for model input
        
        Args:
            image_path (str): Path to the image
            
        Returns:
            torch.Tensor: Preprocessed image tensor
        """
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)
        return image.unsqueeze(0)  # Add batch dimension

class CaptionPreprocessor:
    def __init__(self):
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words('english'))
        
    def preprocess(self, caption):
        """
        Preprocess a caption text
        
        Args:
            caption (str): Input caption
            
        Returns:
            list: List of preprocessed tokens
        """
        # Tokenize
        tokens = word_tokenize(caption.lower())
        
        # Remove stopwords and stem
        tokens = [self.stemmer.stem(token) 
                 for token in tokens 
                 if token.isalnum() and token not in self.stop_words]
        
        return tokens
    
    def clean_caption(self, caption):
        """
        Clean caption text without removing stopwords or stemming
        
        Args:
            caption (str): Input caption
            
        Returns:
            str: Cleaned caption
        """
        # Tokenize and clean
        tokens = word_tokenize(caption.lower())
        tokens = [token for token in tokens if token.isalnum()]
        
        return ' '.join(tokens)

def prepare_image_for_captioning(image_path):
    """Prepare an image for the captioning model"""
    preprocessor = ImagePreprocessor()
    return preprocessor.preprocess(image_path)

def prepare_image_for_segmentation(image_path):
    """Prepare an image for the segmentation model"""
    # YOLOv8 handles preprocessing internally
    return image_path 