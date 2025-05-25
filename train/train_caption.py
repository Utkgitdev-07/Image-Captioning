import os
import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm
import json

from models.caption_model import CaptionModel
from utils.data_loader import get_loader, Vocabulary
from config.config import Config

def train_caption_model(data_loader, model, criterion, optimizer, epoch, device):
    """Train the model for one epoch"""
    model.train()
    total_loss = 0
    
    with tqdm(data_loader, desc=f'Epoch {epoch}') as pbar:
        for images, captions, lengths in pbar:
            # Move to device
            images = images.to(device)
            captions = captions.to(device)
            
            # Zero the gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(images, captions)
            
            # Calculate loss
            loss = criterion(outputs.view(-1, outputs.size(-1)), 
                           captions.view(-1))
            
            # Backward pass
            loss.backward()
            
            # Update weights
            optimizer.step()
            
            # Update statistics
            total_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
    
    return total_loss / len(data_loader)

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create vocabulary
    vocab = Vocabulary()
    
    # Load COCO dataset
    train_loader = get_loader(
        root_dir=os.path.join(Config.COCO_DIR, 'train2017'),
        ann_file=os.path.join(Config.COCO_DIR, 'annotations/captions_train2017.json'),
        vocab=vocab,
        batch_size=Config.BATCH_SIZE
    )
    
    val_loader = get_loader(
        root_dir=os.path.join(Config.COCO_DIR, 'val2017'),
        ann_file=os.path.join(Config.COCO_DIR, 'annotations/captions_val2017.json'),
        vocab=vocab,
        batch_size=Config.BATCH_SIZE
    )
    
    # Initialize model
    model = CaptionModel(
        vocab_size=len(vocab),
        embed_dim=Config.EMBED_DIM,
        hidden_dim=Config.HIDDEN_DIM,
        num_layers=Config.NUM_LAYERS
    ).to(device)
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss(ignore_index=vocab.get_id('<PAD>'))
    optimizer = Adam(model.parameters(), lr=Config.LEARNING_RATE)
    
    # Training loop
    best_val_loss = float('inf')
    for epoch in range(Config.NUM_EPOCHS):
        # Train
        train_loss = train_caption_model(
            train_loader, model, criterion, optimizer, epoch, device
        )
        
        # Validate
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for images, captions, lengths in val_loader:
                images = images.to(device)
                captions = captions.to(device)
                outputs = model(images, captions)
                loss = criterion(outputs.view(-1, outputs.size(-1)), 
                               captions.view(-1))
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        
        print(f'Epoch {epoch}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}')
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'vocab': vocab,
                'val_loss': val_loss
            }, os.path.join(Config.MODELS_DIR, 'caption_model_best.pth'))

    results = model.train(
        data='coco.yaml',  # COCO dataset config file
        epochs=100,
        imgsz=640,
        batch=16,
        name='yolov8_segmentation',
        project=Config.MODELS_DIR
    )

if __name__ == '__main__':
    # Create necessary directories
    Config.setup()
    main() 