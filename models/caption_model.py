import torch
import torch.nn as nn
import torchvision.models as models
from config.config import Config

class EncoderCNN(nn.Module):
    def __init__(self):
        super(EncoderCNN, self).__init__()
        # Load pre-trained VGG16 model
        vgg16 = models.vgg16(pretrained=True)
        # Remove the last fully connected layer
        modules = list(vgg16.children())[:-1]
        self.vgg16 = nn.Sequential(*modules)
        # Freeze the parameters
        for param in self.vgg16.parameters():
            param.requires_grad = False
            
    def forward(self, images):
        features = self.vgg16(images)
        features = features.view(features.size(0), -1)
        return features

class DecoderRNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers):
        super(DecoderRNN, self).__init__()
        
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_dim, vocab_size)
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, features, captions):
        embeddings = self.dropout(self.embed(captions))
        hiddens, _ = self.lstm(embeddings)
        outputs = self.linear(hiddens)
        return outputs

class CaptionModel(nn.Module):
    def __init__(self, vocab_size=Config.VOCAB_SIZE, 
                 embed_dim=Config.EMBED_DIM,
                 hidden_dim=Config.HIDDEN_DIM,
                 num_layers=Config.NUM_LAYERS):
        super(CaptionModel, self).__init__()
        
        self.encoder = EncoderCNN()
        self.decoder = DecoderRNN(vocab_size, embed_dim, hidden_dim, num_layers)
        
    def forward(self, images, captions):
        features = self.encoder(images)
        outputs = self.decoder(features, captions)
        return outputs
    
    def generate_caption(self, image, vocabulary, max_length=20):
        """Generate caption for a single image"""
        self.eval()
        with torch.no_grad():
            features = self.encoder(image.unsqueeze(0))
            
            # Start with <START> token
            ids = []
            states = None
            
            # Generate caption word by word
            for i in range(max_length):
                current_word = torch.tensor([vocabulary.get_id('<START>')] if not ids else [ids[-1]])
                embeddings = self.decoder.embed(current_word)
                
                hiddens, states = self.decoder.lstm(embeddings.unsqueeze(1), states)
                outputs = self.decoder.linear(hiddens.squeeze(1))
                
                predicted = outputs.argmax(1)
                ids.append(predicted.item())
                
                if vocabulary.get_word(predicted.item()) == '<END>':
                    break
            
            # Convert indices to words
            caption = [vocabulary.get_word(idx) for idx in ids]
            caption = ' '.join(caption[1:-1])  # Remove <START> and <END> tokens
            
        self.train()
        return caption 