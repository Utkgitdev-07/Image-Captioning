import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
from pycocotools.coco import COCO
from config.config import Config

class Vocabulary:
    def __init__(self):
        self.word2idx = {'<PAD>': 0, '<START>': 1, '<END>': 2, '<UNK>': 3}
        self.idx2word = {0: '<PAD>', 1: '<START>', 2: '<END>', 3: '<UNK>'}
        self.idx = 4
        
    def add_word(self, word):
        if word not in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1
            
    def get_id(self, word):
        return self.word2idx.get(word, self.word2idx['<UNK>'])
    
    def get_word(self, idx):
        return self.idx2word.get(idx, '<UNK>')
    
    def __len__(self):
        return len(self.word2idx)

class CocoDataset(Dataset):
    def __init__(self, root_dir, ann_file, vocab, transform=None):
        self.root_dir = root_dir
        self.coco = COCO(ann_file)
        self.ids = list(self.coco.anns.keys())
        self.vocab = vocab
        self.transform = transform or transforms.Compose([
            transforms.Resize(Config.IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
        ])
        
    def __getitem__(self, index):
        ann_id = self.ids[index]
        caption = self.coco.anns[ann_id]['caption']
        img_id = self.coco.anns[ann_id]['image_id']
        path = self.coco.loadImgs(img_id)[0]['file_name']
        
        # Load image
        image = Image.open(os.path.join(self.root_dir, path)).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
            
        # Convert caption to tensor
        tokens = caption.lower().split()
        caption = []
        caption.append(self.vocab.get_id('<START>'))
        caption.extend([self.vocab.get_id(token) for token in tokens])
        caption.append(self.vocab.get_id('<END>'))
        caption = torch.Tensor(caption).long()
        
        return image, caption
    
    def __len__(self):
        return len(self.ids)

def get_loader(root_dir, ann_file, vocab, batch_size=32, shuffle=True, num_workers=4):
    """Create a data loader for the COCO dataset"""
    dataset = CocoDataset(root_dir=root_dir,
                         ann_file=ann_file,
                         vocab=vocab)
    
    data_loader = DataLoader(dataset=dataset,
                           batch_size=batch_size,
                           shuffle=shuffle,
                           num_workers=num_workers,
                           collate_fn=collate_fn)
    return data_loader

def collate_fn(data):
    """Create mini-batch tensors from a list of (image, caption) tuples"""
    # Sort data by caption length
    data.sort(key=lambda x: len(x[1]), reverse=True)
    images, captions = zip(*data)
    
    # Stack images
    images = torch.stack(images, 0)
    
    # Merge captions (from tuple of 1D tensor to 2D tensor)
    lengths = [len(cap) for cap in captions]
    targets = torch.zeros(len(captions), max(lengths)).long()
    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[:end]
        
    return images, targets, lengths