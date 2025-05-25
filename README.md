# Image Caption and Segmentation Project

This project combines image captioning and segmentation using deep learning models. It uses VGG16+LSTM for captioning and YOLOv8 for segmentation.

## Features
- Image Captioning using VGG16 encoder and LSTM decoder
- Image Segmentation using YOLOv8
- Flask web interface for easy interaction
- Support for MS COCO dataset

## Setup

1. Download the MS COCO dataset:
- https://www.kaggle.com/datasets/awsaf49/coco-2017-dataset/data
- Place them in the appropriate directory structure:
```
data/
└── coco/
    ├── train2017/
    ├── val2017/
    └── annotations/
```

2. Create a virtual environment:
```
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate
```

3. Install dependencies:
```
pip install -r requirements.txt
```

4. Configure the environment:
- Create a `.env` file and set the required paths
- Download pre-trained models (will be handled by the code)

## Project Structure
```
├── data/
│   └── coco/
│    ├── train2017/
│    ├── val2017/
│    └── annotations/
├── config/
│   └── config.py           # Configuration settings
├── models/
│   ├── caption_model.py    # Image captioning model
│   └── segment_model.py    # Segmentation model
├── utils/
│   ├── data_loader.py      # Data loading utilities
│   └── preprocessing.py    # Image preprocessing
├── app/
│   ├── static/            # Static files for web interface
│   ├── templates/         # HTML templates
│   └── app.py            # Flask application
└── train/
    ├── train_caption.py   # Training script for captioning
    └── train_segment.py   # Training script for segmentation
├── requirements.txt
├── README.md
```

## Usage
1. Start the Flask server:
```bash
python app/app.py
```

2. Open a web browser and navigate to `http://localhost:5000`

## Training
1. For image captioning:
```bash
python train/train_caption.py
```

2. For segmentation:
```bash
python train/train_segment.py
```

## Models
- Captioning: VGG16 + LSTM
- Segmentation: YOLOv8

## Dataset
This project uses the MS COCO dataset, which provides:
- Over 330K images
- More than 1.5 million object instances
- 5 captions per image
- Instance segmentation masks 