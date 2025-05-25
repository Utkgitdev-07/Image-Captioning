import os
from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename
import torch
from models.caption_model import CaptionModel
from models.segment_model import SegmentationModel
from utils.preprocessing import prepare_image_for_captioning, prepare_image_for_segmentation
from config.config import Config

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = Config.UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize models
caption_model = CaptionModel()
segment_model = SegmentationModel()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in Config.ALLOWED_EXTENSIONS

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process_image():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    if file and allowed_file(file.filename):
        # Save the uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            # Generate caption
            image_tensor = prepare_image_for_captioning(filepath)
            caption = caption_model.generate_caption(image_tensor, max_length=Config.MAX_CAPTION_LENGTH)
            
            # Perform segmentation
            segmentation_results = segment_model.predict(filepath)
            
            # Save visualization
            vis_path = os.path.join(app.config['UPLOAD_FOLDER'], f'vis_{filename}')
            segment_model.visualize(filepath, save_path=vis_path)
            
            return jsonify({
                'success': True,
                'caption': caption,
                'segmentation': {
                    'classes': segmentation_results['classes'],
                    'scores': segmentation_results['scores'].tolist(),
                    'visualization': f'uploads/vis_{filename}'
                }
            })
            
        except Exception as e:
            return jsonify({'error': str(e)})
    
    return jsonify({'error': 'Invalid file type'})

if __name__ == '__main__':
    app.run(debug=True) 