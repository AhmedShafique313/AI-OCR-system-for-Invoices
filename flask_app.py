from flask import Flask, render_template, request, send_file, jsonify
import os
from utils.ocr import extract_text
from utils.gemini import process_with_gemini
from utils.handler import save_to_csv
import json

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

CSV_FILE_PATH = "extracted_data.csv"

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handles file upload and saves it."""
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and file.filename.lower().endswith('.png'):
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)
        return jsonify({'message': 'File uploaded successfully', 'file_path': file_path}), 200
    else:
        return jsonify({'error': 'Invalid file format. Please upload a PNG file.'}), 400

@app.route('/process', methods=['POST'])
def process_invoice():
    """Extracts text, processes it using Gemini API, and saves as CSV."""
    data = request.get_json()
    file_path = data.get('file_path')

    if not file_path or not os.path.exists(file_path):
        return jsonify({'error': 'Invalid file path'}), 400

    extracted_text = extract_text(file_path)
    json_data = process_with_gemini(extracted_text)

    if not json_data:
        return jsonify({'error': 'Failed to process invoice'}), 500

    save_to_csv(json_data, CSV_FILE_PATH)

    return jsonify({'message': 'Processing complete', 'csv_path': CSV_FILE_PATH}), 200

@app.route('/download')
def download_file():
    """Allows downloading the extracted CSV file."""
    if os.path.exists(CSV_FILE_PATH):
        return send_file(CSV_FILE_PATH, as_attachment=True)
    return jsonify({'error': 'CSV file not found'}), 404

if __name__ == '__main__':
    app.run(debug=True)
