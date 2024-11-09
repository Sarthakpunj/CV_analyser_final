from flask import Flask, render_template, request, jsonify, send_from_directory
import os
from model import TrainModel, prediction_result
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Configure upload folder
app.config['UPLOAD_FOLDER'] = 'uploads'
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# Initialize and train the model
model = TrainModel()  # Ensures the model is trained upon initialization

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Check if a file is included in the request
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # Save the file securely
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    # Get the desired skill from the form data
    desired_skill = request.form.get('skill', '').strip()
    if not desired_skill:
        return jsonify({'error': 'Desired skill not provided'}), 400

    # Determine file extension and read the file content
    file_extension = os.path.splitext(filename)[1].lower()
    with open(filepath, 'rb') as file_content:
        # Run analysis on the resume content
        analysis = prediction_result(file_content.read(), file_extension, model, desired_skill)

    return jsonify(analysis)

@app.route('/uploads/<filename>')
def serve_file(filename):
    # Serve files from the uploads folder
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)
