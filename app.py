from flask import Flask, render_template, request, jsonify
import pandas as pd
from sklearn import linear_model
import pdfplumber
from docx import Document
from io import BytesIO
import re

app = Flask(__name__)

# Model Training Class
class TrainModel:
    def __init__(self):
        self.mul_lr = linear_model.LogisticRegression(multi_class='multinomial', solver='newton-cg', max_iter=1000)
        self.train()

    def train(self):
        # Load and process the training data
        data = pd.read_csv('training_dataset.csv')
        data['Gender'] = data['Gender'].apply(lambda x: 1 if x == 'Male' else 0)
        train_x = data.iloc[:, :-1].values
        train_y = data.iloc[:, -1].values
        self.mul_lr.fit(train_x, train_y)

    def predict(self, features):
        try:
            return self.mul_lr.predict([features])[0]
        except Exception as e:
            return str(e)

# Initialize and train the model
model = TrainModel()

# PDF text extraction
def extract_text_from_pdf(file):
    with pdfplumber.open(BytesIO(file)) as pdf:
        return ' '.join([page.extract_text() for page in pdf.pages])

# DOCX text extraction
def extract_text_from_docx(file):
    doc = Document(BytesIO(file))
    return ' '.join([para.text for para in doc.paragraphs])

# Resume analysis
def analyze_resume(text, desired_skill):
    keywords = ['experience', 'education', 'certifications', 'achievements',
                'leadership', 'teamwork', 'communication', 'problem-solving', 'project management']
    analysis = {'strengths': [], 'improvements': [], 'matched_skills': [], 'skill_status': ''}

    text_lower = text.lower()
    for keyword in keywords:
        if keyword in text_lower:
            analysis['strengths'].append(f"Strong emphasis on '{keyword}'")
        else:
            analysis['improvements'].append(f"Consider adding more about '{keyword}'")

    if desired_skill.lower() in text_lower:
        analysis['matched_skills'].append(desired_skill)
        analysis['skill_status'] = f"The skill '{desired_skill}' was found in the resume."
    else:
        analysis['improvements'].append(f"The skill '{desired_skill}' was not found in the resume.")
        analysis['skill_status'] = f"The skill '{desired_skill}' was not found in the resume."

    return analysis

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files or request.files['file'].filename == '':
        return jsonify({'error': 'No file uploaded'})

    file = request.files['file']
    file_extension = file.filename.split('.')[-1].lower()

    if file_extension == 'pdf':
        resume_text = extract_text_from_pdf(file.read())
    elif file_extension == 'docx':
        resume_text = extract_text_from_docx(file.read())
    else:
        return jsonify({'error': 'Unsupported file type'})

    desired_skill = request.form.get('skill', '').strip()
    analysis = analyze_resume(resume_text, desired_skill)

    return jsonify(analysis)

if __name__ == '__main__':
    app.run(debug=True)
