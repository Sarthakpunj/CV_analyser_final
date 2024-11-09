import pandas as pd
from sklearn.linear_model import LogisticRegression
import pdfplumber
from docx import Document
from io import BytesIO
import re

class TrainModel:
    def __init__(self):
        self.model = None
        self.train()  # Automatically train the model on initialization

    def train(self):
        # Load and preprocess training data
        data = pd.read_csv('training_dataset.csv')
        
        # Convert 'Gender' column to binary values
        data.iloc[:, 0] = data.iloc[:, 0].apply(lambda x: 1 if x == "Male" else 0)
        
        # Split data into features (X) and target (y)
        train_x = data.iloc[:, :-1].values
        train_y = data.iloc[:, -1].values

        # Initialize and train the logistic regression model
        self.model = LogisticRegression(multi_class='multinomial', solver='newton-cg', max_iter=1000)
        self.model.fit(train_x, train_y)

    def test(self, test_data):
        # Predict based on test data
        try:
            y_pred = self.model.predict([test_data])
            return y_pred[0]
        except:
            return "All factors for finding personality not entered!"

# Text extraction from PDF
def extract_text_from_pdf(file_content):
    with pdfplumber.open(BytesIO(file_content)) as pdf:
        text = ''.join([page.extract_text() for page in pdf.pages if page.extract_text() is not None])
    return text

# Text extraction from DOCX
def extract_text_from_docx(file_content):
    doc = Document(BytesIO(file_content))
    text = ''.join([para.text + '\n' for para in doc.paragraphs])
    return text

# Analysis function
def analyze_resume(text, model, desired_skill):
    """Analyze the resume text and provide strengths, weaknesses, and matched skills."""
    keywords = ['experience', 'education', 'certifications', 'achievements',
                'leadership', 'teamwork', 'communication', 'problem-solving', 'project management']
    
    analysis = {'strengths': [], 'weaknesses': [], 'improvements': [], 'matched_skills': [], 'skill_status': ''}

    # Convert text to lowercase for analysis
    text_lower = text.lower()
    words = re.findall(r'\w+', text_lower)

    # Keyword Frequency Analysis
    for keyword in keywords:
        if keyword in words:
            analysis['strengths'].append(f"Strong emphasis on '{keyword}'")
        else:
            analysis['improvements'].append(f"Consider adding more about '{keyword}'")

    # Identify Experience Level
    experience_years = re.findall(r'\d+\s*(?:years?|yrs?)', text_lower)
    if experience_years:
        total_experience = sum(int(re.findall(r'\d+', exp)[0]) for exp in experience_years)
        analysis['strengths'].append(f"Estimated total experience: {total_experience} years")

    # Skill Matching
    if desired_skill.lower() in text_lower:
        analysis['matched_skills'].append(desired_skill)
        analysis['skill_status'] = f"The skill '{desired_skill}' was found in the resume."
    else:
        analysis['improvements'].append(f"The skill '{desired_skill}' was not found in the resume.")
        analysis['skill_status'] = f"The skill '{desired_skill}' was not found in the resume."

    # Use the model to predict something, if necessary
    if model:
        word_count = len(words)  # Example feature, adjust as needed
        prediction = model.test([word_count])  # Customize input if needed based on model's training
        analysis['predicted_value'] = prediction

    return analysis

# Prediction result function
def prediction_result(file_content, file_extension, model, desired_skill):
    # Extract text based on file extension
    if file_extension.lower() == '.pdf':
        resume_text = extract_text_from_pdf(file_content)
    elif file_extension.lower() == '.docx':
        resume_text = extract_text_from_docx(file_content)
    else:
        return {'error': 'Unsupported file type'}

    # Analyze resume content
    analysis = analyze_resume(resume_text, model, desired_skill)
    return analysis
