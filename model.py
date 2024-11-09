import pandas as pd
from sklearn import linear_model
import pdfplumber
from docx import Document
from io import BytesIO
import re

class train_model:
    def train(self):
        data = pd.read_csv('training_dataset.csv')
        array = data.values
        for i in range(len(array)):
            array[i][0] = 1 if array[i][0] == "Male" else 0

        df = pd.DataFrame(array)
        maindf = df[[0, 1, 2, 3, 4, 5, 6]]
        mainarray = maindf.values
        temp = df[7]
        train_y = temp.values

        self.mul_lr = linear_model.LogisticRegression(multi_class='multinomial', solver='newton-cg', max_iter=1000)
        self.mul_lr.fit(mainarray, train_y)

    def test(self, test_data):
        try:
            test_predict = [int(i) for i in test_data]
            y_pred = self.mul_lr.predict([test_predict])
            return y_pred[0]
        except:
            return "All factors for finding personality not entered!"

# Text extraction from PDF
def extract_text_from_pdf(file_content):
    with pdfplumber.open(BytesIO(file_content)) as pdf:
        text = ''.join([page.extract_text() for page in pdf.pages])
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

    # Convert text to lower case and split into words
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
    if desired_skill in text_lower:
        analysis['matched_skills'].append(desired_skill)
        analysis['skill_status'] = f"The skill '{desired_skill}' was found in the resume."
    else:
        analysis['improvements'].append(f"The skill '{desired_skill}' was not found in the resume.")
        analysis['skill_status'] = f"The skill '{desired_skill}' was not found in the resume."

    # Use the model to predict something, if necessary
    if model:
        prediction = model.test([len(words)])  # This is just an example of how you might use the model
        analysis['predicted_value'] = prediction

    return analysis

# Prediction result
def prediction_result(file_content, file_extension, model, desired_skill):
    if file_extension == '.pdf':
        resume_text = extract_text_from_pdf(file_content)
    elif file_extension == '.docx':
        resume_text = extract_text_from_docx(file_content)
    else:
        return {'error': 'Unsupported file type'}

    analysis = analyze_resume(resume_text, model, desired_skill)
    return analysis
