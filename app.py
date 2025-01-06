from flask import Flask, render_template, request
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
import PyPDF2
import os

app = Flask(__name__)

# Load trained model and vectorizer
with open("models/tfidf_vectorizer.pkl", "rb") as vec_file:
    vectorizer = pickle.load(vec_file)

with open("models/resume_ranker_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

# Function to extract text from PDF
def extract_text_from_pdf(pdf_file):
    reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    job_role = request.form['job_role']
    uploaded_file = request.files['resume']
    
    if not uploaded_file or not job_role:
        return "Please provide a job role and upload a resume."
    
    # Extract resume text
    resume_content = ""
    if uploaded_file.filename.endswith('.pdf'):
        resume_content = extract_text_from_pdf(uploaded_file)
    elif uploaded_file.filename.endswith('.txt'):
        resume_content = uploaded_file.read().decode('utf-8')
    else:
        return "Unsupported file format. Please upload a .txt or .pdf file."
    
    # Combine job role and resume for vectorization
    data = [job_role, resume_content]
    transformed_data = vectorizer.transform(data)
    
    # Predict the relevance score for the resume based on the job role
    job_role_vector = transformed_data[0]
    resume_vector = transformed_data[1]
    score = model.predict_proba(resume_vector.reshape(1, -1))[0][1] * 100  # Percentage score

    return render_template('results.html', score=round(score, 2), job_role=job_role, filename=uploaded_file.filename)

if __name__ == '__main__':
    app.run(debug=True)
