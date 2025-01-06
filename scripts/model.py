from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pickle
import numpy as np

# Generate TF-IDF features
def generate_tfidf_features(resumes, job_description):
    vectorizer = TfidfVectorizer(max_features=5000)
    all_text = resumes + [job_description]
    tfidf_matrix = vectorizer.fit_transform(all_text)

    with open("models/tfidf_vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)

    return tfidf_matrix[:-1], tfidf_matrix[-1]

# Train and save the model
def train_model(resumes, job_description, scores):
    X_resumes, X_job = generate_tfidf_features(resumes, job_description)
    model = LogisticRegression()
    model.fit(X_resumes, scores)

    with open("models/resume_ranker_model.pkl", "wb") as f:
        pickle.dump(model, f)

    return model

# Rank resumes
def rank_resumes_with_model(job_description, resumes):
    with open("models/tfidf_vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    with open("models/resume_ranker_model.pkl", "rb") as f:
        model = pickle.load(f)

    resumes_vectorized = vectorizer.transform(resumes)
    scores = model.predict_proba(resumes_vectorized)[:, 1] * 100  # Percentage scores
    ranked_resumes = sorted(zip(resumes, scores), key=lambda x: x[1], reverse=True)

    return ranked_resumes
