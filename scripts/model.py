from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd
from scripts.feature_extraction import extract_features
from scripts.preprocessing import preprocess_text


def rank_resumes(resumes, job_description):
    """
    Rank resumes based on their similarity to the job description.
    :param resumes: List of raw resume texts
    :param job_description: Raw job description text
    :return: Ranked list of resumes with similarity scores
    """
    # Preprocess all texts
    preprocessed_resumes = [preprocess_text(resume) for resume in resumes]
    preprocessed_job_description = preprocess_text(job_description)
    
    # Extract features using TF-IDF
    resume_features, job_features, vectorizer = extract_features(preprocessed_resumes, [preprocessed_job_description])
    
    # Compute cosine similarity
    similarity_scores = cosine_similarity(resume_features, job_features).flatten()
    
    # Rank resumes based on similarity scores
    ranked_indices = np.argsort(similarity_scores)[::-1]
    ranked_resumes = [(resumes[i], similarity_scores[i]) for i in ranked_indices]
    
    return ranked_resumes

# Example usage
if __name__ == "__main__":
    # Example resumes and job description
    resumes = [
        "Experienced data scientist with expertise in Python, SQL, and Machine Learning.",
        "Software engineer skilled in Java, C++, and database management.",
        "Machine learning enthusiast with hands-on experience in TensorFlow and scikit-learn."
    ]
    job_description = "Looking for a data scientist proficient in Python and SQL with experience in machine learning."
    
    ranked_resumes = rank_resumes(resumes, job_description)
    
    print("Ranked Resumes:")
    for rank, (resume, score) in enumerate(ranked_resumes, start=1):
        print(f"{rank}. Resume: {resume}\n   Score: {score:.2f}")
