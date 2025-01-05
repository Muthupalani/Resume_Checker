from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

def extract_features(resumes, job_descriptions):
    """
    Convert resumes and job descriptions into TF-IDF feature vectors.
    :param resumes: List of preprocessed resume texts
    :param job_descriptions: List of preprocessed job description texts
    :return: TF-IDF feature matrices for resumes and job descriptions
    """
    # Combine all texts for vectorization
    all_texts = resumes + job_descriptions
    
    # Initialize TF-IDF Vectorizer
    vectorizer = TfidfVectorizer(max_features=500)  # Use top 500 features
    tfidf_matrix = vectorizer.fit_transform(all_texts)
    
    # Split back into resume and job description matrices
    resume_features = tfidf_matrix[:len(resumes), :]
    job_description_features = tfidf_matrix[len(resumes):, :]
    
    return resume_features, job_description_features, vectorizer

# Example usage
if __name__ == "__main__":
    resumes = ["data scientist python machine learning", "software engineer java sql"]
    job_descriptions = ["looking for data scientist with python skills", "hiring software engineer experienced in java"]
    
    resume_features, job_features, vectorizer = extract_features(resumes, job_descriptions)
    
    print("TF-IDF Feature Matrix (Resumes):")
    print(pd.DataFrame(resume_features.toarray(), columns=vectorizer.get_feature_names_out()))
