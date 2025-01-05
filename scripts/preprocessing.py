import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Initialize stopwords and lemmatizer
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    """
    Preprocess the input text.
    :param text: Raw text string
    :return: Cleaned and preprocessed text string
    """
    if not text:
        return ""  # Return an empty string if the input text is empty
    
    # Ensure the text is a string
    if isinstance(text, list):
        text = " ".join(text)  # If the input is a list, join it into a single string
    
    # Lowercase the text
    text = text.lower()
    
    # Remove special characters and numbers
    text = re.sub(r'[^a-z\s]', '', text)
    
    # Tokenize the text
    tokens = word_tokenize(text)
    
    # Remove stopwords and lemmatize
    cleaned_tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    
    # Join tokens back to a single string
    return ' '.join(cleaned_tokens)

# Example usage
if __name__ == "__main__":
    sample_text = "Experienced Data Scientist with 5+ years of experience in Python, SQL, and Machine Learning!"
    processed_text = preprocess_text(sample_text)
    print(processed_text)
