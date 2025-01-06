from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pickle

# Sample resumes and job descriptions
resumes = [
    "Experienced software engineer with knowledge of Python, Flask, and Django.",
    "Data scientist skilled in machine learning, TensorFlow, and data analysis.",
    "Marketing professional with expertise in digital marketing and SEO strategies.",
    "Graphic designer proficient in Adobe Photoshop, Illustrator, and UI/UX design.",
    "Project manager with experience in agile methodologies and team leadership.",
]

job_description = [
    "Looking for a data scientist with expertise in machine learning and TensorFlow.",
    "Hiring a software engineer with Python, Flask, and API development skills.",
    "Seeking a marketing professional for digital campaigns and SEO.",
    "Need a graphic designer for UI/UX and visual branding projects.",
    "Project manager required for agile-based project delivery and team management.",
]

# Labels: 1 = relevant, 0 = not relevant
labels = [0, 1, 0, 0, 1]

# Combine resumes and job descriptions for vectorization
all_text = resumes + job_description

# Vectorize text using TF-IDF
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(all_text)

# Use only resumes for training
X_resumes = X[:len(resumes)]

# Stratified splitting to maintain class distribution
X_train, X_test, y_train, y_test = train_test_split(
    X_resumes, labels, test_size=0.4, random_state=42, stratify=labels
)

# Check if stratified splitting is successful
print("Training class distribution:", {label: y_train.count(label) for label in set(y_train)})
print("Testing class distribution:", {label: y_test.count(label) for label in set(y_test)})

# Train logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Test the model
y_pred = model.predict(X_test)
print("Model Accuracy:", accuracy_score(y_test, y_pred))

# Save vectorizer and model to files
with open("models/tfidf_vectorizer.pkl", "wb") as vec_file:
    pickle.dump(vectorizer, vec_file)

with open("models/resume_ranker_model.pkl", "wb") as model_file:
    pickle.dump(model, model_file)

print("Files 'tfidf_vectorizer.pkl' and 'resume_ranker_model.pkl' created successfully!")
