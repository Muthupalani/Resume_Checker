from flask import Flask, request, render_template
import os
from werkzeug.utils import secure_filename
from scripts.preprocessing import preprocess_text
from scripts.model import rank_resumes

UPLOAD_FOLDER = "data/resumes"
ALLOWED_EXTENSIONS = {'txt', 'pdf'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Function to check allowed file extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload():
    job_description = request.form.get("job_description", "").strip()
    if isinstance(job_description, list):
        job_description = " ".join(job_description)

    uploaded_files = request.files.getlist("resumes")

    if not job_description or not uploaded_files:
        return "Please provide both job description and resumes."

    resumes = []
    for file in uploaded_files:
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            # Read and preprocess the resume content
            if filename.endswith(".txt"):
                with open(file_path, "r", encoding="utf-8") as f:
                    resumes.append(f.read())
            elif filename.endswith(".pdf"):
                import PyPDF2
                with open(file_path, "rb") as f:
                    reader = PyPDF2.PdfReader(f)
                    content = " ".join(page.extract_text() for page in reader.pages)
                    resumes.append(content)

    # Debugging: Print the types and values of inputs to `rank_resumes`
    print(f"Job Description Type: {type(job_description)}, Content: {job_description[:100]}")
    print(f"Resumes Type: {type(resumes)}, Length: {len(resumes)}")

    # Rank resumes based on the job description
    try:
        ranked_resumes = rank_resumes(job_description, resumes)
    except Exception as e:
        return f"An error occurred while ranking resumes: {str(e)}"

    # Pass the results to the results page
    return render_template("results.html", ranked_resumes=ranked_resumes)


if __name__ == "__main__":
    # Ensure upload folder exists
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    app.run(debug=True)
