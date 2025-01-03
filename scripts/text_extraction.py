import pdfplumber

def extract_text_from_pdf(file_path):
    """
    Extract text from a PDF file.
    :param file_path: Path to the PDF file
    :return: Extracted text as a string
    """
    try:
        with pdfplumber.open(file_path) as pdf:
            text = ''
            for page in pdf.pages:
                text += page.extract_text()
            return text
    except Exception as e:
        print(f"Error reading PDF: {e}")
        return None

# Example usage
if __name__ == "__main__":
    file_path = "C:/Users/muthu/OneDrive/ドキュメント/project/ai_resume_checker/data/resume.pdf"
    extracted_text = extract_text_from_pdf(file_path)
    print(extracted_text)
