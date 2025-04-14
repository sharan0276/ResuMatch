from flask import Flask, render_template, request
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import PyPDF2
import io
import re
from model import ResumeJobPredictor

app = Flask(__name__)

# Initialize and load the model
try:
    predictor = ResumeJobPredictor()
    predictor.load_model()
    print("✅ Model loaded successfully")
except Exception as e:
    print("❌ Error loading model:", e)
    predictor = None

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    return text

def read_pdf(file):
    pdf_reader = PyPDF2.PdfReader(io.BytesIO(file.read()))
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def calculate_similarity(resume_text, job_description):
    resume_text = preprocess_text(resume_text)
    job_description = preprocess_text(job_description)
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([resume_text, job_description])
    similarity = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]
    return round(similarity * 100, 2)

@app.route('/', methods=['GET', 'POST'])
def index():
    match_percentage = None
    job_recommendations = None
    error = None

    if request.method == 'POST':
        try:
            resume_file = request.files['resume']

            if not resume_file.filename.endswith('.pdf'):
                raise ValueError("Please upload a PDF file")

            resume_text = read_pdf(resume_file)
            job_description = request.form.get('job_description', '').strip()

            if job_description:
                match_percentage = calculate_similarity(resume_text, job_description)

            if predictor:
                job_recommendations = predictor.predict(resume_text)
                print("🔍 First recommendation:", job_recommendations[0])
            else:
                error = "Model not loaded."

        except Exception as e:
            error = str(e)

    return render_template('index.html',
                           match_percentage=match_percentage,
                           job_recommendations=job_recommendations,
                           error=error)

if __name__ == '__main__':
    app.run(debug=True)
