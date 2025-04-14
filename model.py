import pandas as pd
import numpy as np
import os
import pickle
import re
import ast
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV

# Dummy fallback skills for display/demo
dummy_skills = {
    'software engineer': {
        'matched': ['Python', 'Git', 'REST APIs'],
        'missing': ['Docker', 'CI/CD', 'System Design']
    },
    'frontend developer': {
        'matched': ['React', 'JavaScript', 'CSS'],
        'missing': ['TypeScript', 'Next.js', 'Unit Testing']
    },
    'backend developer': {
        'matched': ['Node.js', 'Express', 'MongoDB'],
        'missing': ['Redis', 'Microservices', 'Authentication']
    },
    'data scientist': {
        'matched': ['Data Cleaning', 'EDA', 'Jupyter'],
        'missing': ['Feature Engineering', 'A/B Testing', 'SQL Optimization']
    },
    'machine learning engineer': {
        'matched': ['Scikit-learn', 'Pandas', 'Linear Regression'],
        'missing': ['XGBoost', 'Model Deployment', 'Pipeline Automation']
    },
    'data analyst': {
        'matched': ['Excel', 'Tableau', 'SQL'],
        'missing': ['Power BI', 'Looker', 'Data Warehousing']
    },
    'devops engineer': {
        'matched': ['Linux', 'Shell Scripting', 'AWS'],
        'missing': ['Terraform', 'Kubernetes', 'Prometheus']
    },
    'ai engineer': {
        'matched': ['TensorFlow', 'Keras', 'Neural Networks'],
        'missing': ['Prompt Engineering', 'LangChain', 'GPU Acceleration']
    },
    'intern': {
        'matched': ['Python', 'Basic Git', 'OOP'],
        'missing': ['Project Experience', 'Production Code', 'Deployment Tools']
    }
}

class ResumeJobPredictor:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=8000, stop_words='english', ngram_range=(1, 2), sublinear_tf=True)
        base_lr = LogisticRegression(max_iter=1000, C=10, class_weight='balanced')
        self.classifier = OneVsRestClassifier(CalibratedClassifierCV(base_lr, cv=3))
        self.mlb = MultiLabelBinarizer()

    def preprocess_text(self, text):
        if isinstance(text, list):
            text = ' '.join(text)
        if not isinstance(text, str):
            return ''
        return re.sub(r'[^\w\s]', '', text.lower().strip())

    def combine_resume_fields(self, row):
        fields = [
            row.get('skills', []),
            row.get('major_field_of_studies', []),
            row.get('related_skils_in_job', []),
            row.get('responsibilities', ''),
            row.get('company_expectations', ''),
            row.get('skills_required', ''),
            row.get('career_objective', '')
        ]
        return ' '.join([self.preprocess_text(field) for field in fields if field])

    def normalize_title(self, title):
        title = title.lower().strip()
        title = re.sub(r'(senior|junior|lead|entry level)', '', title)
        return title.strip()

    def get_job_titles(self, row):
        titles = set()

        def parse_list_field(field):
            if pd.isna(field):
                return []
            if isinstance(field, list):
                return field
            try:
                parsed = ast.literal_eval(field)
                if isinstance(parsed, list):
                    return parsed
            except:
                pass
            return [field]

        job_fields = parse_list_field(row.get('job_position_name')) + parse_list_field(row.get('positions'))
        for title in job_fields:
            titles.add(self.normalize_title(title))

        return list(titles)

    def train(self, resume_data_path):
        df = pd.read_csv(resume_data_path)
        df['resume_text'] = df.apply(self.combine_resume_fields, axis=1)
        job_title_lists = df.apply(self.get_job_titles, axis=1)
        job_titles_flat = [[str(t).strip() for t in titles] for titles in job_title_lists]
        df['job_titles'] = job_titles_flat

        X = self.vectorizer.fit_transform(df['resume_text'])
        y = self.mlb.fit_transform(df['job_titles'])

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.classifier.fit(X_train, y_train)

        print(f"Validation Accuracy: {self.classifier.score(X_test, y_test):.2f}")

    def get_missing_and_matched_keywords(self, resume_text, job_title):
        job_title = job_title.lower().strip()
        resume_tokens = set(self.preprocess_text(resume_text).split())

        try:
            class_map = {cls.lower(): idx for idx, cls in enumerate(self.mlb.classes_)}
            matched_index = class_map.get(job_title)
            if matched_index is None:
                return [], []
        except Exception as e:
            print("Matching job title failed:", e)
            return [], []

        try:
            calibrated_model = self.classifier.estimators_[matched_index]
            coef = calibrated_model.base_estimator.coef_.flatten() if hasattr(calibrated_model, 'base_estimator') else calibrated_model.coef_.flatten()
            feature_array = np.array(self.vectorizer.get_feature_names_out())
            top_features = feature_array[np.argsort(coef)[-50:]]
            missing = [word for word in top_features if word not in resume_tokens][:5]
            matched = [word for word in top_features if word in resume_tokens][:5]
            return missing, matched
        except Exception as e:
            print("Model coefficient access failed:", e)
            return [], []

    def predict(self, resume_text):
        processed_text = self.preprocess_text(resume_text)
        X = self.vectorizer.transform([processed_text])
        pred_probs = self.classifier.predict_proba(X)

        top_indices = np.argsort(pred_probs[0])[-10:][::-1]
        max_prob = np.max(pred_probs[0])
        recommendations = []

        for idx in top_indices:
            prob = pred_probs[0][idx]
            norm_prob = prob / max_prob if max_prob > 0 else 0
            raw_label = str(self.mlb.classes_[idx]).strip("[]'")
            job_title = raw_label.title()

            # Get keyword comparison
            missing_keywords, matched_keywords = self.get_missing_and_matched_keywords(resume_text, raw_label.lower())

            # Fallback to dummy if empty
            job_key = raw_label.lower().strip()
            if job_key in dummy_skills:
                if not matched_keywords:
                    matched_keywords = dummy_skills[job_key]['matched']
                if not missing_keywords:
                    missing_keywords = dummy_skills[job_key]['missing']
            else:
                if not matched_keywords:
                    matched_keywords = ['Python', 'Teamwork']
                if not missing_keywords:
                    missing_keywords = ['Deployment', 'Documentation']

            recommendations.append({
                'job_title': job_title,
                'match_percentage': round(norm_prob * 100, 2),
                'missing_keywords': missing_keywords,
                'matched_keywords': matched_keywords
            })

        return recommendations if recommendations else [{
            'job_title': 'No strong match found',
            'match_percentage': 0.0,
            'missing_keywords': [],
            'matched_keywords': []
        }]

    def save_model(self):
        os.makedirs('models', exist_ok=True)
        with open('models/vectorizer.pkl', 'wb') as f:
            pickle.dump(self.vectorizer, f)
        with open('models/classifier.pkl', 'wb') as f:
            pickle.dump(self.classifier, f)
        with open('models/mlb.pkl', 'wb') as f:
            pickle.dump(self.mlb, f)

    def load_model(self):
        with open('models/vectorizer.pkl', 'rb') as f:
            self.vectorizer = pickle.load(f)
        with open('models/classifier.pkl', 'rb') as f:
            self.classifier = pickle.load(f)
        with open('models/mlb.pkl', 'rb') as f:
            self.mlb = pickle.load(f)
