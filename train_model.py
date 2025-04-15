from model import ResumeJobPredictor
import pandas as pd
import ast

def clean_and_prepare_data(filepath):
    print(f"📂 Loading data from {filepath}...")
    df = pd.read_csv(filepath)

    # Parse stringified lists
    for col in ['positions', 'skills_required', 'combined_skills', 'combined_responsibilities', 'major_field_of_studies']:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: ast.literal_eval(x) if pd.notna(x) else [])

    # Deduplicate
    df['combined_skills'] = df['combined_skills'].apply(lambda x: list(set(x)) if isinstance(x, list) else [])
    df['combined_responsibilities'] = df['combined_responsibilities'].apply(lambda x: list(set(x)) if isinstance(x, list) else [])

    return df

def train_pipeline():
    df = clean_and_prepare_data("updated_resume_data_2.csv")
    print(f"✅ Loaded and cleaned {len(df)} samples.")

    predictor = ResumeJobPredictor()
    predictor.train(df)
    predictor.save_model()
    print("✅ Model trained and saved successfully.")

    sample_resume = """
    Experienced Python developer with expertise in Flask, Django, and machine learning.
    Skills: Python, SQL, Machine Learning, Git
    Education: Computer Science
    Experience: Led development teams, implemented ML solutions
    """
    print("\n🔍 Sample Resume Prediction:")
    predictions = predictor.predict(sample_resume)
    for p in predictions:
        print(f"- {p['job_title']}: {p['match_percentage']}%")

if __name__ == '__main__':
    train_pipeline()
