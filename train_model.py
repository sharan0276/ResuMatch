from model import ResumeJobPredictor
import pandas as pd

def train():
    print("Starting model training...")
    
    predictor = ResumeJobPredictor()
    
    # Load data and print some information
    df = pd.read_csv('updated_resume_data.csv')
    print(f"Loaded {len(df)} resume samples")
    
    # Print unique job positions
    job_positions = set()
    for pos in df['job_position_name'].dropna():
        job_positions.add(pos.strip())
    for pos in df['positions'].dropna():
        job_positions.update([p.strip() for p in pos.split(',')])
    
    print("\nUnique job positions found:")
    for pos in sorted(job_positions):
        print(f"- {pos}")
    
    # Train the model
    print("\nTraining model...")
    predictor.train('updated_resume_data.csv')
    
    # Save the model
    print("Saving model...")
    predictor.save_model()
    
    print("\nTraining completed successfully!")
    
    # Test the model with a sample resume
    print("\nTesting model with a sample resume...")
    sample_resume = """
    Experienced Python developer with expertise in Flask, Django, and machine learning.
    Skills: Python, SQL, Machine Learning, Git
    Education: Computer Science
    Experience: Led development teams, implemented ML solutions
    """
    
    recommendations = predictor.predict(sample_resume)
    print("\nSample predictions:")
    for rec in recommendations:
        print(f"- {rec['job_title']}: {rec['match_percentage']}%")

if __name__ == '__main__':
    train() 