# ResuMatch: Smart Resume Parser & Job Fit Analyzer

1. Description:
This project aims to develop a website that analyzes a user's resume when they give a job description to it. Unlike conventional resume parsers, our system goes beyond keyword matching by identifying related tools and technologies that align with job requirements. Additionally, it suggests job roles based on the candidate’s skills and experience, helping users discover suitable opportunities.

2. Dataset:
Resume Dataset: 
https://www.kaggle.com/datasets/saugataroyarghya/resume-dataset
Job Descriptions: Collected from job boards like LinkedIn, Glassdoor, and Indeed using web scraping techniques.

3. Methodology and Expected Results:
The ResuMatch system applies a structured machine learning pipeline to analyze resumes and recommend relevant job roles. The process begins with data preprocessing, where raw text from resumes is cleaned and standardized. This involves converting all text to lowercase, removing punctuation, eliminating stop words, and tokenizing the text into individual terms. These steps help reduce noise and ensure consistency across the dataset. Once preprocessed, the textual data is transformed into numerical features using TF-IDF vectorization. TF-IDF captures the importance of words by measuring their frequency relative to the entire corpus, allowing the system to focus on terms that are more informative for classification. These vectorized features are then used as input to the machine learning model. To further evaluate how well a resume aligns with a specific job description, cosine similarity is applied to the TF-IDF vectors of both texts. This measures the angular similarity between the two, producing a match percentage that reflects how closely the resume matches the job requirements. These vectorized features and similarity scores are then used as input to the machine learning model.
Given that a single resume can correspond to multiple job roles, the problem is framed as a multi-label classification task. To handle this, job titles are encoded using a MultiLabelBinarizer, which converts the target labels into binary vectors. We use a One-vs-Rest logistic regression model for classification, training a separate binary classifier for each job title. This model is well-suited for high-dimensional text data and supports multi-label prediction efficiently. Additionally, we use CalibratedClassifierCV to calibrate the model’s output probabilities, ensuring that the confidence scores for each predicted job role are reliable and interpretable. To enhance the system’s feedback mechanism, we implemented a skill matching module that provides insights into matched and missing skills for each predicted job title. For each role, we maintain a predefined list of key skills. The system checks which of these skills are present in the resume, highlighting them as “matched,” while the absent ones are listed as “missing.” This offers valuable guidance for job seekers looking to tailor their resumes for specific roles.
The model’s performance is evaluated using standard multi-label metrics such as accuracy, Hamming loss, precision, recall, and F1-score. Our experiments showed a marked improvement in performance after applying preprocessing and feature engineering, with accuracy improving from approximately 86% to 95%. 
The website connects to a back-end built with Flask, which calculates match scores and suggests improvements for resumes. It provides users with real-time insights to enhance their resumes as they apply for jobs, ensuring better alignment with job requirements.


How to run the project:

1. source venv/bin/activate
2. pip install -r requirements.txt
3. python train_model.py
4. python app.py
