import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Function to generate random dates
def random_date(start_year=2015, end_year=2023):
    start = datetime(start_year, 1, 1)
    end = datetime(end_year, 12, 31)
    return start + timedelta(
        days=np.random.randint(0, (end - start).days)
    )

# Create sample data
sample_data = {
    'address': [
        "123 Tech Street, Silicon Valley, CA",
        "456 Marketing Ave, New York, NY",
        "789 Data Drive, Boston, MA",
        "321 Web Lane, Seattle, WA",
        "654 Project Road, Austin, TX"
    ],
    'career_objective': [
        "Seeking a challenging position as a Python Developer to leverage my expertise in web development and machine learning",
        "Dynamic marketing professional aiming to lead digital marketing initiatives in a growth-oriented organization",
        "Passionate data scientist looking to solve complex business problems through data-driven solutions",
        "Creative frontend developer eager to build engaging user experiences with modern web technologies",
        "Results-driven project manager seeking to lead high-impact technology projects"
    ],
    'skills': [
        "Python, Flask, Django, SQL, Machine Learning, Git",
        "Digital Marketing, Social Media, Content Strategy, SEO, Analytics",
        "Python, R, Machine Learning, Statistical Analysis, SQL, Tableau",
        "React, JavaScript, HTML5, CSS3, Redux, Webpack",
        "Agile, Scrum, JIRA, Risk Management, Stakeholder Management"
    ],
    'educational_institution_name': [
        "Stanford University",
        "Columbia University",
        "MIT",
        "University of Washington",
        "University of Texas"
    ],
    'degree_names': [
        "Bachelor of Science in Computer Science",
        "Master of Business Administration",
        "Master of Science in Data Science",
        "Bachelor of Science in Computer Engineering",
        "Master of Project Management"
    ],
    'passing_years': [
        "2019",
        "2018",
        "2020",
        "2019",
        "2017"
    ],
    'educational_results': [
        "3.8",
        "3.9",
        "3.7",
        "3.6",
        "3.9"
    ],
    'result_types': [
        "GPA",
        "GPA",
        "GPA",
        "GPA",
        "GPA"
    ],
    'major_field_of_studies': [
        "Computer Science",
        "Marketing",
        "Data Science",
        "Computer Engineering",
        "Project Management"
    ],
    'professional_company_names': [
        "Google, Amazon",
        "Facebook, Twitter",
        "Microsoft, IBM",
        "Apple, Netflix",
        "Oracle, Dell"
    ],
    'company_urls': [
        "www.google.com, www.amazon.com",
        "www.facebook.com, www.twitter.com",
        "www.microsoft.com, www.ibm.com",
        "www.apple.com, www.netflix.com",
        "www.oracle.com, www.dell.com"
    ],
    'start_dates': [
        "2019-01-15, 2021-03-01",
        "2018-06-01, 2020-07-15",
        "2020-02-01, 2022-01-15",
        "2019-08-15, 2021-09-01",
        "2017-11-01, 2020-01-15"
    ],
    'end_dates': [
        "2021-02-28, Present",
        "2020-07-14, Present",
        "2021-12-31, Present",
        "2021-08-31, Present",
        "2019-12-31, Present"
    ],
    'related_skils_in_job': [
        "Python, Machine Learning, SQL",
        "Digital Marketing, Social Media, Content Creation",
        "Data Analysis, Python, Machine Learning",
        "React, JavaScript, Frontend Development",
        "Project Management, Agile, Leadership"
    ],
    'positions': [
        "Senior Software Engineer, Lead Developer",
        "Marketing Manager, Digital Marketing Lead",
        "Senior Data Scientist, ML Engineer",
        "Frontend Developer, UI Lead",
        "Project Manager, Program Manager"
    ],
    'locations': [
        "San Francisco, CA",
        "New York, NY",
        "Boston, MA",
        "Seattle, WA",
        "Austin, TX"
    ],
    'responsibilities': [
        "Developed scalable applications, Led machine learning projects",
        "Managed marketing campaigns, Led digital strategy",
        "Built predictive models, Conducted data analysis",
        "Developed user interfaces, Implemented responsive designs",
        "Managed project lifecycle, Led cross-functional teams"
    ],
    'extra_curricular_activity_types': [
        "Hackathons, Open Source",
        "Marketing Competitions, Volunteering",
        "Data Science Competitions, Research",
        "Web Development Workshops, Mentoring",
        "Leadership Programs, Community Service"
    ],
    'languages': [
        "English, Spanish",
        "English, French",
        "English, Mandarin",
        "English, German",
        "English, Hindi"
    ],
    'proficiency_levels': [
        "Native, Intermediate",
        "Native, Advanced",
        "Native, Basic",
        "Native, Intermediate",
        "Native, Advanced"
    ],
    'certification_providers': [
        "AWS, Google Cloud",
        "Google Analytics, HubSpot",
        "IBM, Microsoft",
        "Meta, MongoDB",
        "PMI, Scrum Alliance"
    ],
    'certification_skills': [
        "Cloud Computing, Machine Learning",
        "Digital Marketing, Content Strategy",
        "Data Science, AI",
        "Frontend Development, Database",
        "Project Management, Agile"
    ],
    'job_position_name': [
        "Software Engineer",
        "Marketing Manager",
        "Data Scientist",
        "Frontend Developer",
        "Project Manager"
    ],
    'educationaL_requirements': [
        "Bachelor's in Computer Science or related field",
        "Bachelor's in Marketing or Business",
        "Master's in Data Science, Statistics or related field",
        "Bachelor's in Computer Science or related field",
        "Bachelor's in any field with PMP certification"
    ],
    'experiencere_requirement': [
        "5+ years",
        "3+ years",
        "4+ years",
        "3+ years",
        "5+ years"
    ],
    'age_requirement': [
        "25-35",
        "25-40",
        "25-35",
        "25-35",
        "28-45"
    ],
    'skills_required': [
        "Python, Django, SQL, Machine Learning",
        "Digital Marketing, Social Media, SEO",
        "Python, R, Machine Learning, Statistics",
        "React, JavaScript, HTML, CSS",
        "Project Management, Agile, Leadership"
    ],
    'matched_score': [
        0.85,
        0.78,
        0.92,
        0.88,
        0.82
    ]
}

# Create DataFrame
df = pd.DataFrame(sample_data)

# Add any missing columns with empty values
all_columns = ['address', 'career_objective', 'skills', 'educational_institution_name',
               'degree_names', 'passing_years', 'educational_results', 'result_types',
               'major_field_of_studies', 'professional_company_names', 'company_urls',
               'start_dates', 'end_dates', 'related_skils_in_job', 'positions',
               'locations', 'responsibilities', 'extra_curricular_activity_types',
               'extra_curricular_organization_names', 'extra_curricular_organization_links',
               'role_positions', 'languages', 'proficiency_levels', 'certification_providers',
               'certification_skills', 'online_links', 'issue_dates', 'expiry_dates',
               '﻿job_position_name', 'educationaL_requirements', 'experiencere_requirement',
               'age_requirement', 'responsibilities.1', 'skills_required', 'matched_score']

for col in all_columns:
    if col not in df.columns:
        df[col] = ""

# Save to CSV
df.to_csv('resume_Data.csv', index=False)

print("Sample resume data has been created successfully!") 