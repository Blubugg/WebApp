# Import Library
import streamlit as st
import sklearn
import joblib, os
import numpy as np
import pandas as pd
from PIL import Image
import base64
from io import BytesIO

# Loading Models
model = joblib.load('./models/Final.pkl')

# Function to convert image to base64
def image_to_base64(img):
    buffered = BytesIO()
    img.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

def main():
    #st.caption("based on Experience Level, Employment Type, Job Title, Company Location, Company Size")
    activity = ["Determine Salary", "Salary Dataset", "Visualization", "About", "Profile"]
    choice = st.sidebar.selectbox("Menu", activity)
    
    # Determine Salary Choice
    if choice == 'Determine Salary':
        st.markdown("<h1 style='text-align: center;'>Determine Employee Salaries</h1>", unsafe_allow_html=True)
        experience_level = st.selectbox('Experience Level', ('Entry-Level', 'Experienced', 'Mid-Level', 'Senior'))
        employment_type = st.selectbox('Employment Type', ('Contractor', 'Freelancer', 'Full-Time', 'Part-Time'))
        all_job_titles = [
            '3D Computer Vision Researcher', 'AI Developer', 'AI Programmer', 
            'AI Scientist', 'Analytics Engineer', 'Applied Data Scientist',
            'Applied Machine Learning Engineer', 'Applied Machine Learning Scientist',
            'Applied Scientist', 'Autonomous Vehicle Technician', 'Azure Data Engineer',
            'BI Analyst', 'BI Data Analyst', 'BI Data Engineer', 'BI Developer',
            'Big Data Architect', 'Big Data Engineer', 'Business Data Analyst',
            'Business Intelligence Engineer', 'Cloud Data Architect', 'Cloud Data Engineer',
            'Cloud Database Engineer', 'Compliance Data Analyst', 'Computer Vision Engineer',
            'Computer Vision Software Engineer', 'Data Analyst', 'Data Analytics Consultant',
            'Data Analytics Engineer', 'Data Analytics Lead', 'Data Analytics Manager', 
            'Data Analytics Specialist', 'Data Architect', 'Data DevOps Engineer', 
            'Data Engineer', 'Data Infrastructure Engineer', 'Data Lead', 
            'Data Management Specialist', 'Data Manager', 'Data Modeler',
            'Data Operations Analyst', 'Data Operations Engineer', 'Data Quality Analyst',
            'Data Science Consultant', 'Data Science Engineer', 'Data Science Lead',
            'Data Science Manager', 'Data Science Tech Lead', 'Data Scientist', 
            'Data Scientist Lead', 'Data Specialist', 'Data Strategist', 
            'Deep Learning Engineer', 'Deep Learning Researcher', 'Director of Data Science', 
            'ETL Developer', 'ETL Engineer', 'Finance Data Analyst', 'Financial Data Analyst',
            'Head of Data', 'Head of Data Science', 'Head of Machine Learning', 
            'Insight Analyst', 'Lead Data Analyst', 'Lead Data Engineer', 'Lead Data Scientist',
            'Lead Machine Learning Engineer', 'Machine Learning Developer', 
            'Machine Learning Engineer', 'Machine Learning Infrastructure Engineer',
            'Machine Learning Manager', 'Machine Learning Research Engineer',
            'Machine Learning Researcher', 'Machine Learning Scientist',
            'Machine Learning Software Engineer', 'Manager Data Management', 
            'Marketing Data Analyst', 'Marketing Data Engineer', 'ML Engineer', 'MLOps Engineer',
            'NLP Engineer', 'Power BI Developer', 'Principal Data Analyst', 
            'Principal Data Architect', 'Principal Data Engineer', 'Principal Data Scientist',
            'Principal Machine Learning Engineer', 'Product Data Analyst', 'Product Data Scientist',
            'Research Engineer', 'Research Scientist', 'Software Data Engineer',
            'Staff Data Analyst', 'Staff Data Scientist' 
        ]
        job_title = st.selectbox('Job Title', all_job_titles)
        all_company_location = [
            'United Arab Emirates', 'Albania', 'Armenia', 'Argentina', 
            'American Samoa', 'Austria', 'Australia', 'Bosnia Herzegovina', 
            'Belgium', 'Bolivia', 'Brazil', 'Bahama', 'California', 'Chad', 
            'Switzerland', 'Chile', 'China', 'Colombia', 'Costa Rica', 
            'Czech Republic', 'Germany', 'Denmark', 'Algeria', 
            'Republic of Estonia', 'Egypt', 'Spain', 'Finlandia', 'France', 
            'United Kingdom', 'Ghana', 'Greece', 'Hong Kong', 'Honduras', 
            'Croatia', 'Hongaria', 'Indonesia', 'Côte d Ivoire', 'Illinois', 
            'India', 'Iraq', 'Iran', 'Italy', 'Japan', 'Kenya', 'Lithuania', 
            'Luxemburg', 'Latvia', 'Maroko', 'Maryland', 
            'Republic of North Macedonia', 'Malta', 'Mexico', 'Malaysia', 
            'Nigeria', 'Netherlands', 'New Zealand', 'Phillippines', 
            'Pakistan', 'Poland', 'Puerto Riko', 'Portugal', 'Romania', 
            'Russia', 'Sweden', 'Singapore', 'Slovenia', 'Slovakia', 
            'Thailand', 'Turkey', 'Ukraine', 'United States', 'Vietnam'
        ]
        company_location = st.selectbox('Company Location', all_company_location)
        company_size = st.selectbox('Company Size', ('Large', 'Medium', 'Small'))
        predict_button = st.button("Predict Salary")

        # Display Salary Prediction Result
        if predict_button:
            experience_mapping = {
                'Entry-Level': 0, 
                'Experienced': 1,
                'Mid-Level': 2, 
                'Senior': 3
            }
            employment_mapping = {
                'Contractor': 0, 
                'Freelancer': 1, 
                'Full-Time': 2, 
                'Part-Time': 3
            }
            job_mapping = {title: idx for idx, title in enumerate(all_job_titles)}
            location_mapping = {title: idx for idx, title in enumerate(all_company_location)}
            size_mapping = {
                'Large': 0, 
                'Medium': 1, 
                'Small': 2
            }

            experience_index = experience_mapping[experience_level]
            employment_index = employment_mapping[employment_type]
            job_index = job_mapping[job_title]
            location_index = location_mapping[company_location]
            size_index = size_mapping[company_size]

            input_data = pd.DataFrame({
                'work_year': 2023,
                'experience_level': [experience_index],
                'employment_type': [employment_index],
                'job_title': [job_index],
                'company_location': [location_index],
                'company_size': [size_index]
            })

            predicted_salary = model.predict(input_data)

            st.write(f"Predicted Salary for {job_title} with {employment_type} and {experience_level} experience at {company_location} with {company_size} is: ${predicted_salary[0]:.2f}")
    
    # Salary Dataset Choice
    if choice == 'Salary Dataset':
        df = pd.read_csv('./Data/FinalProject/Data Science Salary 2021 to 2023.csv')

        st.markdown("<h1 style='text-align: center;'>Data Science Salary 2021 to 2023</h1>", unsafe_allow_html=True)
        st.dataframe(df)

    # Result Choice
    if choice == 'Visualization':
        st.markdown("<h1 style='text-align: center;'>Correlation</h1>", unsafe_allow_html=True)
        
        # Load Image
        imgcorr1 = Image.open('./src/corr1.png')
        imgcorr2 = Image.open('./src/corr2.png')
        imgcorr3 = Image.open('./src/corr3.png')
        imgcorr4 = Image.open('./src/corr4.png')
        imgcorr5 = Image.open('./src/corr5.png')

        # Display Image
        st.image(imgcorr1, caption = 'Correlation Matrix', use_column_width = True)
        
        # Create Two Columns Layout
        col1, col2 = st.columns(2)

        # Display Image in Each Column
        with col1:
            st.image(imgcorr2, caption = 'Distribution of Salary in USD', use_column_width = True)

        with col2:
            st.image(imgcorr5, caption = 'Pair Plot of Numerical Features', use_column_width = True)

        # Create Two Columns Layout
        col3, col4 = st.columns(2)

        # Display Image in Each Column
        with col3:
            st.image(imgcorr3, caption = 'Experience Level vs Salary in USD', use_column_width = True)
        
        with col4:
            st.image(imgcorr4, caption = 'Employment Type vs Salary in USD', use_column_width = True)

        st.markdown("<h1 style='text-align: center;'>Result</h1>", unsafe_allow_html=True)

        # Load Image
        imgres1 = Image.open('./src/result1.png')
        imgres2 = Image.open('./src/result2.png')
        imgres3 = Image.open('./src/result3.png')

        # Display Image
        st.image(imgres1, use_column_width = True)
        st.image(imgres2, use_column_width = True)
        st.image(imgres3, use_column_width = True)
    
    # About Choice
    if choice == 'About':
        st.title("About Us")
        
        # Load Text with Style Justify
        text = """
        <div style="text-align: justify;">
        <p>Welcome to the salary prediction platform designed exclusively for Data Science professionals. We were founded with the aim of providing valuable insights to data experts to help them make informed decisions regarding their salaries and career development.</p>
        <p>Our mission is to provide accurate, transparent, and up-to-date information about salaries in the field of Data Science. We believe that having a good understanding of salary structures can empower Data Science professionals to negotiate with confidence, plan their career development, and achieve long-term success.</p>
        </div>
        """

        # Display Text
        st.markdown(text, unsafe_allow_html=True)

    # Profile Choice
    if choice == 'Profile':
        st.title("Profile")

        # Load Image
        profil = Image.open('./src/profil.jpg')

        
        # Display Image
        st.markdown(
        f'<div style="display: flex; justify-content: center;">'
        f'<img src="data:image/jpeg;base64,{image_to_base64(profil)}" alt="Your Profile Image" style="width: 200px; height: auto;">'
        f'</div>',
        unsafe_allow_html=True
        )

        # Load Text with Style Justify
        text = """
        <div style="text-align: center;">
        <p></p>
        <p>Valentino Banyu Biru</p>
        <p>21537141023</p>
        <p>Information Technology</p>
        </div>
        """

        # Display Text
        st.markdown(text, unsafe_allow_html=True)


if __name__ == '__main__':
    main()