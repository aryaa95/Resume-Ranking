import streamlit as st
from PyPDF2 import PdfReader
import pandas as pd
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

# Set Page Config (Wide Mode)
st.set_page_config(page_title="AI Resume Screening", layout="wide")

# 🔹 Theme Toggle Button
dark_mode = st.sidebar.checkbox("🌙 Dark Mode")

# 🔹 Custom CSS for Light & Dark Mode
if dark_mode:
    bg_color = "#181818"
    text_color = "#ffffff"
    card_bg = "#282828"
else:
    bg_color = "#ffffff"
    text_color = "#000000"
    card_bg = "#f9f9f9"

st.markdown(f"""
    <style>
        .stApp {{
            background-color: {bg_color};
        }}

        .title {{
            text-align: center;
            font-size: 42px;
            font-weight: bold;
            color: {text_color};
            text-shadow: 2px 2px 8px rgba(0, 0, 0, 0.5);
        }}

        .main-container {{
            background: rgba(255, 255, 255, 0.9);
            padding: 20px;
            border-radius: 12px;
            box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.2);
        }}

        .stButton>button {{
            background-color: #007bff;
            color: white;
            font-size: 18px;
            font-weight: bold;
            border-radius: 8px;
            padding: 10px;
        }}

        .card {{
            background-color: {card_bg};
            padding: 15px;
            border-radius: 10px;
            margin-bottom: 10px;
            box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1);
            color: {text_color};
        }}

        .stTextArea textarea {{
            font-size: 16px !important;
        }}
    </style>
""", unsafe_allow_html=True)

# 🔹 Main UI
st.markdown('<p class="title">📄 AI Resume Screening & Ranking System</p>', unsafe_allow_html=True)

st.markdown('<div class="main-container">', unsafe_allow_html=True)

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("🔹 Enter Job Description:")
    job_description = st.text_area("Describe the ideal candidate for the role", height=150)

with col2:
    st.subheader("📂 Upload Resumes (PDF only)")
    uploaded_files = st.file_uploader("Upload multiple resumes", type=["pdf"], accept_multiple_files=True)

st.subheader("🖼️ Upload Candidate Profile Images (Optional)")
uploaded_images = st.file_uploader("Upload profile pictures (JPG, PNG)", type=["jpg", "png"], accept_multiple_files=True)

st.markdown('</div>', unsafe_allow_html=True)

# 🔹 Function to Extract Text from PDFs
def extract_text_from_pdf(file):
    pdf = PdfReader(file)
    text = ""
    for page in pdf.pages:
        extracted = page.extract_text()
        if extracted:
            text += extracted
    return text

# 🔹 Function to Rank Resumes
def rank_resumes(job_desc, resumes):
    documents = [job_desc] + resumes
    vectorizer = TfidfVectorizer().fit_transform(documents)
    vectors = vectorizer.toarray()

    job_vector = vectors[0]
    resume_vectors = vectors[1:]
    scores = cosine_similarity([job_vector], resume_vectors).flatten()

    return scores

# 🔹 Processing & Ranking
if st.button("🔍 Rank Candidates"):
    if job_description and uploaded_files:
        with st.spinner("Processing resumes..."):
            time.sleep(2)
            resumes_text = [extract_text_from_pdf(file) for file in uploaded_files]
            scores = rank_resumes(job_description, resumes_text)

            results = pd.DataFrame({
                "Candidate": [file.name for file in uploaded_files],
                "Match Score": scores
            }).sort_values(by="Match Score", ascending=False)

            st.markdown('<div class="main-container">', unsafe_allow_html=True)
            st.subheader("🏆 Ranking Results")

            # 🔹 Display Candidate Details with Profile Pictures
            for i, row in results.iterrows():
                profile_image = None
                if uploaded_images and i < len(uploaded_images):
                    profile_image = uploaded_images[i]

                with st.container():
                    col1, col2 = st.columns([1, 4])
                    with col1:
                        if profile_image:
                            st.image(profile_image, width=80)
                    with col2:
                        st.markdown(f"""
                            <div class="card">
                                <h4 style="color: {text_color};">🎯 {row['Candidate']}</h4>
                                <p style="color: {text_color};"><strong>Match Score:</strong> {row['Match Score']:.2f}</p>
                            </div>
                        """, unsafe_allow_html=True)

            # 🔹 Graph for Visual Score Comparison
            st.subheader("📊 Match Score Visualization")
            fig, ax = plt.subplots()
            ax.barh(results["Candidate"], results["Match Score"], color="skyblue")
            ax.set_xlabel("Match Score")
            ax.set_title("Candidate Match Comparison")
            ax.invert_yaxis()
            st.pyplot(fig)

            st.download_button("📥 Download Results (CSV)", results.to_csv(index=False), "resume_ranking_results.csv", "text/csv")

            st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.warning("⚠️ Please enter a job description and upload resumes.")
