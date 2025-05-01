import os
import fitz  # PyMuPDF
import spacy
import numpy as np
import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict

# Load spaCy for preprocessing
nlp = spacy.load("en_core_web_sm")

# Load Sentence-BERT for semantic similarity
model = SentenceTransformer('all-MiniLM-L6-v2')

# Function to extract text from PDF (PyMuPDF for better accuracy)
def extract_text_from_pdf(pdf_file):
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text

# Function to preprocess text (lemmatization, stopword removal)
def preprocess(text):
    doc = nlp(text)
    tokens = [
        token.lemma_.lower() 
        for token in doc 
        if not token.is_stop and not token.is_punct
    ]
    return " ".join(tokens)

# Function to extract key sections (Skills, Experience)
def extract_sections(text):
    sections = defaultdict(str)
    current_section = None
    for line in text.split("\n"):
        line = line.strip()
        if line.endswith(":"):
            current_section = line.lower().replace(":", "")
        elif current_section:
            sections[current_section] += line + " "
    return sections

# Streamlit UI
st.title("📄 Resume-Job Alignment Tool")
st.markdown("Upload job description and resumes to find the best matches.")

# Job description input
job_description = st.text_area(
    "✏️ **Paste Job Description**", 
    height=200,
    placeholder="Enter job description here..."
)

# Resume uploader
uploaded_resumes = st.file_uploader(
    "📤 **Upload Resumes (PDF)**", 
    type=["pdf"], 
    accept_multiple_files=True
)

if job_description and uploaded_resumes:
    st.success(f"✅ {len(uploaded_resumes)} resumes uploaded. Processing...")

    # Preprocess job description
    job_desc_processed = preprocess(job_description)
    job_desc_embedding = model.encode([job_desc_processed])

    resume_data = []
    
    for resume in uploaded_resumes:
        # Extract and preprocess text
        raw_text = extract_text_from_pdf(resume)
        processed_text = preprocess(raw_text)
        
        # Extract sections (Skills, Experience get higher weight)
        sections = extract_sections(raw_text)
        skills_exp = f"{sections.get('skills', '')} {sections.get('experience', '')}"
        
        # Generate embeddings
        general_embedding = model.encode([processed_text])[0]
        skills_embedding = model.encode([preprocess(skills_exp)])[0]
        
        # Combined score (60% skills/experience, 40% general match)
        similarity = 0.6 * cosine_similarity(
            [job_desc_embedding[0]], 
            [skills_embedding]
        )[0][0] + 0.4 * cosine_similarity(
            [job_desc_embedding[0]], 
            [general_embedding]
        )[0][0]
        
        # Normalize to 0-100 scale
        normalized_score = int(similarity * 100)
        
        resume_data.append({
            "name": resume.name,
            "score": normalized_score,
            "text": raw_text,
            "file": resume
        })

    # Sort by score (descending)
    ranked_resumes = sorted(resume_data, key=lambda x: x["score"], reverse=True)

    # Display top 3 results
    st.subheader("🏆 Top Matching Resumes")
    for idx, resume in enumerate(ranked_resumes[:3], 1):
        st.markdown(f"### #{idx}: **{resume['name']}** (Score: {resume['score']}/100)")
        
        # Show score breakdown
        with st.expander("📊 **Match Breakdown**"):
            st.markdown(f"```\n{resume['text'][:500]}...\n```")
        
        # Download button
        st.download_button(
            label="⬇️ Download Resume",
            data=resume["file"],
            file_name=resume["name"],
            mime="application/pdf"
        )

else:
    st.warning("⚠️ Please upload a job description and at least one resume.")
