import streamlit as st
from pypdf import PdfReader
import ollama

# --------------------------------
# PAGE CONFIG
# --------------------------------
st.set_page_config(
    page_title="AI Resume Intelligence System",
    page_icon="📄",
    layout="centered"
)

# --------------------------------
# TITLE
# --------------------------------
st.title("📄 AI Resume Intelligence System")
st.caption("AI-Powered Resume Analysis, Job Matching & Interview Preparation")

# --------------------------------
# SIDEBAR
# --------------------------------
st.sidebar.title("⚙ Settings")

model_name = st.sidebar.selectbox(
    "Choose AI Model",
    ["gemma:2b", "tinyllama"]
)

# --------------------------------
# FILE UPLOAD
# --------------------------------
uploaded_file = st.file_uploader(
    "Upload Your Resume (PDF)",
    type=["pdf"]
)

resume_text = ""

# --------------------------------
# PDF EXTRACTION
# --------------------------------
if uploaded_file is not None:

    st.success("Resume Uploaded Successfully!")

    try:

        reader = PdfReader(uploaded_file)

        for page in reader.pages:

            text = page.extract_text()

            if text:
                resume_text += text

        # Empty check
        if resume_text.strip() == "":
            st.error("Could not extract text from PDF.")
            st.stop()

        # --------------------------------
        # SHOW RESUME TEXT
        # --------------------------------
        st.subheader("📄 Extracted Resume")

        st.text_area(
            "Resume Content",
            resume_text,
            height=250
        )

        # --------------------------------
        # JOB DESCRIPTION
        # --------------------------------
        st.subheader("💼 Job Matcher")

        job_desc = st.text_area(
            "Paste Job Description Here",
            height=200
        )

        # --------------------------------
        # BUTTONS
        # --------------------------------
        col1, col2, col3 = st.columns(3)

        analyze_btn = col1.button("Analyze Resume")
        match_btn = col2.button("Match Job")
        interview_btn = col3.button("Interview Questions")

        # ==========================================
        # RESUME ANALYZER
        # ==========================================
        if analyze_btn:

            with st.spinner("Analyzing Resume..."):

                prompt = f"""
You are a professional ATS and HR expert.

Analyze this resume carefully.

Return:
1. Resume Score out of 10
2. Key Skills
3. Missing Skills
4. Improvement Suggestions
5. Best Career Roles

Be professional and structured.

Resume:
{resume_text}
"""

                response = ollama.chat(
                    model=model_name,

                    messages=[
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],

                    options={
                        "num_predict": 180,
                        "temperature": 0.3
                    }
                )

                ai_reply = response["message"]["content"]

                st.subheader("📊 Resume Analysis")
                st.success(ai_reply)

        # ==========================================
        # JOB MATCHER
        # ==========================================
        if match_btn:

            if job_desc.strip() == "":
                st.warning("Please paste job description first.")

            else:

                with st.spinner("Matching Resume with Job..."):

                    prompt = f"""
You are a strict ATS system.

Compare the resume with the job description.

Return ONLY:

Match Score:
Match Percentage:
Matching Skills:
Missing Skills:
Suggestions:

Resume:
{resume_text}

Job Description:
{job_desc}
"""

                    response = ollama.chat(
                        model=model_name,

                        messages=[
                            {
                                "role": "user",
                                "content": prompt
                            }
                        ],

                        options={
                            "num_predict": 150,
                            "temperature": 0.2
                        }
                    )

                    result = response["message"]["content"]

                    st.subheader("🎯 Job Match Result")
                    st.info(result)

        # ==========================================
        # INTERVIEW GENERATOR
        # ==========================================
        if interview_btn:

            with st.spinner("Generating Interview Questions..."):

                prompt = f"""
You are a professional technical interviewer.

Generate:

1. HR Interview Questions
2. Technical Questions
3. Python Questions
4. Machine Learning Questions
5. Project-Based Questions

based on this resume.

Resume:
{resume_text}

Return clear bullet points.
"""

                response = ollama.chat(
                    model=model_name,

                    messages=[
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],

                    options={
                        "num_predict": 220,
                        "temperature": 0.4
                    }
                )

                result = response["message"]["content"]

                st.subheader("🧠 Interview Questions")
                st.write(result)

    except Exception as e:

        st.error(f"Error: {e}")

# --------------------------------
# FOOTER
# --------------------------------
st.markdown("---")
st.caption("Developed by Syed Misbah Uddin")