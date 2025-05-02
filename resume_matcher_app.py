import streamlit as st
import pymupdf
import os
from dotenv import load_dotenv
import requests
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import pandas as pd
import re

# Load API key from .env file
load_dotenv()
API_KEY = os.getenv("OPENROUTER_API_KEY")

# --- Extract Resume Text from PDF ---
def extract_text_from_pdf(uploaded_file):
    try:
        uploaded_file.seek(0)
        doc = pymupdf.open(stream=uploaded_file.read(), filetype="pdf")
        return "\n".join([page.get_text() for page in doc])
    except Exception as e:
        return f"❌ Error reading PDF: {e}"

# --- GPT API Call via OpenRouter ---
def compare_resume_and_jd(resume_text, jd_text):
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    data = {
        "model": "openai/gpt-3.5-turbo",
        "messages": [
            {
                "role": "system",
                "content": "You are an expert in resume matching and screening."
            },
            {
                "role": "user",
                "content": f"""Resume:
{resume_text}

Job Description:
{jd_text}

Now compare them and provide:
1. Matched Skills
2. Missing Skills
3. Match Score (0–100)
4. Suggestions to improve the resume."""
            }
        ]
    }

    response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=data)
    try:
        result_json = response.json()
        return result_json["choices"][0]["message"]["content"]
    except Exception as e:
        return f"❌ API error or unexpected format: {e}"

# --- Extract Skills from GPT Output ---
def extract_skills_from_gpt_output(gpt_text):
    matched = []
    missing = []

    matched_section = re.search(r"Matched Skills:\s*(.*?)\n\n", gpt_text, re.DOTALL)
    if matched_section:
        matched = [line.strip("-• ") for line in matched_section.group(1).split("\n") if line.strip()]

    missing_section = re.search(r"Missing Skills:\s*(.*?)\n\n", gpt_text, re.DOTALL)
    if missing_section:
        missing = [line.strip("-• ") for line in missing_section.group(1).split("\n") if line.strip()]

    return matched, missing

# --- Extract Suggestions from GPT Output ---
def extract_suggestions(gpt_text):
    suggestions = []
    suggestion_section = re.search(r"Suggestions to improve the resume:\s*(.*)", gpt_text, re.DOTALL)
    if suggestion_section:
        suggestions = [line.strip("-• ") for line in suggestion_section.group(1).split("\n") if line.strip()]
    return suggestions

# --- Streamlit App UI ---
st.set_page_config(page_title="Resume Matcher", layout="wide")
st.title("📄 Resume Matcher with GPT (via OpenRouter)")

resume_file = st.file_uploader("Upload Resume (PDF only)", type=["pdf"])
job_desc = st.text_area("Paste Job Description", height=200)

if resume_file and job_desc:
    with st.spinner("Extracting resume text..."):
        resume_text = extract_text_from_pdf(resume_file)

    with st.spinner("Analyzing with GPT..."):
        try:
            result = compare_resume_and_jd(resume_text, job_desc)
            st.subheader("🧠 GPT Resume Analysis")
            st.markdown(result)

            matched_skills, missing_skills = extract_skills_from_gpt_output(result)
            suggestions = extract_suggestions(result)

            # 1. Horizontal Bar Chart
            st.subheader("📊 Skill Match Overview")
            skills = ["Matched", "Missing"]
            counts = [len(matched_skills), len(missing_skills)]
            fig, ax = plt.subplots()
            ax.barh(skills, counts, color=["green", "red"])
            ax.set_xlabel("Number of Skills")
            ax.set_title("Matched vs Missing Skills")
            st.pyplot(fig)

            # 2. Expanders
            with st.expander("✅ Matched Skills"):
                st.write(matched_skills)
            with st.expander("❌ Missing Skills"):
                st.write(missing_skills)

            # 3. GPT Skill Summary
            st.subheader("💡 GPT Skill Fit Summary")
            st.markdown(f"- Strong in: {', '.join(matched_skills[:3])}")
            st.markdown(f"- Improve or add: {', '.join(missing_skills[:3])}")

            # 4. Match Score Gauge
            score = round(100 * len(matched_skills) / max(1, (len(matched_skills) + len(missing_skills))), 2)
            st.subheader("🧭 Overall Match Score")
            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = score,
                title = {"text": "Skill Match %"},
                gauge = {"axis": {"range": [0, 100]}}
            ))
            st.plotly_chart(fig)

            # 5. Skill Table
            skill_df = pd.DataFrame({
                "Skill": matched_skills + missing_skills,
                "Status": ["Matched"] * len(matched_skills) + ["Missing"] * len(missing_skills)
            })
            st.subheader("📋 Skill Match Table")
            st.dataframe(skill_df)

            # 6. Suggestions
            with st.expander("📌 Suggestions to Improve Resume"):
                for s in suggestions:
                    st.markdown("- " + s)

        except Exception as e:
            st.error(f"❌ API call failed: {e}")
