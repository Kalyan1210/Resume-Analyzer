import os
import re
import requests
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import streamlit as st
import pymupdf  # PyMuPDF

# ---- App config ----
st.set_page_config(page_title="Resume Matcher", layout="wide")
st.title("📄 Resume Matcher with GPT (via OpenRouter)")

# ---- Secrets / key handling ----
OPENROUTER_API_KEY = st.secrets.get("OPENROUTER_API_KEY") or os.getenv("OPENROUTER_API_KEY")
APP_URL   = st.secrets.get("APP_URL", "")
APP_TITLE = st.secrets.get("APP_TITLE", "Resume Analyzer")

# BYO-key fallback for local runs or forks
if not OPENROUTER_API_KEY:
    with st.sidebar:
        st.info("Enter your OpenRouter key to run locally or on forks.")
        user_key = st.text_input("OpenRouter API key", type="password")
        if user_key:
            OPENROUTER_API_KEY = user_key

if not OPENROUTER_API_KEY:
    st.warning("No API key found. Add it in Streamlit Secrets or enter your own in the sidebar.")
    st.stop()

def openrouter_headers():
    return {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "HTTP-Referer": APP_URL,   # optional analytics on OpenRouter
        "X-Title": APP_TITLE
    }

# ---- PDF text extraction ----
def extract_text_from_pdf(uploaded_file):
    try:
        uploaded_file.seek(0)
        doc = pymupdf.open(stream=uploaded_file.read(), filetype="pdf")
        return "\n".join(page.get_text() for page in doc)
    except Exception as e:
        return f"❌ Error reading PDF: {e}"

# ---- GPT call (OpenRouter, GPT-only) ----
GPT_MODEL = "openai/gpt-4o"  # or "openai/gpt-4o" if you prefer

def call_gpt(messages, model=GPT_MODEL, temperature=0.2, max_tokens=900):
    url = "https://openrouter.ai/api/v1/chat/completions"
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    try:
        r = requests.post(url, headers=openrouter_headers(), json=payload, timeout=60)
        r.raise_for_status()
        data = r.json()
        return data["choices"][0]["message"]["content"]
    except requests.HTTPError:
        st.error(f"OpenRouter error {r.status_code}: {r.text[:400]}")
    except Exception as e:
        st.error(f"Request failed: {e}")
    return None

def compare_resume_and_jd(resume_text, jd_text):
    prompt = f"""You are an expert in resume matching and screening.

Compare the following resume and job description and return:
1. Matched Skills
2. Missing Skills
3. Match Score (0-100)
4. Suggestions to improve the resume

Resume:
{resume_text}

Job Description:
{jd_text}
"""
    return call_gpt(
        messages=[
            {"role": "system", "content": "You evaluate resume-to-JD matches clearly and concisely."},
            {"role": "user", "content": prompt},
        ]
    )

# ---- Parsing helpers ----
def extract_skills_from_gpt_output(gpt_text):
    matched, missing = [], []

    # Try to find sections robustly
    m = re.search(r"Matched Skills[:\s]*([\s\S]*?)(?:\n\s*\n|$)", gpt_text, re.IGNORECASE)
    if m:
        matched = [s.strip(" -•\t") for s in m.group(1).splitlines() if s.strip()]

    n = re.search(r"Missing Skills[:\s]*([\s\S]*?)(?:\n\s*\n|$)", gpt_text, re.IGNORECASE)
    if n:
        missing = [s.strip(" -•\t") for s in n.group(1).splitlines() if s.strip()]

    return matched, missing

def extract_suggestions(gpt_text):
    s = re.search(r"Suggestions(?: to improve the resume)?[:\s]*([\s\S]*?)(?:\n\s*\n|$)", gpt_text, re.IGNORECASE)
    if not s:
        return []
    return [line.strip(" -•\t") for line in s.group(1).splitlines() if line.strip()]

# ---- UI ----
resume_file = st.file_uploader("Upload Resume (PDF only)", type=["pdf"])
job_desc = st.text_area("Paste Job Description", height=200)

if resume_file and job_desc:
    with st.spinner("Extracting resume text..."):
        resume_text = extract_text_from_pdf(resume_file)

    if resume_text.startswith("❌ Error"):
        st.error(resume_text)
    else:
        with st.spinner("Analyzing with GPT..."):
            result = compare_resume_and_jd(resume_text, job_desc)

        if not result:
            st.stop()

        st.subheader("🧠 GPT Resume Analysis")
        st.markdown(result)

        matched_skills, missing_skills = extract_skills_from_gpt_output(result)
        suggestions = extract_suggestions(result)

        # 1) Horizontal Bar Chart
        st.subheader("📊 Skill Match Overview")
        labels = ["Matched", "Missing"]
        counts = [len(matched_skills), len(missing_skills)]
        fig, ax = plt.subplots()
        ax.barh(labels, counts, color=["green", "red"])
        ax.set_xlabel("Number of Skills")
        ax.set_title("Matched vs Missing Skills")
        st.pyplot(fig)

        # 2) Expanders
        with st.expander("✅ Matched Skills"):
            st.write(matched_skills if matched_skills else "None detected.")
        with st.expander("❌ Missing Skills"):
            st.write(missing_skills if missing_skills else "None detected.")

        # 3) GPT Skill Summary
        st.subheader("💡 GPT Skill Fit Summary")
        st.markdown(f"- Strong in: {', '.join(matched_skills[:3]) or 'n/a'}")
        st.markdown(f"- Improve or add: {', '.join(missing_skills[:3]) or 'n/a'}")

        # 4) Match Score Gauge
        score = round(100 * len(matched_skills) / max(1, len(matched_skills) + len(missing_skills)), 2)
        st.subheader("🧭 Overall Match Score")
        gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=score,
            title={"text": "Skill Match %"},
            gauge={"axis": {"range": [0, 100]}}
        ))
        st.plotly_chart(gauge, use_container_width=True)

        # 5) Skill Table
        skill_df = pd.DataFrame({
            "Skill": matched_skills + missing_skills,
            "Status": (["Matched"] * len(matched_skills)) + (["Missing"] * len(missing_skills)),
        })
        st.subheader("📋 Skill Match Table")
        st.dataframe(skill_df, use_container_width=True)

        # 6) Suggestions
        with st.expander("📌 Suggestions to Improve Resume"):
            if suggestions:
                for s in suggestions:
                    st.markdown("- " + s)
            else:
                st.write("No suggestions parsed.")
