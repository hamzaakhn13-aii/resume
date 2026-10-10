# resume# 📄 AI Resume ATS Analyzer

Upload a resume (PDF, DOCX or TXT) and get:

- an **estimated ATS score** out of 100
- a **score breakdown** (keywords, impact, formatting, structure, language)
- **prioritised improvements**, each with a ready-to-use example rewrite
- **matched and missing keywords**, more accurate if you paste a job description

Built with [Streamlit](https://streamlit.io) and Google **Gemini Flash**.

## How the score works

Gemini rates five categories from 0 to 100. The app then calculates the overall score itself using fixed weights, so results stay consistent:

| Category | Weight |
|---|---|
| Keywords & Relevance | 30% |
| Impact & Achievements | 25% |
| ATS-Friendly Formatting | 20% |
| Structure & Completeness | 15% |
| Language & Clarity | 10% |

> ⚠️ The score is an AI **estimate** based on text extracted from your file. Real ATS products differ, so treat it as guidance, not a guarantee.

## Run locally

```bash
# 1. (optional) create a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 2. install dependencies
pip install -r requirements.txt

# 3. set your Gemini API key (free key: https://aistudio.google.com/apikey)
export GEMINI_API_KEY="your-key"      # Windows PowerShell: $env:GEMINI_API_KEY="your-key"

# 4. run
streamlit run app.py
```

You can also skip step 3 and paste the key into the app's sidebar.

## Deploy on Streamlit Community Cloud

1. Push this repo to GitHub (`app.py`, `requirements.txt`, `README.md` in the root).
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
3. Click **Create app**, pick your repo and branch, and set the main file to `app.py`.
4. Open **Advanced settings → Secrets** and add:
   ```toml
   GEMINI_API_KEY = "your-key"
   ```
5. Click **Deploy**.

## Project structure

```
.
├── app.py             # Streamlit app
├── requirements.txt   # Python dependencies
└── README.md
```

## Configuration

- **Model**: defaults to `gemini-2.5-flash`. Change it in the sidebar, or edit `DEFAULT_MODEL` in `app.py`, if you want a different Flash version.
- **Limits**: max file size 5 MB; resume text is truncated at 15,000 characters.

## Privacy

Resumes are processed in memory and sent to the Gemini API for analysis. The app does not save them. Review Google's API data policy before uploading sensitive documents.

## Limitations

- Scanned or image-only PDFs have no extractable text. The app will ask for a text-based file.
- Formatting problems such as columns, graphics or text boxes are inferred from the extracted text, not from the visual layout.
