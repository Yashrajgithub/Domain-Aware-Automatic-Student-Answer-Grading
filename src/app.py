
import sys
from pathlib import Path

import streamlit as st

# ------------------------------------------------------------
# PATH AND MODEL IMPORT
# ------------------------------------------------------------

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from predict import grade_answer


# ------------------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------------------

st.set_page_config(
    page_title="AnswerGrade AI",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ------------------------------------------------------------
# CUSTOM STYLING
# ------------------------------------------------------------

st.markdown(
    """
    <style>
    
.stApp {
    background-color: #0e1117 !important;
    color: #f1f5f9 !important;
}

.block-container {
    max-width: 1250px;
    padding-top: 1.8rem;
    padding-bottom: 3rem;
}

/* Dark sidebar */
[data-testid="stSidebar"] {
    background-color: #080c14 !important;
}

[data-testid="stSidebar"] * {
    color: #f1f5f9;
}

/* Main headings and labels */
.stMarkdown,
.stMarkdown p,
.stMarkdown h1,
.stMarkdown h2,
.stMarkdown h3,
.stMarkdown h4,
[data-testid="stWidgetLabel"],
[data-testid="stWidgetLabel"] p,
[data-testid="stWidgetLabel"] label,
.stSelectbox label,
.stTextArea label,
.stCaption {
    color: #f1f5f9 !important;
}

/* Header banner */
.hero {
    background: linear-gradient(120deg, #111827, #1d4ed8);
    padding: 30px;
    border: 1px solid #334155;
    border-radius: 18px;
    margin-bottom: 25px;
}

.hero h1 {
    color: #ffffff !important;
    font-size: 2.1rem;
    margin-bottom: 8px;
}

.hero p {
    color: #dbeafe !important;
    font-size: 1.02rem;
    margin-bottom: 0;
}

/* Section headings */
.section-heading {
    font-size: 1.15rem;
    font-weight: 700;
    color: #93c5fd !important;
    margin: 12px 0 8px 0;
}

/* Dark text areas */
[data-testid="stTextArea"] textarea,
[data-testid="stTextInput"] input {
    background-color: #1e293b !important;
    color: #ffffff !important;
    border: 1px solid #475569 !important;
    border-radius: 10px;
}

[data-testid="stTextArea"] textarea::placeholder,
[data-testid="stTextInput"] input::placeholder {
    color: #94a3b8 !important;
    opacity: 1 !important;
}

/* Dropdown */
[data-baseweb="select"] > div {
    background-color: #1e293b !important;
    color: #ffffff !important;
    border-color: #475569 !important;
}

[data-baseweb="popover"],
[data-baseweb="menu"] {
    background-color: #1e293b !important;
}

[data-baseweb="menu"] * {
    color: #ffffff !important;
}

/* Result cards */
.result-card {
    background-color: #161b27;
    border: 1px solid #334155;
    padding: 22px;
    border-radius: 15px;
    margin: 12px 0;
}

.score-label {
    color: #cbd5e1 !important;
    font-size: 0.9rem;
    font-weight: 600;
}

.score-number {
    color: #60a5fa !important;
    font-size: 2.8rem;
    font-weight: 800;
    line-height: 1.25;
}

/* Buttons */
div.stButton > button,
div[data-testid="stFormSubmitButton"] button {
    border-radius: 10px;
    min-height: 48px;
    font-weight: 700;
}

div[data-testid="stFormSubmitButton"] button[kind="primary"] {
    background-color: #2563eb;
    color: white;
    border: none;
}

/* Alerts */
[data-testid="stAlert"] {
    border-radius: 10px;
}

/* Footer */
footer {
    visibility: hidden;
}


    </style>
    """,
    unsafe_allow_html=True,
)


# ------------------------------------------------------------
# DOMAIN CONFIGURATION
# ------------------------------------------------------------

DOMAIN_MAP = {
    "General / Theory": {
        "key": "general",
        "description": "Conceptual, scientific and theory-based answers.",
        "icon": "📚",
    },
    "Machine Learning": {
        "key": "ml",
        "description": "Machine learning concepts and technical answers.",
        "icon": "🤖",
    },
    "Data Engineering": {
        "key": "de",
        "description": "Data processing, distributed systems and engineering.",
        "icon": "⚙️",
    },
}


# ------------------------------------------------------------
# SIDEBAR
# ------------------------------------------------------------

with st.sidebar:
    st.markdown("## 📝 AnswerGrade AI")
    st.caption("Domain-aware automatic grading")

    st.divider()

    st.markdown("### About the project")
    st.write(
        "An ML-based prototype that estimates a student's answer "
        "score by comparing it with a reference answer."
    )

    st.markdown("### Available domains")

    for name, details in DOMAIN_MAP.items():
        st.markdown(f"**{details['icon']} {name}**")
        st.caption(details["description"])

    st.divider()

    st.markdown("### How it works")
    st.markdown(
        "1. Select a domain\n"
        "2. Enter the question\n"
        "3. Provide the reference answer\n"
        "4. Enter the student's answer\n"
        "5. Generate the predicted grade"
    )

    st.divider()
    st.caption("M.Tech Data Science | Mini Project")


# ------------------------------------------------------------
# HERO HEADER
# ------------------------------------------------------------

st.markdown(
    """
    <div class="hero">
        <h1>📝 Automatic Student Answer Grader</h1>
        <p>
            Evaluate student responses using machine learning,
            semantic similarity and domain-specific models.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ------------------------------------------------------------
# DOMAIN SELECTION
# ------------------------------------------------------------

st.markdown(
    '<div class="section-heading">01 &nbsp; Select your domain</div>',
    unsafe_allow_html=True,
)

domain_option = st.selectbox(
    "Choose the subject for evaluation",
    options=list(DOMAIN_MAP.keys()),
    format_func=lambda name: (
        f"{DOMAIN_MAP[name]['icon']}  {name}"
    ),
)

selected_domain = DOMAIN_MAP[domain_option]

st.info(
    f"**Selected domain:** {domain_option}  \n"
    f"{selected_domain['description']}"
)

# ------------------------------------------------------------
# ANSWER INPUT FORM
# ------------------------------------------------------------

st.markdown(
    '<div class="section-heading">02 &nbsp; Enter answer details</div>',
    unsafe_allow_html=True,
)

with st.form("grading_form", clear_on_submit=False):

    question_col, reference_col = st.columns(2, gap="large")

    with question_col:
        st.markdown("#### ❓ Question")
        question = st.text_area(
            "Question",
            placeholder="Enter the question asked to the student...",
            height=150,
            label_visibility="collapsed",
        )

    with reference_col:
        st.markdown("#### 📖 Reference answer")
        reference_answer = st.text_area(
            "Reference answer",
            placeholder="Enter the expected or model answer...",
            height=150,
            label_visibility="collapsed",
        )

    st.markdown("#### ✍️ Student response")

    student_answer = st.text_area(
        "Student answer",
        placeholder="Paste the answer submitted by the student...",
        height=190,
        label_visibility="collapsed",
    )

    submitted = st.form_submit_button(
        "🎯  Evaluate Answer",
        type="primary",
        use_container_width=True,
    )


# ------------------------------------------------------------
# GRADING LOGIC
# ------------------------------------------------------------

if submitted:

    if not question.strip():
        st.session_state.pop("grading_result", None)
        st.warning("Please enter the question.")

    elif not reference_answer.strip():
        st.session_state.pop("grading_result", None)
        st.warning("Please enter the reference answer.")

    elif not student_answer.strip():
        st.session_state.pop("grading_result", None)
        st.warning("Please enter the student's answer.")

    else:
        try:
            with st.spinner("Analyzing the student response..."):

                result = grade_answer(
                    domain=selected_domain["key"],
                    question=question.strip(),
                    reference_answer=reference_answer.strip(),
                    student_answer=student_answer.strip(),
                )

            st.session_state["grading_result"] = {
                **result,
                "selected_domain": domain_option,
            }

        except Exception as error:
            st.session_state.pop("grading_result", None)
            st.error("Unable to evaluate this answer.")
            with st.expander("Technical error details"):
                st.code(str(error))


# ------------------------------------------------------------
# DISPLAY RESULTS
# ------------------------------------------------------------

result = st.session_state.get("grading_result")

if result:

    st.divider()

    st.markdown(
        '<div class="section-heading">03 &nbsp; Evaluation results</div>',
        unsafe_allow_html=True,
    )

    try:
        score = float(result["score"])
        normalized_score = float(result["normalized_score"])

        if not 0 <= score <= 10:
            raise ValueError("Score is outside the expected 0–10 range.")

        if not 0 <= normalized_score <= 1:
            raise ValueError(
                "Normalized score is outside the expected 0–1 range."
            )

    except (KeyError, TypeError, ValueError) as error:
        st.error(f"The model returned an unexpected result: {error}")
    else:
        score_col, normalized_col, domain_col = st.columns(3, gap="medium")

        with score_col:
            st.markdown(
                f"""
                <div class="result-card">
                    <div class="score-label">PREDICTED SCORE</div>
                    <div class="score-number">{score:.2f}<span
                        style="font-size:1.2rem;color:#64748b;"> / 10</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with normalized_col:
            st.markdown(
                f"""
                <div class="result-card">
                    <div class="score-label">NORMALIZED SCORE</div>
                    <div style="font-size:2.1rem;font-weight:800;
                                color:#0f766e;margin-top:8px;">
                        {normalized_score:.4f}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with domain_col:
            st.markdown(
                f"""
                <div class="result-card">
                    <div class="score-label">EVALUATED DOMAIN</div>
                    <div style="font-size:1.25rem;font-weight:750;
                                color:#334155;margin-top:14px;">
                        {result.get('selected_domain', domain_option)}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("#### Score visualization")
        st.progress(normalized_score)
        st.caption(
            "The bar represents the predicted score normalized to a 0–1 scale."
        )

        st.markdown("#### 💬 Automated feedback")
        st.info(result.get("feedback", "No feedback was returned by the model."))

        with st.expander("Review submitted answers"):
            st.markdown("**Question**")
            st.write(question)

            st.markdown("**Reference answer**")
            st.write(reference_answer)

            st.markdown("**Student answer**")
            st.write(student_answer)

        st.caption(
            "This is a model-generated estimate. Scores may not perfectly "
            "reflect factual correctness or the full quality of an answer."
        )


# ------------------------------------------------------------
# BULK GRADING: CSV / EXCEL UPLOAD
# ------------------------------------------------------------

st.divider()
st.markdown(
    '<div class="section-heading">04 &nbsp; Bulk student answer grading</div>',
    unsafe_allow_html=True,
)
st.write(
    "Upload a CSV or Excel file containing student answers. Enter the question "
    "and reference answer below, then grade all rows in one batch."
)

with st.container(border=True):
    bulk_question = st.text_area(
        "Bulk grading question",
        placeholder="Enter the question shared by all students...",
        height=110,
        key="bulk_question",
    )
    bulk_reference = st.text_area(
        "Bulk grading reference answer",
        placeholder="Enter the expected/model answer...",
        height=130,
        key="bulk_reference",
    )

    uploaded_file = st.file_uploader(
        "Upload student answers (.csv or .xlsx)",
        type=["csv", "xlsx"],
        help=(
            "Include one column containing student answers. Optional columns "
            "such as Student ID and Student Name will be retained in the results."
        ),
        key="bulk_answer_file",
    )

    if uploaded_file is not None:
        try:
            import pandas as pd

            if uploaded_file.name.lower().endswith(".csv"):
                input_df = pd.read_csv(uploaded_file)
            else:
                input_df = pd.read_excel(uploaded_file, engine="openpyxl")

            if input_df.empty:
                st.warning("The uploaded file has no rows.")
            elif len(input_df.columns) == 0:
                st.warning("The uploaded file has no columns.")
            else:
                st.success(
                    f"Loaded {len(input_df):,} rows and "
                    f"{len(input_df.columns):,} columns."
                )
                st.dataframe(input_df.head(10), use_container_width=True)

                answer_column = st.selectbox(
                    "Select the column containing student answers",
                    options=list(input_df.columns),
                    key="bulk_answer_column",
                )

                with st.expander("Input file details"):
                    st.write("Available columns:", [str(c) for c in input_df.columns])
                    st.caption(
                        "Rows with an empty answer are skipped. Other columns "
                        "are preserved in the downloaded results."
                    )

                run_bulk = st.button(
                    "📊 Grade All Student Answers",
                    type="primary",
                    use_container_width=True,
                    key="run_bulk_grading",
                )

                if run_bulk:
                    if not bulk_question.strip():
                        st.error("Enter the question before grading the file.")
                    elif not bulk_reference.strip():
                        st.error("Enter the reference answer before grading the file.")
                    else:
                        results = []
                        total_rows = len(input_df)
                        progress_bar = st.progress(0)
                        status_text = st.empty()

                        for row_index, row in input_df.iterrows():
                            raw_answer = row[answer_column]

                            if pd.isna(raw_answer) or not str(raw_answer).strip():
                                results.append({
                                    "Predicted Score (/10)": None,
                                    "Normalized Score (0-1)": None,
                                    "Automated Feedback": "Skipped: student answer is empty.",
                                    "Grading Status": "Skipped - empty answer",
                                })
                            else:
                                try:
                                    prediction = grade_answer(
                                        domain=selected_domain["key"],
                                        question=bulk_question.strip(),
                                        reference_answer=bulk_reference.strip(),
                                        student_answer=str(raw_answer).strip(),
                                    )
                                    results.append({
                                        "Predicted Score (/10)": prediction.get("score"),
                                        "Normalized Score (0-1)": prediction.get(
                                            "normalized_score"
                                        ),
                                        "Automated Feedback": prediction.get(
                                            "feedback", ""
                                        ),
                                        "Grading Status": "Graded",
                                    })
                                except Exception as error:
                                    results.append({
                                        "Predicted Score (/10)": None,
                                        "Normalized Score (0-1)": None,
                                        "Automated Feedback": f"Could not grade this row: {error}",
                                        "Grading Status": "Error",
                                    })

                            completed = row_index + 1
                            progress_bar.progress(min(completed / total_rows, 1.0))
                            status_text.text(
                                f"Processing student answer {completed} of {total_rows}..."
                            )

                        result_df = input_df.copy()
                        result_df["Question"] = bulk_question.strip()
                        result_df["Reference Answer"] = bulk_reference.strip()
                        result_df["Selected Domain"] = domain_option
                        prediction_df = pd.DataFrame(results, index=input_df.index)
                        result_df = pd.concat([result_df, prediction_df], axis=1)

                        st.session_state["bulk_grading_results"] = result_df
                        st.session_state["bulk_grading_filename"] = uploaded_file.name
                        status_text.success("Bulk grading finished.")

        except ImportError as error:
            st.error(
                "A required package for reading Excel files is missing. "
                "Install it with: python -m pip install openpyxl"
            )
            with st.expander("Technical details"):
                st.code(str(error))
        except Exception as error:
            st.error("Could not read the uploaded file.")
            with st.expander("Technical error details"):
                st.code(str(error))

bulk_results = st.session_state.get("bulk_grading_results")

if bulk_results is not None:
    st.markdown("#### Bulk grading results")
    graded_count = int((bulk_results["Grading Status"] == "Graded").sum())
    skipped_count = int(
        (bulk_results["Grading Status"] == "Skipped - empty answer").sum()
    )
    error_count = int((bulk_results["Grading Status"] == "Error").sum())

    metric1, metric2, metric3 = st.columns(3)
    metric1.metric("Graded", graded_count)
    metric2.metric("Skipped", skipped_count)
    metric3.metric("Errors", error_count)

    st.dataframe(bulk_results, use_container_width=True, hide_index=True)

    csv_bytes = bulk_results.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        "⬇️ Download results as CSV",
        data=csv_bytes,
        file_name="answer_grading_results.csv",
        mime="text/csv",
        use_container_width=True,
        key="download_bulk_csv",
    )

    try:
        from io import BytesIO
        import pandas as pd

        excel_buffer = BytesIO()
        with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
            bulk_results.to_excel(writer, index=False, sheet_name="Grading Results")
        st.download_button(
            "⬇️ Download results as Excel",
            data=excel_buffer.getvalue(),
            file_name="answer_grading_results.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
            key="download_bulk_excel",
        )
    except ImportError:
        st.info(
            "Excel export requires openpyxl. Install it with: "
            "python -m pip install openpyxl. CSV export is still available."
        )

    st.caption(
        "Scores are model-generated estimates, not guaranteed correctness judgments. "
        "Review unusual or high-impact results manually."
    )


# ------------------------------------------------------------
# FOOTER
# ------------------------------------------------------------

st.divider()

footer_col1, footer_col2 = st.columns(2)

with footer_col1:
    st.caption("🎓 M.Tech Data Science | Automatic Answer Grading")

with footer_col2:
    st.markdown(
        '<p style="text-align:right;color:#64748b;font-size:0.8rem;">'
        'Machine Learning Prototype</p>',
        unsafe_allow_html=True,
    )
