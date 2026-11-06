# Domain-Aware Automatic Student Answer Grading

**M.Tech Data Science Mini Project**\

## Overview

This project explores automated evaluation of student answers using
Natural Language Processing (NLP) and supervised Machine Learning. It
compares a student's response with a reference answer, predicts a score
out of 10, and provides automated feedback.

The system combines Sentence Transformer semantic similarity, TF-IDF
similarity, and lexical features. A Random Forest regression model uses
these features to estimate answer quality. Domain-specific model
pipelines are provided for General/Theory, Machine Learning, and Data
Engineering.

> The application is a grading-assistance prototype. Similarity does not
> guarantee factual correctness, so human review remains important.

## Objectives

-   Automate preliminary evaluation of written student answers.
-   Combine semantic, TF-IDF, and lexical features.
-   Compare a TF-IDF cosine-similarity baseline with Random Forest
    regression.
-   Explore domain-specific grading for General/Theory, Machine
    Learning, and Data Engineering.
-   Provide individual and bulk grading through a Streamlit application.
-   Analyze grading errors and identify future improvements.

## System Workflow

``` text
Question + Reference Answer + Student Answer
                    |
                    v
              Text Processing
                    |
                    v
              Feature Extraction
       +------------+-------------+
       |            |             |
  Semantic       TF-IDF       Lexical features
  similarity    similarity    (length, ratio,
                              common words,
                              word overlap)
       +------------+-------------+
                    |
                    v
        Domain-specific Random Forest
                    |
                    v
           Predicted score / 10
                    |
                    v
         Normalized score + feedback
```

## Datasets

Two dataset sources were used:

-   **ASAG2024:** Used for the overall model and General/Theory and Data
    Engineering experiments.
-   **DigiKlausur:** Used for the Machine Learning domain experiment.

Recorded cleaned ASAG2024 splits:

  Split          Samples
  ------------ ---------
  Training        14,276
  Validation       1,775
  Test             1,787

Recorded DigiKlausur Machine Learning split:

  Split          Samples
  ------------ ---------
  Training           456
  Validation         151

Raw datasets are excluded from this repository. Obtain them from their
original sources and follow their access and usage conditions.

## Methodology

### Feature extraction

The pipeline uses seven features:

1.  Semantic similarity from Sentence Transformer `all-MiniLM-L6-v2`.
2.  TF-IDF similarity.
3.  Reference-answer length.
4.  Student-answer length.
5.  Length ratio.
6.  Common words.
7.  Word overlap.

### Models

-   **Baseline:** TF-IDF cosine similarity.
-   **Feature-based model:** Random Forest regression using semantic,
    TF-IDF, and lexical features.
-   **Domain-specific models:** Separate model pipelines for
    General/Theory, Machine Learning, and Data Engineering.

## Experimental Results

The following baseline and overall Random Forest results were recorded
on the same validation split.

  Metric            TF-IDF baseline   Random Forest
  --------------- ----------------- ---------------
  MAE                        0.4904          0.2621
  MSE (approx.)              0.3326          0.1050
  RMSE                       0.5767          0.3240
  R²                        -1.3323          0.2637

The Random Forest model reduced MAE by approximately **46.6%** and RMSE
by approximately **43.8%** compared with the baseline. MSE values are
approximate, calculated by squaring the reported rounded RMSE values.

### Additional 40-answer functional test

  Expected category     Nonblank answers   Mean predicted score (/10)
  ------------------- ------------------ ----------------------------
  Correct                             10                         6.60
  Partially correct                   10                         3.65
  Incorrect                           10                         4.93
  Irrelevant                           8                         1.29
  Blank                                2                      Skipped

The functional test identified incorrect answers receiving high scores
and some correct answers being underscored. This indicates that text
similarity alone does not guarantee factual correctness.

## Application Features

-   Domain selection.
-   Individual answer evaluation.
-   Predicted score out of 10, normalized score, and automated feedback.
-   Bulk grading through CSV and Excel files.
-   Student-answer column selection.
-   Export of grading results.

## Project Structure

``` text
Domain-Aware-Automatic-Student-Answer-Grading/
|
|-- src/
|   |-- app.py
|   |-- predict.py
|   |-- baseline_model.py
|   |-- train_ml_model.py
|   |-- train_domain_model.py
|   |-- train_ml_domain.py
|
|-- models/
|   |-- trained model and vectorizer files
|
|-- evaluation_results.csv
|-- Example.txt
|-- inspect_dataset.py
|-- ML_Report_Final.docx
|-- Student answers.csv
|-- .gitignore
|-- README.md
```

The exact contents may vary with the latest project version. The
application requires the model and vectorizer files expected by
`src/predict.py`.

## Installation and Execution

### Prerequisites

-   Python (use a version compatible with the installed dependencies).
-   Git.
-   Required trained model and vectorizer files.
-   Dataset files if retraining the models.

### Clone the repository

``` bash
git clone https://github.com/Yashrajgithub/Domain-Aware-Automatic-Student-Answer-Grading.git
cd Domain-Aware-Automatic-Student-Answer-Grading
```

### Create a virtual environment

Windows PowerShell:

``` powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, use Command Prompt:

``` cmd
.venv\Scripts\activate.bat
```

### Install dependencies

``` bash
pip install streamlit pandas numpy scikit-learn sentence-transformers openpyxl pyarrow
```

If a `requirements.txt` is available, use:

``` bash
pip install -r requirements.txt
```

### Run the application

From the repository root:

``` bash
python -m streamlit run src/app.py
```

Open the local URL printed by Streamlit in your browser.

## Limitations

-   Semantic and lexical similarity do not independently verify factual
    correctness.
-   Incorrect or contradictory answers may receive higher scores than
    expected.
-   Some correct paraphrased answers may be underscored.
-   Performance may vary across domains and datasets.
-   The system should be treated as a grading-assistance prototype, with
    human review for final assessment.

## Future Work

-   Add and validate contradiction detection.
-   Incorporate key-concept coverage and factual consistency checks.
-   Evaluate on larger and more diverse datasets.
-   Improve partial-credit assignment.
-   Conduct further domain-wise and question-wise error analysis.

## Technologies

Python · NLP · Sentence Transformers · TF-IDF · scikit-learn · Random
Forest Regression · Streamlit · Pandas

## Author

**Yashraj Kalshetti**\
M.Tech Data Science, COEP Technological University

## GitHub Repository

[Domain-Aware Automatic Student Answer
Grading](https://github.com/Yashrajgithub/Domain-Aware-Automatic-Student-Answer-Grading)
