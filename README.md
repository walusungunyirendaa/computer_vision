# Diabetic Retinopathy Grading from Fundus Photographs

It grades a single retinal fundus photograph on the 5-point ICDR diabetic
retinopathy severity scale: **No DR, Mild, Moderate, Severe, Proliferative DR**.

> **Not a medical device.** This model has not been clinically validated and must
> not be used to make, support, or delay any real medical decision.

## Project contents

| File | Purpose |
|---|---|
| `Diabetic_retinopathy_classification.ipynb` | Full documentation (sections A1-A8): problem framing, dataset, splitting, model, training, evaluation, export, limitations |
| `app.py` | Streamlit portal (UI only) |
| `model_utils.py` | Model loading, preprocessing, prediction, Grad-CAM (no Streamlit dependency) |
| `test_model_utils.py` | pytest suite for preprocessing and Grad-CAM |
| `retinopathy_model.keras` | Trained model saved from the notebook |
| `requirements.txt` | Pinned dependencies |

## Dataset

APTOS 2019 Blindness Detection (Aravind Eye Hospital / Asia Pacific Tele-Ophthalmology
Society), accessed through the Kaggle mirror
[`mariaherrerot/aptos2019`](https://www.kaggle.com/datasets/mariaherrerot/aptos2019).
Original competition:
[aptos2019-blindness-detection](https://www.kaggle.com/competitions/aptos2019-blindness-detection).
Check the dataset's Kaggle page for its current licence and usage terms. The data
is not included in this repository.

## Model

- Backbone: MobileNetV2 (frozen, ImageNet weights) with a small dense classification head
- Input: 224x224 RGB, pixel values scaled to [0, 1]
- Output: probability for each of the 5 severity grades
- Preprocessing at inference is identical to training (see notebook section A7 and
  `preprocess()` in `model_utils.py`)

Evaluation (confusion matrix, per-class precision/recall/F1) and an honest discussion
of failure cases are in notebook sections A6 and A8.

## Setup

Requires Python 3.10+.

    git clone <your-repo-url>
    cd <repo-folder>
    python3 -m venv env
    source env/bin/activate        # Windows: env\Scripts\activate
    pip install -r requirements.txt

## Run the app

    streamlit run app.py

Opens at http://localhost:8501. Upload a PNG or JPG fundus photograph to see the
predicted grade, a confidence score, and (optionally) a Grad-CAM heatmap showing the
image regions that most influenced the prediction. Predictions below 50% confidence
are shown with a warning.

## Run the tests

    pytest -v

## Reproduce the model

Run `Diabetic_retinopathy_classification.ipynb` top to bottom. It downloads the
dataset through `kagglehub` (requires a free Kaggle account) and writes
`retinopathy_model.keras` next to the notebook.

## Limitations

- Trained on data from a single hospital system in India; performance on other
  populations, cameras, or image quality is unknown.
- The dataset is class-imbalanced, and labels carry inter-grader subjectivity,
  particularly between adjacent grades (e.g. Moderate vs. Severe).
- Intended for single 2D colour fundus photographs only.
- A real clinical tool would require multi-site validation, calibration, and
  clinician oversight.
