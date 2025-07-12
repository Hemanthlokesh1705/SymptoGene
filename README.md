# 🧬 SymptoGene

**SymptoGene** is a machine learning-powered web application that predicts the most likely genetic diseases based on user-input symptoms. It provides top-3 disease suggestions with probabilities and includes additional features like doctor information, appointment booking, and a simple login system.

---

## 🚀 Features

- 🔍 **Symptom-Based Prediction** – Input up to 8 symptoms to get disease predictions.
- 🧠 **Ensemble ML Model** – Combines Naive Bayes, Random Forest, and Logistic Regression.
- 🧾 **Symptom Normalization** – Handles synonyms like "tiredness" → "fatigue".
- 📊 **Probability Scoring** – Ranks top 3 genetic diseases with weighted confidence.
- 👨‍⚕️ **Doctor Info & Booking Page** – Dummy page for selecting doctors and booking appointments.
- 🔐 **Simple Login Page** – Basic frontend login interface.

---

## 🛠 Tech Stack

- **Frontend**: HTML, CSS, JavaScript
- **Backend**: Python (Flask)
- **ML Libraries**: `scikit-learn`, `pandas`, `numpy`
- **Model**: Voting Ensemble (TF-IDF + Naive Bayes + Random Forest + Logistic Regression)

---

## 🧪 How It Works

1. **Data Preparation**:
   - Cleans and normalizes symptoms.
   - Applies TF-IDF vectorization on symptoms.

2. **Model Training**:
   - Encodes diseases.
   - Trains a VotingClassifier with cross-validation.

3. **Prediction**:
   - Accepts up to 8 symptoms.
   - Outputs top 3 probable diseases with confidence scores.
   - Boosts probabilities based on symptom similarity.

---

## 📁 Project Structure

SymptoGene/
├── data/ # Contains the dataset (e.g., c.csv)
├── model/ # (Optional) For storing trained models if separated
├── static/ # CSS, JS, and image files
├── templates/ # HTML templates (Flask views)
├── app.py # Flask application entry point
├── predictor.py # Machine learning logic and prediction class
└── README.md # Project documentation
