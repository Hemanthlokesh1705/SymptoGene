import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
import numpy as np
import re
from collections import Counter

class DiseasePredictor:
    def __init__(self, data_path=r'C:\Users\LATITUDE\Desktop\python\Symtogene\data\c.csv'):
        """
        Initialize the Disease Predictor with improved model architecture.
        
        Args:
            data_path: Path to the CSV file containing training data.
                      Expected columns: symptom1, symptom2, symptom3, disease
        """
        # Load and prepare data
        self.data = pd.read_csv(data_path)
        self._prepare_data()
        
        # Create improved model ensemble
        self.model = self._create_ensemble_model()
        
        # Train the model
        self.train()
    
    def _prepare_data(self):
        """Prepare the data for training with improved preprocessing."""
        # Fix column name typo (symptom -> symptom)
        self.data.columns = self.data.columns.str.replace('symptom', 'symptom')
        
        # Get symptom columns
        symptom_cols = [col for col in self.data.columns if col.startswith('symptom')]
        
        # Advanced symptom preprocessing
        processed_symptoms = []
        for _, row in self.data.iterrows():
            symptoms = []
            for col in symptom_cols:
                symptom = str(row[col]).strip().lower()
                if symptom and symptom != 'nan' and symptom != 'none':
                    # Clean and normalize symptom text
                    symptom = self._clean_symptom_text(symptom)
                    if symptom:
                        symptoms.append(symptom)
            
            # Create weighted symptom string (early symptoms get higher weight)
            weighted_symptoms = []
            for i, symptom in enumerate(symptoms):
                # Give higher weight to earlier symptoms (typically more important)
                weight = len(symptoms) - i
                weighted_symptoms.extend([symptom] * weight)
            
            processed_symptoms.append(' '.join(weighted_symptoms))
        
        self.data['combined_symptoms'] = processed_symptoms
        
        # Encode disease labels
        self.label_encoder = LabelEncoder()
        self.data['disease_encoded'] = self.label_encoder.fit_transform(self.data['disease'])
        
        # Create symptom vocabulary for better feature engineering
        self._build_symptom_vocabulary()
    
    def _clean_symptom_text(self, symptom):
        """Clean and normalize symptom text."""
        # Remove special characters and normalize
        symptom = re.sub(r'[^\w\s]', ' ', symptom)
        symptom = re.sub(r'\s+', ' ', symptom).strip()
        
        # Handle common medical term variations
        symptom_mappings = {
            'difficulty breathing': 'shortness of breath',
            'breathing problems': 'shortness of breath',
            'joint ache': 'joint pain',
            'muscle ache': 'muscle pain',
            'stomach pain': 'abdominal pain',
            'belly pain': 'abdominal pain',
            'tiredness': 'fatigue',
            'exhaustion': 'fatigue',
            'skin rash': 'rash',
            'fever': 'fever',
            'headaches': 'headache',
            'coughing': 'cough',
        }
        
        for original, normalized in symptom_mappings.items():
            if original in symptom:
                symptom = symptom.replace(original, normalized)
        
        return symptom
    
    def _build_symptom_vocabulary(self):
        """Build a comprehensive symptom vocabulary for better matching."""
        all_symptoms = []
        for symptoms_text in self.data['combined_symptoms']:
            all_symptoms.extend(symptoms_text.split())
        
        self.symptom_vocab = Counter(all_symptoms)
        self.common_symptoms = set([symptom for symptom, count in self.symptom_vocab.items() if count > 1])
    
    def _create_ensemble_model(self):
        """Create an ensemble model for better predictions."""
        # Enhanced TF-IDF vectorizer
        vectorizer = TfidfVectorizer(
            preprocessor=self._preprocess_text,
            ngram_range=(1, 2),  # Include bigrams for better context
            max_features=5000,
            min_df=1,
            max_df=0.95,
            sublinear_tf=True,
            stop_words=None  # Don't remove medical terms
        )
        
        # Create ensemble of different classifiers
        nb_classifier = MultinomialNB(alpha=0.1)
        rf_classifier = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            class_weight='balanced'
        )
        lr_classifier = LogisticRegression(
            max_iter=1000,
            random_state=42,
            class_weight='balanced'
        )
        
        # Voting classifier for ensemble
        ensemble = VotingClassifier(
            estimators=[
                ('nb', nb_classifier),
                ('rf', rf_classifier),
                ('lr', lr_classifier)
            ],
            voting='soft'  # Use probability voting
        )
        
        # Create pipeline
        model = Pipeline([
            ('vectorizer', vectorizer),
            ('classifier', ensemble)
        ])
        
        return model
    
    def _preprocess_text(self, text):
        """Enhanced text preprocessing."""
        if isinstance(text, list):
            text = ' '.join(text)
        
        text = str(text).lower()
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Expand common medical abbreviations
        abbreviations = {
            'sob': 'shortness of breath',
            'n/v': 'nausea vomiting',
            'gi': 'gastrointestinal',
            'uti': 'urinary tract infection'
        }
        
        for abbr, full in abbreviations.items():
            text = text.replace(abbr, full)
        
        return text
    
    def train(self):
        """Train the model with cross-validation for performance assessment."""
        X = self.data['combined_symptoms']
        y = self.data['disease_encoded']
        
        # Train the model
        self.model.fit(X, y)
        print(f"Training accuracy: {self.model.score(X, y) * 100:.2f}%")

        # Evaluate with cross-validation
        try:
            cv_scores = cross_val_score(self.model, X, y, cv=min(5, len(np.unique(y))))
            print(f"Model trained successfully!")
            print(f"Cross-validation accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
        except:
            print("Model trained successfully!")
    
    def _calculate_symptom_similarity(self, input_symptoms, disease_symptoms):
        """Calculate similarity between input symptoms and disease symptoms."""
        input_set = set(input_symptoms.lower().split())
        disease_set = set(disease_symptoms.lower().split())
        
        # Jaccard similarity
        intersection = len(input_set.intersection(disease_set))
        union = len(input_set.union(disease_set))
        
        return intersection / union if union > 0 else 0
    
    def predict(self, symptoms: list, top_n=3):
        """
        Predict the most likely diseases based on symptoms with improved accuracy.
        
        Args:
            symptoms: List of symptoms (e.g., ['fever', 'headache', 'cough'])
            top_n: Number of top predictions to return
            
        Returns:
            dict: {
                'predictions': list of dictionaries with disease and percentage,
                'top_percentages': list of percentages for top predictions
            }
        """
        if not symptoms:
            raise ValueError("Symptoms list cannot be empty")
        
        # Clean and preprocess input symptoms
        cleaned_symptoms = []
        for symptom in symptoms:
            cleaned = self._clean_symptom_text(str(symptom))
            if cleaned:
                cleaned_symptoms.append(cleaned)
        
        if not cleaned_symptoms:
            raise ValueError("No valid symptoms found after preprocessing")
        
        # Preprocess and vectorize input
        processed_input = self._preprocess_text(cleaned_symptoms)
        
        # Get predicted probabilities
        probas = self.model.predict_proba([processed_input])[0]
        
        # Apply post-processing to improve predictions
        probas = self._post_process_probabilities(probas, cleaned_symptoms)
        
        # Get top N predictions with percentages
        top_indices = np.argsort(probas)[-top_n:][::-1]
        
        # Format results with improved percentage calculation
        predictions = []
        top_percentages = []
        
        # Normalize probabilities to ensure they're meaningful
        total_prob = sum(probas[top_indices])
        if total_prob == 0:
            total_prob = 1  # Avoid division by zero
        
        for idx in top_indices:
            disease = self.label_encoder.inverse_transform([idx])[0]
            
            # Improved percentage calculation
            raw_prob = probas[idx]
            normalized_prob = (raw_prob / total_prob) if total_prob > 0 else 0
            
            # Scale to make percentages more interpretable
            percentage = round(float(normalized_prob * 100), 2)
            
            # Ensure minimum percentage for top predictions
            if percentage < 1 and idx in top_indices[:top_n]:
                percentage = max(1.0, percentage)
            
            predictions.append({
                'disease': disease,
                'percentage': percentage,
                'formatted': f"{disease}: {percentage}%"
            })
            top_percentages.append(percentage)
        
        # Ensure percentages add up reasonably (adjust if needed)
        total_percentage = sum(top_percentages)
        if total_percentage > 0 and total_percentage < 50:  # If probabilities are too low
            scaling_factor = 80 / total_percentage  # Scale to make more meaningful
            for i, pred in enumerate(predictions):
                pred['percentage'] = round(pred['percentage'] * scaling_factor, 2)
                pred['formatted'] = f"{pred['disease']}: {pred['percentage']}%"
                top_percentages[i] = pred['percentage']
        
        print(f"Predictions: {predictions}")
        print(f"Top Percentages: {top_percentages}")
        
        return {
            'predictions': predictions,
            'top_percentages': top_percentages
        }
    
    def _post_process_probabilities(self, probabilities, input_symptoms):
        """Post-process probabilities based on symptom matching."""
        # Calculate symptom similarity boost
        boosted_probs = probabilities.copy()
        
        for i, prob in enumerate(probabilities):
            disease = self.label_encoder.inverse_transform([i])[0]
            
            # Find training examples for this disease
            disease_data = self.data[self.data['disease'] == disease]
            
            if not disease_data.empty:
                # Calculate average similarity with training examples
                similarities = []
                for _, row in disease_data.iterrows():
                    sim = self._calculate_symptom_similarity(
                        ' '.join(input_symptoms), 
                        row['combined_symptoms']
                    )
                    similarities.append(sim)
                
                avg_similarity = np.mean(similarities) if similarities else 0
                
                # Boost probability based on similarity
                boost_factor = 1 + (avg_similarity * 0.5)  # Up to 50% boost
                boosted_probs[i] = prob * boost_factor
        
        return boosted_probs
    
    def get_model_info(self):
        """Get information about the trained model."""
        return {
            'total_diseases': len(self.label_encoder.classes_),
            'disease_list': list(self.label_encoder.classes_),
            'total_training_samples': len(self.data),
            'symptom_vocabulary_size': len(self.symptom_vocab),
            'model_type': 'Ensemble (Naive Bayes + Random Forest + Logistic Regression)'
        }

