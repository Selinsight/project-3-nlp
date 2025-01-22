from flask import Flask, render_template, request
from joblib import load
import scipy.sparse as sp
import numpy as np

app = Flask(__name__)
app_name = "Fake News Detector"

# Load the saved Random Forest model and vectorizer
model = load('random_forest_model.joblib')
vectorizer = load('tfidf_vectorizer.joblib')

# Define misleading words
misleading_words = ["shocking", "hidden", "exposed", "warning", "secret"]

# Function to count misleading words in a headline
def count_misleading_words(text):
    return sum(1 for word in text.split() if word.lower() in misleading_words)

# Prediction function
def predict_headline(headline):
    # Vectorize the headline
    vectorized = vectorizer.transform([headline])
    
    # Count misleading words
    misleading_count = np.array([[count_misleading_words(headline)]])
    
    # Combine vectorized features with misleading word count
    combined_features = sp.hstack([vectorized, sp.csr_matrix(misleading_count)])
    
    # Predict with the Random Forest model
    prediction = model.predict(combined_features)[0]
    return "Real" if prediction == 1 else "Fake"

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        headline = request.form.get('headline')
        if not headline or not headline.strip():
            return render_template('index.html', prediction="Error: Empty input provided.", headline=None)
        result = predict_headline(headline)
        return render_template('index.html', prediction=result, headline=headline)
    return render_template('index.html', prediction=None, headline=None)

if __name__ == '__main__':
    app.run(debug=True)