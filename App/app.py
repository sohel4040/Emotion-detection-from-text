
from flask import Flask, render_template, request
import joblib
import numpy as np
import json  # Import the json module

app = Flask(__name__)

# Load machine learning model
pipe_lr = joblib.load(open("models/best_algorithm_test2.pkl", "rb"))

# Dictionary mapping emotions to emojis

emotions_emoji_dict = {
    "anger": "😠",
    "fear": "😨",
    "joy": "😃",
    "love": "❤️",
    "sadness": "😔",
    "surprise": "😮",
}


# Function to predict emotions
def predict_emotions(docx):
    results = pipe_lr.predict([docx])
    #print(results[0])
    return results[0]

def get_prediction_proba(docx):
    results = pipe_lr.predict_proba([docx])
    #print(results[0])
    return results.tolist()  # Convert ndarray to list

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        raw_text = request.form['raw_text']

        # Make prediction
        prediction = predict_emotions(raw_text)
        #print(prediction)
        prediction_probabilities = get_prediction_proba(raw_text)  # Use a new variable name for the probabilities
        #print(prediction_probabilities)

        # Get emoji for the predicted emotion
        emoji_icon = emotions_emoji_dict[prediction]
        
        probability = np.max(get_prediction_proba(raw_text))
        
        print(prediction_probabilities)

        # Render prediction result template with data
        return render_template('index.html', raw_text=raw_text, prediction=prediction,
                               emoji_icon=emoji_icon, prediction_probabilities=prediction_probabilities, probability=probability)  # Serialize the probabilities using json.dumps

        
 
if __name__ == '__main__':
    app.run(debug=True)
