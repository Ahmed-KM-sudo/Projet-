
from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

# Load the trained model and preprocessor
model = joblib.load('recommendation_model.joblib')
preprocessor = joblib.load('preprocessor.joblib')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    # Get user input from the form
    features = {
        'Age': int(request.form['Age']),
        'Budget': int(request.form['Budget']),
        'Interet': request.form['Interet'],
        'Duree': int(request.form['Duree']),
        'Climat': request.form['Climat']
    }

    # Create a DataFrame from the user input
    input_df = pd.DataFrame([features])

    # Preprocess the user input
    input_processed = preprocessor.transform(input_df)

    # Make a prediction
    prediction = model.predict(input_processed)[0]

    return render_template('index.html', recommendation=prediction)

if __name__ == '__main__':
    app.run(debug=True)
