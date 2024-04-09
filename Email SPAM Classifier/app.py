from flask import Flask, request, jsonify
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

app = Flask(__name__)

# Load your trained model
# Assuming `model` is your trained classifier

# Define a route for making predictions
@app.route('/predict', methods=['POST'])
def predict():
    # Get input data from the request
    data = request.form['email']  # Assuming JSON format for input data

    
    # Preprocess input text (vectorize it)
    cv = CountVectorizer()
    email_vector = cv.transform([data])
    
    # Make predictions using the model
    predicted_label = model.predict(email_vector)
    
    # Convert the prediction to the appropriate format (e.g., JSON)
    response = {'predicted_label': predicted_label.item()}  # Assuming the prediction is a single label
    
    # Return the prediction as JSON
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
