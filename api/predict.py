from flask import Flask, request, jsonify
import joblib
import nltk
nltk.download('punkt_tab')

# Load the trained SVM model and TF-IDF vectorizer
svm_model = joblib.load('svm_model.pkl')
tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Initialize the Flask app
app = Flask(__name__)


@app.route('/predict', methods=['POST'])
def predict():
    # Get the comment from the POST request
    data = request.json
    comment = data.get('comment', '')

    if not comment:
        return jsonify({'error': 'No comment provided'}), 400

    # Preprocess the comment (you may refine this if needed)
    def preprocess_text(text):
        from nltk.tokenize import word_tokenize
        from nltk.corpus import stopwords
        import nltk
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        
        stop_words = set(stopwords.words('english'))
        words = word_tokenize(text.lower())
        words = [word for word in words if word.isalpha() and word not in stop_words]
        return ' '.join(words)

    processed_comment = preprocess_text(comment)

    # Transform the comment using the TF-IDF vectorizer
    vectorized_comment = tfidf_vectorizer.transform([processed_comment])

    # Make a prediction using the SVM model
    prediction = svm_model.predict(vectorized_comment)[0]

    # Return the prediction as a JSON response
    return jsonify({'comment': comment, 'prediction': prediction})

if __name__ == '__main__':
    app.run(debug=False)
