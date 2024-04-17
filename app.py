from flask import Flask, request, jsonify, render_template
import joblib
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Load the pre-trained vectorizer
vectorizer = joblib.load("/home/ec2-user/tfidfvectorizer.joblib")

@app.route('/')
def index():
    print("Accessing the index page")
    return render_template('index.html')

@app.route('/similarity', methods=['POST'])
def get_similarity_score():
    print("Accessing the similarity endpoint")
    data = request.json
    if not all(k in data for k in ("text1", "text2")):
        return jsonify({"error": "Missing 'text1' or 'text2' in request"}), 400
    text1 = data['text1']
    text2 = data['text2']
    tfidf_vector1 = vectorizer.transform([text1])
    tfidf_vector2 = vectorizer.transform([text2])
    similarity_score = cosine_similarity(tfidf_vector1, tfidf_vector2)[0][0]
    return jsonify({"similarity score": similarity_score})

if __name__ == '__main__':
    app.run(host='0.0.0.0',port = 8080 )
