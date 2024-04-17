import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load and prepare the dataset
data = pd.read_csv(r"C:\Users\drash\OneDrive\Desktop\DataNeuron\Preprocessed_Data.csv")
data['text1'] = data['text1'].apply(lambda x: x.lower())
data['text2'] = data['text2'].apply(lambda x: x.lower())

# Initialize the vectorizer
vectorizer = TfidfVectorizer()
all_texts = pd.concat([data['text1'], data['text2']]) 
tfidf_matrix = vectorizer.fit_transform(all_texts)
half_point = len(tfidf_matrix.toarray()) // 2
similarity_scores = cosine_similarity(tfidf_matrix[:half_point], tfidf_matrix[half_point:])

# Add similarity scores to the dataframe
data['similarity_score'] = similarity_scores.diagonal()

# Print each pair of texts with their similarity score
for index, row in data.iterrows():
    print(f"Text 1: {row['text1']}\nText 2: {row['text2']}\nSimilarity Score: {row['similarity_score']}\n---")

# Optionally, save the results to a new CSV file if you need to use it later
data.to_csv(r"C:\Users\drash\OneDrive\Desktop\DataNeuron\Similarity_Scores1.csv", index=False)

import joblib

# After your vectorizer is fit to the data:
joblib.dump(vectorizer, 'tfidf_vectorizer.joblib')
