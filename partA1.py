import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string

# Load your dataset
data = pd.read_csv(r"C:\Users\drash\OneDrive\Desktop\DataNeuron\DataNeuron_Text_Similarity.csv")

# Basic preprocessing
data['text1'] = data['text1'].str.lower()
data['text2'] = data['text2'].str.lower()

# Removing punctuation
punctuation_table = str.maketrans('', '', string.punctuation)
data['text1'] = data['text1'].apply(lambda text: text.translate(punctuation_table))
data['text2'] = data['text2'].apply(lambda text: text.translate(punctuation_table))

# Removing stopwords
stop_words = set(stopwords.words('english'))
data['text1'] = data['text1'].apply(lambda text: ' '.join(word for word in text.split() if word not in stop_words))
data['text2'] = data['text2'].apply(lambda text: ' '.join(word for word in text.split() if word not in stop_words))

# Lemmatization
lemmatizer = WordNetLemmatizer()
data['text1'] = data['text1'].apply(lambda text: ' '.join(lemmatizer.lemmatize(word) for word in text.split()))
data['text2'] = data['text2'].apply(lambda text: ' '.join(lemmatizer.lemmatize(word) for word in text.split()))

# Example of how to use the preprocessed data with TF-IDF
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(pd.concat([data['text1'], data['text2']]))

data.to_csv(r"C:\Users\drash\OneDrive\Desktop\DataNeuron\Preprocessed_Data.csv", index=False)
