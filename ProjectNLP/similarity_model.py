# This is a model made by ChatGPT to test the similarity of
# two essays provided. This is not meant to be used to find the 
# actually score for the methods needed for the project. Just a secondary
# testing method to check my scores.

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer, util

# Load the dataset
df = pd.read_csv("nlp_project_train.csv")

model = SentenceTransformer('all-MiniLM-L6-v2')

def compare_essays_by_id(id1, id2, df):
    essay1_row = df[df["Essay_ID"] == id1]
    essay2_row = df[df["Essay_ID"] == id2]

    if essay1_row.empty or essay2_row.empty:
        return f"One or both Essay IDs not found: {id1}, {id2}"

    essay1 = essay1_row["Essay_Text"].values[0]
    essay2 = essay2_row["Essay_Text"].values[0]

    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([essay1, essay2])
    score = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]

    first_sentence1 = essay1.strip().split('.')[0]
    first_sentence2 = essay2.strip().split('.')[0]

    print(f"Essay {id1} vs Essay {id2}")
    print(f"Similarity Score: {score:.4f}")
    print(f"Essay 1: {first_sentence1}")
    print(f"Essay 2: {first_sentence2}")

# Example usage:
compare_essays_by_id("29aa983", "6d25307", df)
