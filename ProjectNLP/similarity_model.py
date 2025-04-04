# This is a model made by ChatGPT to test the similarity of
# two essays provided. This is not meant to be used to find the 
# actually score for the methods needed for the project. Just a secondary
# testing method to check my scores.

from sentence_transformers import SentenceTransformer, util
import pandas as pd

# Load your CSV
df = pd.read_csv("nlp_project_train.csv")

# Load the transformer model once (globally)
model = SentenceTransformer('all-MiniLM-L6-v2')

def compare_essays_by_id(id1, id2, df):
    # Lookup essays by ID
    essay1_row = df[df["Essay_ID"] == id1]
    essay2_row = df[df["Essay_ID"] == id2]

    if essay1_row.empty or essay2_row.empty:
        return f"One or both Essay IDs not found: {id1}, {id2}"

    essay1 = essay1_row["Essay_Text"].values[0]
    essay2 = essay2_row["Essay_Text"].values[0]

    # === SentenceTransformer similarity (THIS IS WHERE IT GOES) ===
    embeddings = model.encode([essay1, essay2])
    score = util.cos_sim(embeddings[0], embeddings[1]).item()

    # Extract first sentences
    first_sentence1 = essay1.strip().split('.')[0]
    first_sentence2 = essay2.strip().split('.')[0]

    # Output
    print(f"Essay {id1} vs Essay {id2}")
    print(f"Similarity Score: {score:.4f}")
    print(f"Essay 1: {first_sentence1}")
    print(f"Essay 2: {first_sentence2}")

compare_essays_by_id("29aa983", "6d25307", df)
