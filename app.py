from flask import Flask, request, jsonify, send_from_directory
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import os

app = Flask(__name__)

model = SentenceTransformer('all-MiniLM-L6-v2')
faq_embeddings = {}

# Dynamic loader
def load_faq(category):
    file_map = {
        "registration": "CA_Registration_FAQs.xlsx",
        "student_enrollment": "student_enrollment_faq.xlsx",
        "skill": "skills_faq.xlsx",
        "exam": "exam_faq.xlsx"
    }
    file_name = file_map.get(category)
    if file_name and os.path.exists(file_name):
        df = pd.read_excel(file_name)
        df = df.dropna(subset=["Question", "Answer"])
        embeddings = model.encode(df["Question"].tolist())
        faq_embeddings[category] = (df, embeddings)
        return df, embeddings
    return pd.DataFrame(columns=["Question", "Answer"]), []

def find_best_answer(user_question, category):
    df, embeddings = faq_embeddings.get(category, (None, None))
    if df is None or embeddings is None or df.empty:
        df, embeddings = load_faq(category)
    if df.empty:
        return "No data available for this division."

    question_vec = model.encode([user_question])
    similarities = cosine_similarity(question_vec, embeddings)[0]
    best_idx = np.argmax(similarities)
    best_score = similarities[best_idx]

    if best_score > 0.6:
        return df.iloc[best_idx]["Answer"]
    else:
        top_indexes = np.argsort(similarities)[-5:][::-1]
        suggestions = df.iloc[top_indexes]["Question"].tolist()
        return "\n\nI'm not sure what you meant. Did you mean:\n- " + "\n- ".join(suggestions)

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json.get("message", "")
    category = request.json.get("category", "registration")
    reply = find_best_answer(user_input, category)
    return jsonify({"reply": reply})

@app.route('/')
def home():
    return send_from_directory('.', 'index.html')

if __name__ == "__main__":
    app.run()