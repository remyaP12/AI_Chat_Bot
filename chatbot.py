import json
import pickle
import numpy as np
import random
from sklearn.metrics.pairwise import cosine_similarity

with open('model/vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

with open('model/encoder.pkl', 'rb') as f:
    encoder = pickle.load(f)

with open('model/responses.pkl', 'rb') as f:
    responses = pickle.load(f)

with open('data/intents.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

patterns = []
tags = []
for intent in data['intents']:
    for pattern in intent['patterns']:
        patterns.append(pattern.lower())
        tags.append(intent['tag'])

def get_response(user_input):
    if not patterns:
        return random.choice(responses['noanswer'])

    user_vec = vectorizer.transform([user_input.lower()])
    pattern_vecs = vectorizer.transform(patterns)

    similarities = cosine_similarity(user_vec, pattern_vecs)
    best_score = np.max(similarities)
    best_idx = np.argmax(similarities)
    best_tag = tags[best_idx]

    # ✅ Changed 0.2 to 0.1 — more flexible matching!
    if best_score < 0.1:
        return random.choice(responses['noanswer'])

    return random.choice(responses[best_tag])