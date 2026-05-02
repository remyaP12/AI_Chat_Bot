# import json
# import pickle
# import numpy as np
# import nltk
# from nltk.stem import LancasterStemmer
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.preprocessing import LabelEncoder

# nltk.download('punkt_tab')

# stemmer = LancasterStemmer()

# with open('data/intents.json', 'r') as f:
#     data = json.load(f)

# patterns = []
# tags = []
# responses = {}

# for intent in data['intents']:
#     responses[intent['tag']] = intent['responses']
#     for pattern in intent['patterns']:
#         patterns.append(pattern.lower())
#         tags.append(intent['tag'])

# # Vectorize using TF-IDF
# vectorizer = TfidfVectorizer()
# X = vectorizer.fit_transform(patterns)

# # Encode labels
# encoder = LabelEncoder()
# y = encoder.fit_transform(tags)

# # Save trained model components
# with open('model/vectorizer.pkl', 'wb') as f:
#     pickle.dump(vectorizer, f)

# with open('model/encoder.pkl', 'wb') as f:
#     pickle.dump(encoder, f)

# with open('model/responses.pkl', 'wb') as f:
#     pickle.dump(responses, f)

# print("✅ Model trained and saved successfully!")
# print(f"✅ Total patterns trained: {len(patterns)}")
# print(f"✅ Total intents: {len(set(tags))}")

import json
import pickle
import numpy as np
import nltk
from nltk.stem import LancasterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

nltk.download('punkt_tab')
stemmer = LancasterStemmer()

# ✅ Fixed — added encoding='utf-8'
with open('data/intents.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

patterns = []
tags = []
responses = {}

for intent in data['intents']:
    responses[intent['tag']] = intent['responses']
    for pattern in intent['patterns']:
        patterns.append(pattern.lower())
        tags.append(intent['tag'])

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(patterns)

encoder = LabelEncoder()
y = encoder.fit_transform(tags)

with open('model/vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

with open('model/encoder.pkl', 'wb') as f:
    pickle.dump(encoder, f)

with open('model/responses.pkl', 'wb') as f:
    pickle.dump(responses, f)

print("✅ Model trained and saved successfully!")
print(f"✅ Total patterns trained: {len(patterns)}")
print(f"✅ Total intents: {len(set(tags))}")