from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')

app = Flask(__name__)

# Sample custom corpus
corpus = [
    "The sky is blue and beautiful",
    "Love this blue and beautiful sky!",
    "The quick brown fox jumps over the lazy dog",
    "A king's breakfast has sausages, ham, and bacon",
    "I love green eggs, ham, sausages, and bacon!",
    "The brown fox is quick and the blue dog is lazy!",
    "The sky is very blue and the sky is very beautiful today",
    "The dog is lazy but the brown fox is quick"
]

stop_words = set(stopwords.words('english'))

# TF-IDF vectorizer
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(corpus)
features = vectorizer.get_feature_names_out()
embedding_matrix = X.toarray().T  # Transpose to make it word x doc

# Dimensionality reduction
def reduce_dimensions(data, method='pca'):
    if method == 'tsne':
        return TSNE(n_components=2, random_state=42).fit_transform(data)
    else:
        return PCA(n_components=2).fit_transform(data)

reduced = reduce_dimensions(embedding_matrix, method='pca')
word_positions = pd.DataFrame(reduced, columns=['x', 'y'])
word_positions['word'] = features

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/embedding', methods=['GET'])
def get_embedding():
    word = request.args.get('word')
    if word not in features:
        return jsonify({"error": f"'{word}' not found in corpus vocabulary."}), 404

    idx = list(features).index(word)
    vector = embedding_matrix[idx]

    # Cosine similarity for nearest neighbors
    similarities = cosine_similarity([vector], embedding_matrix)[0]
    top_indices = similarities.argsort()[-6:][::-1]  # Top 5 + itself
    neighbors = [(features[i], round(similarities[i], 3)) for i in top_indices if i != idx]

    return jsonify({
        "word": word,
        "embedding": vector.tolist(),
        "neighbors": neighbors
    })

@app.route('/api/plot')
def plot_words():
    plt.figure(figsize=(10, 7))
    sns.scatterplot(x='x', y='y', data=word_positions)
    for i in range(len(word_positions)):
        plt.text(word_positions.iloc[i]['x'] + 0.01, word_positions.iloc[i]['y'] + 0.01,
                 word_positions.iloc[i]['word'], fontsize=9)
    plt.title("2D Word Embeddings (PCA)")

    # Ensure static directory exists
    static_dir = os.path.join(os.getcwd(), 'static')
    if not os.path.exists(static_dir):
        os.makedirs(static_dir)

    plot_path = os.path.join('static', 'plot.png')
    plt.savefig(plot_path)
    plt.close()

    return jsonify({"img_path": plot_path})

if __name__ == '__main__':
    app.run(debug=True)
