from flask import Flask, request, jsonify, render_template
import spacy
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import nltk

# Download required NLTK data
nltk.download('punkt')

# Initialize app and models
app = Flask(__name__)
nlp = spacy.load("en_core_web_sm")
stemmer = PorterStemmer()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/preprocess', methods=['POST'])
def preprocess():
    data = request.get_json()
    text = data.get('text', '')

    if not text:
        return jsonify({"error": "Text is required."}), 400

    doc = nlp(text)

    tokens = [token.text for token in doc]
    lemmas = [token.lemma_ for token in doc]
    stems = [stemmer.stem(token) for token in tokens]
    pos_tags = [(token.text, token.pos_) for token in doc]
    entities = [(ent.text, ent.label_) for ent in doc.ents]

    return jsonify({
        'tokens': tokens,
        'lemmas': lemmas,
        'stems': stems,
        'pos_tags': pos_tags,
        'named_entities': entities
    })

@app.route('/compare_lemmatization_stemming', methods=['GET'])
def compare():
    examples = [
        "running", "better", "flies", "caresses", "ponies",
        "cats", "meeting", "lying", "dying", "fishing"
    ]
    comparisons = []
    for word in examples:
        lemma = nlp(word)[0].lemma_
        stem = stemmer.stem(word)
        comparisons.append({"word": word, "lemma": lemma, "stem": stem})

    return jsonify(comparisons)

if __name__ == '__main__':
    app.run(debug=True)
