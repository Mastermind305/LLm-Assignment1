# LLm-Assignment1




NLP Assignments Repository

This repository contains code and notebooks for various Natural Language Processing (NLP) assignments, including text preprocessing, word embeddings, and sequence-to-sequence models for text summarization.

Table of Contents





Overview



Assignments



Setup Instructions



Files



Usage



Results



Contributing



License

Overview

This repository showcases implementations of NLP techniques such as text preprocessing with Flask, word embedding visualization using TF-IDF and PCA, and a sequence-to-sequence model for text summarization using LSTM networks. The code is organized into different assignments as described below.

Assignments

Assignment 1.1





Files: nlp_api.py, index.html (first set)



Description: A Flask-based web application for text preprocessing tasks including tokenization, lemmatization, stemming, POS tagging, and named entity recognition using SpaCy and NLTK. Includes a comparison of lemmatization vs. stemming.

Assignment 1.2





Files: app.py, index.html (second set)



Description: A Flask application that visualizes word embeddings using TF-IDF and PCA. It provides an API to fetch word embeddings and their nearest neighbors based on cosine similarity, along with a plot of 2D embeddings.

Assignment 1.3





File: Assignment1_3 (1).ipynb



Description: A Jupyter notebook implementing a sequence-to-sequence model with LSTM for text summarization. The model is trained on a subset of the news summary dataset, with evaluation using ROUGE and BLEU scores.

Setup Instructions





Clone the Repository:

git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name



Set Up a Virtual Environment:

python -m venv flas1
source flas1/bin/activate  # On Windows: flas1\Scripts\activate



Install Dependencies:

pip install flask spacy nltk tensorflow keras sklearn numpy pandas matplotlib seaborn scikit-learn rouge-score



Download SpaCy Model:

python -m spacy download en_core_web_sm



Run the Applications:





For Assignment 1.1: python nlp_api.py



For Assignment 1.2: python app.py



For Assignment 1.3: Open Assignment1_3 (1).ipynb in Jupyter Notebook and run the cells.

Files





.gitignore: Specifies files and directories to ignore in Git.



README.md: This file.



nlp_api.py: Flask app for text preprocessing.



index.html (first set): HTML template for Assignment 1.1.



app.py: Flask app for word embedding visualization.



index.html (second set): HTML template for Assignment 1.2.



Assignment1_3 (1).ipynb: Jupyter notebook for text summarization.

Usage





Access the web interfaces by navigating to http://localhost:5000 after running the respective Python files.



For the notebook, follow the cell execution order to train the model and evaluate summaries.

Results





Assignment 1.1: Successfully preprocesses text and compares lemmatization and stemming.



Assignment 1.2: Generates a 2D PCA plot of word embeddings and provides nearest neighbor analysis.



Assignment 1.3: Achieves a ROUGE-1 F1 score of ~0.107, ROUGE-L F1 of ~0.102, and BLEU score of ~0.015 on the validation set.

Contributing

Contributions are welcome! Please submit a pull request or open an issue for any improvements or bug fixes.

License

This project is licensed under the MIT License. See the LICENSE file for details.

