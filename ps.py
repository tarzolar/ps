# -*- coding: utf-8 -*-
"""
Created on Fri Jan 24 13:02:31 2025

@author: Tomas Arzola Röber
"""

import os
import re
from pdfminer.high_level import extract_text
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
import nltk

# Download necessary NLTK data (only the first time)
# nltk.download('punkt')
# nltk.download('stopwords')

# Define the categories for the 9 papers
categories = {
    "Bilal Känzig 2024 _The macroeconomic impact of climate change - Global vs local temperature_.pdf": "Macroeconomics",
    "Kalkuhl Wenz 2022 _The impact of climate conditions on economic production. Evidence from a global panel of regions_.pdf": "Macroeconomics",
    "Kotz Levermann Wenz 2024 _The economic commitment of climate change_.pdf": "Macroeconomics",
    "Castells-Quintana_Dienesch_2021.pdf": "Air Pollution",
    "Chang_2016.pdf": "Air Pollution",
    "Feng_2024.pdf": "Air Pollution",
    "acp-20-2303-2020.pdf": "Earth System Modeling",
    "J Adv Model Earth Syst - 2023 - Fu - Using Convolutional Neural Network to Emulate Seasonal Tropical Cyclone Activity.pdf": "Earth System Modeling",
    "Reviews of Geophysics - 2019 - Bellouin - Bounding Global Aerosol Radiative Forcing of Climate Change.pdf": "Earth System Modeling",
    "Bajari-MachineLearningMethods-2015.pdf": "Machine Learning",
    "cesifo1_wp6504.pdf": "Behavioral Economics",
    "SSRN-id3567724.pdf": "Behavioral Economics"
}

# Specify the folder containing the PDFs
pdf_folder = 'data'
pdf_files = [f for f in os.listdir(pdf_folder) if f.endswith('.pdf')]
a = 0
# Extract text from PDFs
def extract_texts_from_pdfs(pdf_folder, pdf_files):
    texts = {}
    for pdf in pdf_files:
        file_path = os.path.join(pdf_folder, pdf)
        texts[pdf] = extract_text(file_path)
    return texts

# Preprocess the text
def preprocess_text(text):
    sentences = sent_tokenize(text)  # Split into sentences
    stop_words = set(stopwords.words('english'))
    stop_words.update(['introduction', 'background', 'method', 'methods', 'conclusion', 'discussion', 'results', 'findings', 'analysis', 'study',
                    'paper', 'article','et', 'al', 'ibid', 'op', 'cit', 'reference', 'figure', 'table', 'approach', 'framework',
                    'perspective', 'context', 'process', 'system', 'issue', 'problem', 'challenge', 'http', 'doi', 'org', 'acp', 
                    'author', 'work', 'dx', 'pol', 'sci', 'vol', 'pp', 'www', 'com', 'net', 'html', 'pdf', 'journal', 'research',
                    'content', 'contents', 'list', 'lists', 'available', 'sciencedirect', 'elsevier', 'rights', 'reserved', 'permission',
                    'print', 'online', 'original', 'article', 'review', 'received', 'revised', 'accepted', 'abstract', 'keywords', 'version',
                    'fabian', 'andreas', 'barth', 'sasan', 'mansouri', 'woebbeking', 'april', 'may', 'june', 'july', 'august', 'september',
                    'october', 'november', 'december', 'january', 'february', 'march', 'diego', 'känzig', 'nber', 'impressum', 'cesifo', 'working',
                    'arno', 'riedl', 'roberto', 'weber', 'issn', 'electronic', 'publisher', 'distributor', 'homepage', 'tandfonline'])

    cleaned_sentences = []
    for sentence in sentences:
        # Remove punctuation, numbers, and special characters
        sentence = re.sub(r'[^a-zA-Z\s]', '', sentence)
        # Convert to lowercase
        sentence = sentence.lower()
        # Remove stop words
        sentence = ' '.join(word for word in sentence.split() if word not in stop_words)
        cleaned_sentences.append(sentence)
    return cleaned_sentences

# Preprocess all extracted texts
def preprocess_all_texts(texts):
    preprocessed_texts = {}
    for pdf, text in texts.items():
        preprocessed_texts[pdf] = preprocess_text(text)
    return preprocessed_texts

# Create TF-IDF embeddings
def create_tfidf_embeddings(preprocessed_texts):
    all_sentences = [sentence for sentences in preprocessed_texts.values() for sentence in sentences]
    vectorizer = TfidfVectorizer(max_features=2500)  # Adjust max_features as needed
    tfidf_matrix = vectorizer.fit_transform(all_sentences)
    return tfidf_matrix, all_sentences

# Perform t-SNE
def perform_tsne(tfidf_matrix, perplexity_value):
    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity_value, n_iter=2500)
    embeddings = tsne.fit_transform(tfidf_matrix.toarray())
    return embeddings

# Visualize t-SNE results with categories
def visualize_tsne_with_categories(embeddings, preprocessed_texts, categories):
    plt.figure(figsize=(12, 8))
    labels = []
    
    # Ensure that the labels are correctly assigned without "Unknown"
    for pdf, sentences in preprocessed_texts.items():
        # Check if the paper is in the categories dictionary
        if pdf in categories:
            category = categories[pdf]
        else:
            category = "Unknown"  # Should not happen, but it helps debugging

        labels.extend([category] * len(sentences))

    unique_categories = list(set(labels))
    colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_categories)))
    color_map = {category: colors[i] for i, category in enumerate(unique_categories)}

    for category in unique_categories:
        indices = [i for i, l in enumerate(labels) if l == category]
        plt.scatter(embeddings[indices, 0], embeddings[indices, 1], label=category, c=[color_map[category]], alpha=0.4)

    plt.legend()
    plt.title("t-SNE Visualization of Sentences by Category")
    plt.show()

# Main Workflow
if __name__ == "__main__":
    # Extract texts from PDFs
    texts = extract_texts_from_pdfs(pdf_folder, pdf_files)
    
    # Preprocess the texts
    preprocessed_texts = preprocess_all_texts(texts)
    
    # Create TF-IDF embeddings
    tfidf_matrix, all_sentences = create_tfidf_embeddings(preprocessed_texts)
    
    # Compute perplexity based on the number of sentences
    perplexity_value = min(10, len(all_sentences) - 1)
    
    # Perform t-SNE with dynamic perplexity
    embeddings = perform_tsne(tfidf_matrix, perplexity_value)
    
    # Visualize the t-SNE results
    visualize_tsne_with_categories(embeddings, preprocessed_texts, categories)
