from flask import Flask, render_template, request, jsonify
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string
import re
import PyPDF2

app = Flask(__name__)

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Initialize lemmatizer and stopwords
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Function to read corpus from PDF
def read_corpus_from_pdf(file_path):
    with open(file_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        corpus = ""
        for page in pdf_reader.pages:
            corpus += page.extract_text()
    return corpus

# Read corpus from PDF file
CORPUS = read_corpus_from_pdf('corpus.pdf')

# Preprocess the corpus
def preprocess_text(text):
    # Tokenize and lowercase
    tokens = word_tokenize(text.lower())
    # Remove punctuation and stopwords, then lemmatize
    tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in string.punctuation and token not in stop_words]
    return tokens

corpus_tokens = preprocess_text(CORPUS)

# Function to find the most relevant sentence in the corpus
def find_most_relevant_sentence(query):
    query_tokens = preprocess_text(query)
    
    max_overlap = 0
    most_relevant_sentence = "I'm sorry, I don't have information about that. Please contact the business directly for more details."
    
    for sentence in re.split(r'(?<=[.!?])\s+', CORPUS):
        sentence_tokens = preprocess_text(sentence)
        overlap = len(set(query_tokens) & set(sentence_tokens))
        
        if overlap > max_overlap:
            max_overlap = overlap
            most_relevant_sentence = sentence
    
    return most_relevant_sentence

conversation_history = []

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json['message']
    
    # Check for greeting
    if user_input.lower() in ['hi', 'hello']:
        response = "Welcome to Jessup Cellars! We're delighted you're here. How may we assist you today with our wines, tasting experiences, or visiting our Yountville location?"
    else:
        response = find_most_relevant_sentence(user_input)
    
    conversation_history.append(f"Human: {user_input}")
    conversation_history.append(f"AI: {response}")
    
    if len(conversation_history) > 10:
        conversation_history.pop(0)
        conversation_history.pop(0)
    
    return jsonify({'response': response})

if __name__ == '__main__':
    app.run(debug=True)