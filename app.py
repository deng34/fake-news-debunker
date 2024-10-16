import gradio as gr
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from openai import OpenAI
import os
import spacy
import subprocess
import sys
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# Set OpenAI API key from environment variables
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Load the tokenizer and the pretrained classification model
tokenizer = AutoTokenizer.from_pretrained("hamzab/roberta-fake-news-classification")
model = AutoModelForSequenceClassification.from_pretrained("hamzab/roberta-fake-news-classification")

# Load spaCy model for keyword extraction
import spacy.cli

try:
    nlp = spacy.load('en_core_web_sm')
except OSError:
    # If spaCy model is not available, download it
    spacy.cli.download("en_core_web_sm")
    nlp = spacy.load('en_core_web_sm')


# Load the WELFake dataset and extract top 500 TF-IDF keywords
def load_data():
    # Load WELFake dataset from CSV file
    wel_fake_data = pd.read_csv('WELFake_Dataset.csv')
    wel_fake_data.dropna(subset=['text'], inplace=True)  # Remove rows with missing 'text'

    # Create a TF-IDF vectorizer and fit it on the dataset's text column
    vectorizer = TfidfVectorizer(max_features=500, stop_words='english')
    X = vectorizer.fit_transform(wel_fake_data['text'])

    # Get the top 500 keywords from the dataset
    top_keywords = vectorizer.get_feature_names_out()
    return top_keywords


# Load top TF-IDF keywords from the WELFake dataset
top_keywords = load_data()


# Function to extract keywords using spaCy and matching them with TF-IDF keywords
def extract_keywords(text):
    # Use spaCy to extract keywords (nouns and proper nouns)
    doc = nlp(text)
    spacy_keywords = [token.text for token in doc if
                      token.is_alpha and not token.is_stop and token.pos_ in ['NOUN', 'PROPN']]

    # Use TF-IDF to match keywords in the input text with the top keywords from the dataset
    tfidf_keywords = [kw for kw in top_keywords if kw.lower() in text.lower()]

    # Combine the keywords from both sources and remove duplicates
    all_keywords = list(set(spacy_keywords + tfidf_keywords))

    return all_keywords


# Function to predict whether the news is real or fake using the classification model
def predict(title, text):
    # Combine the title and text as input to the model
    input_text = title + " " + text

    # Tokenize the input and prepare it for the model
    inputs = tokenizer.encode_plus(
        input_text,
        add_special_tokens=True,
        max_length=512,
        truncation=True,
        padding='max_length',
        return_tensors="pt"
    )

    # Set the model to evaluation mode
    model.eval()

    # Perform the prediction using the model
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=1)
        prediction_value = torch.argmax(probabilities, dim=1).item()

    # Map the model's output to 'Fake' or 'Real'
    if prediction_value == 0:
        label = 'Fake'
    else:
        label = 'Real'

    # Extract keywords from the input text
    keywords = extract_keywords(text)

    return label, keywords


def generate_suggestions(title, text, keywords):
    # Construct the prompt for GPT based on the title, text, and keywords
    prompt = f"""
    You are a specialist in fact-checking. Based on the title, text, and keywords of the fake news, 
    please suggest some ways to know more about the facts. Please give recommendations that are easy to accept.

    Keywords: {', '.join(keywords)}
    Title: {title}
    Text: {text}
    """

    try:
        # Call OpenAI's chat completion method using GPT-4 model
        response = client.chat.completions.create(
            model="gpt-4",  # Using the GPT-4 model
            messages=[
                {"role": "system", "content": "You are a helpful assistant specialized in fact-checking."},  # System role definition
                {"role": "user", "content": prompt}  # User input (the constructed prompt)
            ],
            max_tokens=4000,  # Set the maximum token limit to 4000
            temperature=0.7,  # Controls the randomness in the generated text
        )
        
        # Extract and clean the suggestions from the API response
        suggestions = response.choices[0].message["content"].strip()

    except Exception as e:
        # If there's an error, set a default error message and print the exception details for debugging
        suggestions = "Unable to generate suggestions at this time."
        print(f"Error generating suggestions: {e}")  # Debug: print the error details to the console

    return suggestions


# Main function that predicts and explains the results
def predict_and_explain(title, text):
    # Predict whether the news is real or fake, and extract keywords
    label, keywords = predict(title, text)

    # If the news is classified as fake, generate suggestions
    if label == 'Fake':
        suggestions = generate_suggestions(title, text, keywords)
        return f"""
**Prediction**: Fake News

**Keywords**: {', '.join(keywords)}

**Suggestions**:
{suggestions}
"""
    else:
        # If the news is real, just show the prediction and keywords
        return f"""
**Prediction**: Real News

**Keywords**: {', '.join(keywords)}
"""


# Gradio interface setup
iface = gr.Interface(
    fn=predict_and_explain,  # The function to handle user input and return predictions
    inputs=[
        gr.Textbox(label="Title"),  # Textbox for the news title
        gr.Textbox(label="Text", lines=10)  # Textbox for the news content
    ],
    outputs="markdown",  # Output format is markdown
    title="Fake News Detector with Suggestions",  # Title of the Gradio app
    description="Enter the news title and content to check if it's fake. If fake, get suggestions on how to know more about the facts.",
    # Description of the app
)

# Launch the Gradio app
iface.launch()
