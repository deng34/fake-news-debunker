import gradio as gr
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from openai import OpenAI
import os
import spacy
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
        probabilities = torch.softmax(logits, dim=1).squeeze()
        prediction_value = torch.argmax(probabilities).item()

    # Map the model's output to 'Fake' or 'Real'
    if prediction_value == 0:
        label = 'Fake'
    else:
        label = 'Real'

    # Get the probability for each class
    fake_prob = probabilities[0].item() * 100
    real_prob = probabilities[1].item() * 100

    # Extract keywords from the input text
    keywords = extract_keywords(text)

    return label, fake_prob, real_prob, keywords

# Main function that predicts and explains the results
def predict_and_explain(title, text):
    # Predict whether the news is real or fake, and extract keywords
    label, fake_prob, real_prob, keywords = predict(title, text)

    # Format keywords with line breaks after every 5 keywords
    formatted_keywords = []
    for i in range(0, len(keywords), 5):
        formatted_keywords.append(', '.join(keywords[i:i+5]))
    keywords_text = ',\n'.join(formatted_keywords)

    # If the news is classified as fake, generate suggestions
    if label == 'Fake':
        suggestions = generate_suggestions(title, text, keywords)
        return f"""
## 🔍 Analysis Results

**Prediction**: Fake News

**Probability**:
- Fake: {fake_prob:.2f}%
- Real: {real_prob:.2f}%

**Keywords**:
{keywords_text}

**Fact-Checking Suggestions**:
{suggestions}
"""
    else:
        # If the news is real, just show the prediction and keywords
        return f"""
## 🔍 Analysis Results

**Prediction**: Real News

**Probability**:
- Real: {real_prob:.2f}%
- Fake: {fake_prob:.2f}%

**Keywords**:
{keywords_text}
"""

# Function to generate suggestions for fact-checking
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
        # Use OpenAI GPT-4 API to generate suggestions using chat completion
        response = client.chat.completions.create(
            model="gpt-4",  # Use the GPT-4 model
            messages=[
                {"role": "system", "content": "You are a helpful assistant specialized in fact-checking."},
                {"role": "user", "content": prompt}  # Pass the constructed prompt as user input
            ],
            max_tokens=256,  # Set the maximum number of tokens
            temperature=0.7  # Control the diversity of the generated text
        )
        
        # Correctly access the generated suggestions from the API response
        suggestions = response.choices[0].message.content.strip()
    
    except Exception as e:
        # If there's an error, set a default error message and print the exception details for debugging
        suggestions = f"Unable to generate suggestions at this time. Error: {str(e)}"
        print(f"Error generating suggestions: {e}")  # Debug: print the error details to the console

    return suggestions

# Custom CSS styles
custom_css = """
.gr-interface {
    background-color: #f8f9fa;
}
.gr-form {
    background-color: white;
    padding: 2rem;
    border-radius: 1rem;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
}
.gr-input {
    border: 2px solid #e9ecef;
    border-radius: 0.5rem;
    transition: border-color 0.3s ease;
}
.gr-input:focus {
    border-color: #4a90e2;
    box-shadow: 0 0 0 2px rgba(74, 144, 226, 0.2);
}
.gr-button {
    background-color: #4a90e2;
    border: none;
    border-radius: 0.5rem;
    color: white;
    transition: background-color 0.3s ease;
}
.gr-button:hover {
    background-color: #357abd;
}
.footer {
    text-align: center;
    margin-top: 2rem;
    color: #6c757d;
}
"""

# Create custom theme
theme = gr.themes.Soft(
    primary_hue="blue",
    secondary_hue="gray",
    neutral_hue="gray",
    spacing_size=gr.themes.sizes.spacing_lg,
    radius_size=gr.themes.sizes.radius_lg,
    font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"]
).set(
    body_background_fill="#f8f9fa",
    body_background_fill_dark="#1a1b1e",
    button_primary_background_fill="#4a90e2",
    button_primary_background_fill_hover="#357abd",
    button_primary_text_color="white",
    input_background_fill="white",
    input_border_width="2px",
    input_shadow="0 2px 4px rgba(0,0,0,0.05)",
)

# Gradio interface setup
with gr.Blocks(theme=theme, css=custom_css) as iface:
    gr.Markdown(
        """
        # 🔍 Fake News Detection & Analysis System
        
        ### Your Tool for Identifying Misinformation and Finding Facts
        
        Enter a news article's title and content below to:
        - Analyze the authenticity of the news
        - Extract key topics and themes
        - Get fact-checking recommendations
        """
    )
    
    with gr.Row():
        with gr.Column():
            title_input = gr.Textbox(
                label="📰 News Title",
                placeholder="Enter the news title here...",
                lines=1
            )
            text_input = gr.Textbox(
                label="📝 News Content",
                placeholder="Enter the news content here...",
                lines=10
            )
            submit_btn = gr.Button("Analyze Now 🔍", variant="primary")
        
    output = gr.Markdown(label="Analysis Results")
    
    # Set submit button action
    submit_btn.click(
        fn=predict_and_explain,
        inputs=[title_input, text_input],
        outputs=output,
    )
    
    # Add footer
    gr.Markdown(
        """
        <div class="footer">
        💡 Note: This system uses advanced AI models for analysis. Results should be used as a reference. 
        Always maintain critical thinking and independent judgment.
        </div>
        """,
        visible=True
    )

# Launch the application
iface.launch()