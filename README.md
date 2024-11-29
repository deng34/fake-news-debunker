# Fake News Debunker

An AI-powered tool that helps detect and analyze potential fake news articles using advanced natural language processing techniques.

## Features

- Real-time fake news detection
- Keyword extraction and analysis
- Confidence scoring for predictions
- AI-powered fact-checking suggestions
- User-friendly interface built with Gradio

## Tech Stack

- Python 3.x
- PyTorch
- Transformers (Hugging Face)
- Gradio
- spaCy
- OpenAI GPT-4
- scikit-learn
- pandas

## Prerequisites

- Python 3.x
- Git LFS (for handling large files)
- OpenAI API key

## Installation

```bash
git clone https://github.com/yourusername/fake-news-debunker.git
cd fake-news-debunker
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

Set your OpenAI API key as an environment variable:
```bash
export OPENAI_API_KEY='your-api-key-here'
```

## Usage

1. Start the application:
```bash
python app.py
```

2. Open your browser and navigate to the local address shown in the terminal
3. Enter the news title and content
4. Click "Analyze Now" to get the results

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

- RoBERTa model fine-tuned by [Hamza Benyamina](https://huggingface.co/hamzab/roberta-fake-news-classification)
- Built with [Gradio](https://gradio.app/)
- Powered by [Hugging Face Transformers](https://huggingface.co/transformers/)
