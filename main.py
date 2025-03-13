from flask import Flask, render_template, request, jsonify
import requests
from bs4 import BeautifulSoup
import logging
from urllib.parse import urlparse
import os
from dotenv import load_dotenv
from transformers import pipeline
from sentence_transformers import SentenceTransformer
import numpy as np
import re

load_dotenv()

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)

logger.info("Loading QA models...")
primary_qa = pipeline('question-answering', model='deepset/roberta-base-squad2')
backup_qa = pipeline('question-answering', model='distilbert-base-cased-distilled-squad')

logger.info("Loading sentence transformer model...")
sentence_model = SentenceTransformer('all-MiniLM-L6-v2')

logger.info("Loading NER model...")
ner_pipeline = pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english")


def is_valid_url(url):
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except:
        return False


def clean_text(text):
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[\x00-\x1F\x7F-\x9F]', '', text)
    return text.strip()


def extract_article_content(url):
    if not is_valid_url(url):
        raise ValueError("Invalid URL provided")

    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }

        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, 'html.parser')

        title = soup.title.string if soup.title else "Unknown"

        for element in soup(["script", "style", "nav", "footer", "header", "aside", "iframe", "form"]):
            element.extract()

        main_content = None

        content_selectors = [
            "article", "main", "#article", ".article", ".post", ".content",
            "#content", ".story", ".entry-content", ".post-content"
        ]

        for selector in content_selectors:
            if main_content:
                break

            if selector.startswith(".") or selector.startswith("#"):
                main_content = soup.select_one(selector)
            else:
                main_content = soup.find(selector)

        if main_content:
            paragraphs = main_content.find_all('p')
        else:
            paragraphs = soup.find_all('p')

        text = ' '.join([p.get_text().strip() for p in paragraphs])

        if len(text.split()) < 100:
            text = soup.get_text(separator=' ', strip=True)

        authors = []
        author_selectors = [
            '.author', '.byline', '.writer', '[rel="author"]',
            '[itemprop="author"]', '.meta-author'
        ]

        for selector in author_selectors:
            author_elements = soup.select(selector)
            for element in author_elements:
                author_text = element.get_text().strip()
                if author_text and len(author_text) < 100:
                    authors.append(author_text)

        date = None
        date_selectors = [
            '[itemprop="datePublished"]', '.date', '.published',
            '[property="article:published_time"]', '.timestamp',
            '.meta-date', '.post-date'
        ]

        for selector in date_selectors:
            date_elements = soup.select(selector)
            if date_elements:
                date = date_elements[0].get_text().strip()
                break

        metadata = {
            "title": title,
            "authors": list(set(authors)),
            "publish_date": date
        }

        clean_content = clean_text(text)
        entities = extract_entities(clean_content)
        metadata["entities"] = entities

        return clean_content, metadata

    except Exception as e:
        logger.error(f"Error in content extraction: {e}")
        raise ValueError(f"Could not extract content from URL: {e}")


def extract_entities(text):
    """Extract named entities from text."""
    try:
        sample_text = text[:5000]
        entities_result = ner_pipeline(sample_text)

        people = []
        organizations = []
        locations = []
        dates = []

        current_entity = {"word": "", "type": ""}

        for entity in entities_result:
            if entity.get('entity', '').startswith('B-'):
                if current_entity["word"]:
                    if current_entity["type"] == "PER":
                        if current_entity["word"] not in people:
                            people.append(current_entity["word"])
                    elif current_entity["type"] == "ORG":
                        if current_entity["word"] not in organizations:
                            organizations.append(current_entity["word"])
                    elif current_entity["type"] == "LOC":
                        if current_entity["word"] not in locations:
                            locations.append(current_entity["word"])
                    elif current_entity["type"] == "MISC" and "date" in current_entity["word"].lower():
                        if current_entity["word"] not in dates:
                            dates.append(current_entity["word"])

                current_entity = {
                    "word": entity['word'],
                    "type": entity['entity'][2:]
                }

            elif entity.get('entity', '').startswith('I-'):
                current_entity["word"] += " " + entity['word']

        if current_entity["word"]:
            if current_entity["type"] == "PER":
                if current_entity["word"] not in people:
                    people.append(current_entity["word"])
            elif current_entity["type"] == "ORG":
                if current_entity["word"] not in organizations:
                    organizations.append(current_entity["word"])
            elif current_entity["type"] == "LOC":
                if current_entity["word"] not in locations:
                    locations.append(current_entity["word"])

        return {
            "people": people[:10],
            "organizations": organizations[:10],
            "locations": locations[:10],
            "dates": dates[:5]
        }

    except Exception as e:
        logger.error(f"Error in entity extraction: {e}")
        return {"people": [], "organizations": [], "locations": [], "dates": []}


def chunk_text(text, chunk_size=200, overlap=50):
    words = text.split()
    chunks = []

    for i in range(0, len(words), chunk_size - overlap):
        chunk = ' '.join(words[i:i + chunk_size])
        chunks.append(chunk)

        if i + chunk_size >= len(words):
            break

    return chunks


def answer_question_rag(article_content, question, metadata=None):
    try:
        enhanced_question = question
        if metadata and metadata.get("title"):
            enhanced_question = f"Title: {metadata.get('title')}. Question: {question}"

        chunks = chunk_text(article_content)

        if not chunks:
            return {
                'answer': "Could not process the article content.",
                'confidence': "0%",
                'success': False
            }

        if len(article_content.split()) < 500:
            try:
                result = primary_qa(question=question, context=article_content)
                confidence = result['score']
                return {
                    'answer': result['answer'],
                    'confidence': f"{confidence:.2%}",
                    'success': confidence > 0.3
                }
            except Exception as e:
                logger.error(f"Error with primary QA model: {e}")
                result = backup_qa(question=question, context=article_content)
                confidence = result['score']
                return {
                    'answer': result['answer'],
                    'confidence': f"{confidence:.2%}",
                    'success': confidence > 0.3
                }

        logger.info(f"Encoding {len(chunks)} chunks for semantic search")
        chunk_embeddings = sentence_model.encode(chunks)
        question_embedding = sentence_model.encode([enhanced_question])[0]

        scores = np.dot(chunk_embeddings, question_embedding)
        top_indices = np.argsort(scores)[-3:]  # Get top 3 chunks

        context = " ".join([chunks[i] for i in top_indices])

        try:
            result = primary_qa(question=question, context=context)
            confidence = result['score']

            if confidence < 0.3:
                backup_result = backup_qa(question=question, context=context)
                backup_confidence = backup_result['score']

                if backup_confidence > confidence:
                    result = backup_result
                    confidence = backup_confidence

            if confidence > 0.3:
                return {
                    'answer': result['answer'],
                    'confidence': f"{confidence:.2%}",
                    'success': True
                }
            else:
                best_answer = None
                best_confidence = 0
                window_size = 300
                stride = 150
                words = article_content.split()

                for i in range(0, len(words), stride):
                    window = ' '.join(words[i:i + window_size])
                    result = primary_qa(question=question, context=window)

                    if result['score'] > best_confidence:
                        best_confidence = result['score']
                        best_answer = result['answer']

                if best_confidence > 0.3:
                    return {
                        'answer': best_answer,
                        'confidence': f"{best_confidence:.2%}",
                        'success': True
                    }
                else:
                    return {
                        'answer': "Based on the article, I couldn't find a sufficiently confident answer to your question.",
                        'confidence': f"{confidence:.2%}",
                        'success': False
                    }

        except Exception as e:
            logger.error(f"Error during question answering: {e}")
            return {
                'answer': "An error occurred while processing the question.",
                'confidence': "0%",
                'success': False
            }

    except Exception as e:
        logger.error(f"Error in RAG question answering: {e}")
        return {
            'answer': "An error occurred while processing the question.",
            'confidence': "0%",
            'success': False
        }


@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template('index.html')


@app.route('/api/answer', methods=['POST'])
def api_answer():
    data = request.json
    if not data:
        return jsonify({'error': 'No data provided'}), 400

    article_url = data.get('article_url')
    question = data.get('question')

    if not article_url or not question:
        return jsonify({'error': 'Missing article URL or question'}), 400

    try:
        article_content, metadata = extract_article_content(article_url)
        result = answer_question_rag(article_content, question, metadata)

        result['metadata'] = metadata
        return jsonify(result)
    except Exception as e:
        logger.error(f"Error processing request: {e}")
        return jsonify({'error': str(e), 'success': False}), 500


@app.route('/process', methods=['POST'])
def process():
    article_url = request.form.get('article_url')
    question = request.form.get('question')

    if not article_url or not question:
        return render_template('index.html', error="Please provide both an article URL and a question")

    try:
        article_content, metadata = extract_article_content(article_url)
        result = answer_question_rag(article_content, question, metadata)

        return render_template('index.html',
                               article_url=article_url,
                               question=question,
                               answer=result['answer'],
                               confidence=result['confidence'],
                               metadata=metadata)
    except Exception as e:
        logger.error(f"Error in process route: {e}")
        return render_template('index.html', error=f"Error: {str(e)}")


if __name__ == '__main__':
    print("Starting server...")
    port = int(os.environ.get('PORT', 3000))
    debug = os.environ.get('FLASK_ENV', 'development') == 'development'

    app.run(host='0.0.0.0', port=port, debug=debug)