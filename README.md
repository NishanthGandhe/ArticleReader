# Article Question Answering System
A powerful web application that allows users to extract information from online articles by asking natural language questions. The system uses advanced NLP and transformer models to understand article content and provide accurate answers.

## ✨ Features

- **Smart Content Extraction**: Automatically extracts the main content from any URL
- **Intelligent Question Answering**: Uses state-of-the-art transformer models to answer questions about the article
- **Entity Recognition**: Identifies people, organizations, and locations mentioned in the article
- **Semantic Search**: Finds the most relevant parts of the article to answer your question
- **Confidence Scores**: Provides transparency about answer reliability with confidence metrics
- **REST API**: Easy integration with other applications

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- pip

### Installation

1. Clone the repository
```bash
git clone https://github.com/yourusername/article-qa-system.git
cd article-qa-system
```

2. Create a virtual environment (recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies
```bash
pip install -r requirements.txt
```

4. Run the application
```bash
python main.py
```

5. Open your browser and navigate to `http://localhost:3000`

## 📚 How It Works

1. **Content Extraction**  
   The system uses BeautifulSoup to extract article content, filtering out navigation elements, ads, and other non-content sections.

2. **Text Chunking**  
   Long articles are split into manageable chunks with overlap to ensure context is maintained.

3. **Semantic Search**  
   When a question is asked, the system uses sentence embeddings to find the most relevant chunks of the article.

4. **Question Answering**  
   The top chunks are passed to transformer-based question answering models that extract the precise answer.

5. **Confidence Evaluation**  
   The system evaluates the confidence of the answer and only returns it if it meets a threshold.

## 🛠️ Technologies Used

- **Flask**: Web framework
- **Transformers (Hugging Face)**: Question answering models and NER
- **SentenceTransformers**: Semantic search capabilities
- **BeautifulSoup**: Web content extraction
- **NumPy**: Numerical operations for similarity scores
- **Python-dotenv**: Environment variable management

## 🧠 Models Used

- **deepset/roberta-base-squad2**: Primary question answering model
- **distilbert-base-cased-distilled-squad**: Backup question answering model
- **all-MiniLM-L6-v2**: Sentence transformer for semantic search
- **dbmdz/bert-large-cased-finetuned-conll03-english**: Named entity recognition

## 📋 API Usage

The system provides a simple REST API for integration:

```json
POST /api/answer
Content-Type: application/json

{
  "article_url": "https://example.com/article",
  "question": "Who is the author of the article?"
}
```

Response:
```json
{
  "answer": "Jane Doe",
  "confidence": "92.45%",
  "success": true,
  "metadata": {
    "title": "Example Article",
    "authors": ["Jane Doe"],
    "publish_date": "2025-01-15",
    "entities": {
      "people": ["Jane Doe", "John Smith"],
      "organizations": ["Example Corp", "Research Institute"],
      "locations": ["New York", "London"]
    }
  }
}
```

## 🔧 Configuration

You can customize the application behavior by setting environment variables in a `.env` file:

```
PORT=3000
FLASK_ENV=development
```

## 📊 Performance Considerations

- The first request may be slow as models are loaded into memory
- Large articles may take longer to process
- Complex questions that require reasoning beyond the explicit text may have lower confidence scores

## 🔍 Limitations

- Works best with well-structured articles
- May struggle with heavily JavaScript-rendered content
- Answers are limited to information explicitly stated in the article
- Performance depends on the quality of the content extraction
