# 🎬 Movie Recommendation System

A sophisticated movie recommendation system that uses ChromaDB and sentence transformers to suggest movies based on plot similarities, titles, and tags. This project demonstrates the power of semantic search in content-based recommendation systems.

## 🎯 Project Overview

This project implements a movie recommendation system using the MPST (Movie Plot Synopses with Tags) dataset. It leverages ChromaDB for efficient similarity search and embedding storage, along with sentence transformers for generating meaningful text embeddings.

### Key Features

- **Multiple Search Methods:**
  - 🔍 Plot-based recommendations using semantic similarity
  - 📚 Title-based movie recommendations
  - 🏷️ Tag-based movie search
  
- **Advanced Technology Stack:**
  - ChromaDB for vector similarity search
  - Sentence Transformers for text embeddings
  - NLTK for text preprocessing
  - Streamlit for interactive web interface
  - Plotly for visualization

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- Git
- The MPST dataset

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/movie-recommender.git
cd movie-recommender
```

2. Create and activate a virtual environment (optional but recommended):
```bash
python -m venv venv
# On Windows
venv\Scripts\activate
# On Unix or MacOS
source venv/bin/activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

4. Download the MPST dataset and place it in the `data` folder:
```bash
mkdir data
# Place mpst_data.csv in the data folder
```

### Running the Application

1. Start the Streamlit app:
```bash
streamlit run app.py
```

2. Open your browser and navigate to:
```
http://localhost:8501
```

## 🔧 Technical Architecture

### ChromaDB Integration

The project uses ChromaDB as its vector database for efficient similarity search:

1. **Embedding Generation:**
   - Utilizes the `all-MiniLM-L6-v2` model for generating embeddings
   - Processes movie plots through NLTK for better text representation

2. **Vector Storage:**
   - Movie plots are converted to embeddings and stored in ChromaDB
   - Metadata (title, original plot, tags) is preserved alongside embeddings

3. **Similarity Search:**
   - ChromaDB handles efficient nearest neighbor search
   - Returns similarity scores for recommendations

```python
# Example of ChromaDB initialization in the code
self.chroma_client = chromadb.Client()
self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)
self.collection = self.chroma_client.create_collection(
    name="movie_plots",
    embedding_function=self.embedding_function
)
```

## 📊 Features in Detail

1. **Plot-Based Search:**
   - Enter a plot description
   - System finds movies with similar plot elements
   - Visualizes similarity scores

2. **Title-Based Search:**
   - Search by movie title
   - Get recommendations based on plot similarity
   - View detailed movie information

3. **Tag-Based Search:**
   - Select multiple tags
   - Find movies matching selected categories
   - Sort by tag relevance

## 🖥️ User Interface

The Streamlit interface provides:
- Interactive sidebar navigation
- Similarity score visualizations
- Expandable movie details
- Tag filtering system
- Adjustable number of recommendations

## 📁 Project Structure

```
movie-recommender/
├── app.py                 # Streamlit interface
├── movie_recommender.py   # Core recommendation engine
├── requirements.txt       # Project dependencies
├── README.md             # Project documentation
└── data/
    └── mpst_data.csv     # Movie dataset
```

