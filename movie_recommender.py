import chromadb
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from chromadb.utils import embedding_functions
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re

class MovieRecommender:
    def __init__(self, data_path):
        """
        Initialize the movie recommender system
        
        Args:
            data_path (str): Path to the MPST dataset CSV file
        """
        # Initialize ChromaDB
        self.chroma_client = chromadb.Client()
        
        # Create a collection with OpenAI embeddings
        self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        
        self.collection = self.chroma_client.create_collection(
            name="movie_plots",
            embedding_function=self.embedding_function
        )
        
        # Load and preprocess the dataset
        self.load_data(data_path)
        
        # Download required NLTK data
        nltk.download('punkt')
        nltk.download('stopwords')
        self.stop_words = set(stopwords.words('english'))

    def preprocess_text(self, text):
        """
        Preprocess the plot text by removing special characters, converting to lowercase,
        and removing stopwords
        
        Args:
            text (str): Raw plot text
            
        Returns:
            str: Preprocessed text
        """
        # Convert to lowercase and remove special characters
        text = re.sub(r'[^a-zA-Z\s]', '', text.lower())
        
        # Tokenize and remove stopwords
        tokens = word_tokenize(text)
        tokens = [token for token in tokens if token not in self.stop_words]
        
        return ' '.join(tokens)

    def load_data(self, data_path):
        """
        Load and process the MPST dataset
        
        Args:
            data_path (str): Path to the dataset
        """
        # Load the dataset
        self.df = pd.read_csv(data_path)
        
        # Preprocess plot summaries
        self.df['processed_plot'] = self.df['plot_synopsis'].apply(self.preprocess_text)
        
        # Add movies to ChromaDB collection
        self.collection.add(
            documents=self.df['processed_plot'].tolist(),
            metadatas=[{
                'title': title,
                'original_plot': plot,
                'tags': tags
            } for title, plot, tags in zip(
                self.df['movie_name'],
                self.df['plot_synopsis'],
                self.df['tags']
            )],
            ids=[str(i) for i in range(len(self.df))]
        )

    def get_recommendations(self, query_plot, n_recommendations=5):
        """
        Get movie recommendations based on a plot description
        
        Args:
            query_plot (str): Plot description to find similar movies
            n_recommendations (int): Number of recommendations to return
            
        Returns:
            list: List of dictionaries containing recommended movies and their details
        """
        # Preprocess the query plot
        processed_query = self.preprocess_text(query_plot)
        
        # Query the collection
        results = self.collection.query(
            query_texts=[processed_query],
            n_results=n_recommendations
        )
        
        # Format recommendations
        recommendations = []
        for i in range(len(results['metadatas'][0])):
            recommendations.append({
                'title': results['metadatas'][0][i]['title'],
                'plot': results['metadatas'][0][i]['original_plot'],
                'tags': results['metadatas'][0][i]['tags'],
                'similarity_score': results['distances'][0][i]
            })
        
        return recommendations

    def get_recommendations_by_title(self, movie_title, n_recommendations=5):
        """
        Get movie recommendations based on a movie title
        
        Args:
            movie_title (str): Title of the movie to find similar movies
            n_recommendations (int): Number of recommendations to return
            
        Returns:
            list: List of dictionaries containing recommended movies and their details
        """
        # Find the movie in the dataset
        movie = self.df[self.df['movie_name'].str.lower() == movie_title.lower()]
        
        if len(movie) == 0:
            return f"Movie '{movie_title}' not found in the dataset."
        
        # Get the plot of the movie
        plot = movie['plot_synopsis'].iloc[0]
        
        # Get recommendations based on the plot
        return self.get_recommendations(plot, n_recommendations)

    def search_by_tags(self, tags, n_recommendations=5):
        """
        Find movies with similar tags
        
        Args:
            tags (list): List of tags to search for
            n_recommendations (int): Number of recommendations to return
            
        Returns:
            list: List of movies with matching tags
        """
        matching_movies = []
        
        for _, row in self.df.iterrows():
            movie_tags = set(row['tags'].lower().split(','))
            search_tags = set(tag.lower() for tag in tags)
            
            if search_tags.intersection(movie_tags):
                matching_movies.append({
                    'title': row['movie_name'],
                    'plot': row['plot_synopsis'],
                    'tags': row['tags'],
                    'matching_tags': len(search_tags.intersection(movie_tags))
                })

        # Sort by number of matching tags and return top N
        matching_movies.sort(key=lambda x: x['matching_tags'], reverse=True)
        return matching_movies[:n_recommendations]

    def get_all_unique_tags(self):
        all_tags = set()
        for tags in self.df['tags']:
            all_tags.update([tag.strip() for tag in tags.split(',')])
        return sorted(list(all_tags))