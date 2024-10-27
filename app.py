# app.py
import streamlit as st
import pandas as pd
from movie_recommender import MovieRecommender
import plotly.express as px

class StreamlitMovieApp:
    def __init__(self):
        st.set_page_config(
            page_title="Movie Recommender",
            page_icon="🎬",
            layout="wide"
        )
        
        # Initialize the recommender
        if 'recommender' not in st.session_state:
            try:
                self.recommender = MovieRecommender('data/mpst_data.csv')
                st.session_state['recommender'] = self.recommender
            except Exception as e:
                st.error(f"Error loading data: {str(e)}")
                return
        else:
            self.recommender = st.session_state['recommender']

    def render_sidebar(self):
        st.sidebar.title("🎬 Navigation")
        return st.sidebar.radio(
            "Choose your search method:",
            ["Plot-based Search", "Title-based Search", "Tag-based Search"]
        )

    def plot_similarity_chart(self, recommendations):
        if recommendations:
            data = pd.DataFrame(recommendations)
            fig = px.bar(
                data,
                x='title',
                y='similarity_score',
                title='Similarity Scores',
                labels={'similarity_score': 'Similarity Score', 'title': 'Movie Title'},
                color='similarity_score',
                color_continuous_scale='blues'
            )
            fig.update_layout(yaxis_range=[0, 1])
            st.plotly_chart(fig)

    def display_recommendations(self, recommendations):
        if not recommendations:
            st.warning("No recommendations found.")
            return

        for i, movie in enumerate(recommendations, 1):
            with st.expander(f"#{i} {movie['title']} - {movie['similarity_score']:.2%} Match"):
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.markdown("**Plot Summary:**")
                    st.write(movie['plot'])
                
                with col2:
                    st.markdown("**Tags:**")
                    tags = movie['tags'].split(',')
                    for tag in tags:
                        st.markdown(f"`{tag.strip()}`")

    def render_plot_search(self):
        st.header("Search by Plot Description")
        plot_query = st.text_area(
            "Enter a plot description:",
            height=100,
            placeholder="E.g., A group of teenagers discovers they have supernatural powers..."
        )
        
        num_recommendations = st.slider(
            "Number of recommendations:",
            min_value=1,
            max_value=10,
            value=5
        )
        
        if st.button("Get Recommendations", key="plot_search"):
            if plot_query:
                with st.spinner("Finding similar movies..."):
                    recommendations = self.recommender.get_recommendations(
                        plot_query,
                        num_recommendations
                    )
                    self.plot_similarity_chart(recommendations)
                    self.display_recommendations(recommendations)
            else:
                st.warning("Please enter a plot description.")

    def render_title_search(self):
        st.header("Search by Movie Title")
        title_query = st.text_input(
            "Enter a movie title:",
            placeholder="E.g., The Matrix"
        )
        
        num_recommendations = st.slider(
            "Number of recommendations:",
            min_value=1,
            max_value=10,
            value=5
        )
        
        if st.button("Get Recommendations", key="title_search"):
            if title_query:
                with st.spinner("Finding similar movies..."):
                    recommendations = self.recommender.get_recommendations_by_title(
                        title_query,
                        num_recommendations
                    )
                    if isinstance(recommendations, str):
                        st.warning(recommendations)
                    else:
                        self.plot_similarity_chart(recommendations)
                        self.display_recommendations(recommendations)
            else:
                st.warning("Please enter a movie title.")

    def render_tag_search(self):
        st.header("Search by Tags")
        available_tags = self.recommender.get_all_unique_tags()
        selected_tags = st.multiselect(
            "Select tags:",
            options=available_tags,
            default=None
        )
        
        num_recommendations = st.slider(
            "Number of recommendations:",
            min_value=1,
            max_value=10,
            value=5
        )
        
        if st.button("Get Recommendations", key="tag_search"):
            if selected_tags:
                with st.spinner("Finding movies..."):
                    recommendations = self.recommender.search_by_tags(
                        selected_tags,
                        num_recommendations
                    )
                    self.display_recommendations(recommendations)
            else:
                st.warning("Please select at least one tag.")

    def main(self):
        st.title("🎬 Movie Recommendation System")
        
        search_method = self.render_sidebar()
        
        if search_method == "Plot-based Search":
            self.render_plot_search()
        elif search_method == "Title-based Search":
            self.render_title_search()
        else:
            self.render_tag_search()

if __name__ == "__main__":
    app = StreamlitMovieApp()
    app.main()