import pytest
from flask import Flask
from flask_wtf import FlaskForm
import pandas as pd

from app import app, search, filter_movies_based_on_preferences

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client


def test_index(client):
    response = client.get('/')
    assert response.status_code == 200
    # assert b"Welcome to the Index Page" in response.data

def test_search_no_query(client):
    response = client.get('/search')
    assert response.status_code == 200
    # assert b"query parameter is missing" in response.data

def test_search_with_query(client):
    response = client.get('/search?query=action')
    assert response.status_code == 200
    # Add your assertions here

def test_filter_movies_based_on_preferences():
    movies_df = pd.DataFrame({'Title': ['Movie 1', 'Movie 2'], 'Genre': ['Action', 'Comedy']})
    user_preferences = ['Action']
    filtered_movies = filter_movies_based_on_preferences(movies_df, user_preferences)
    assert len(filtered_movies) == 1
    assert filtered_movies.iloc[0]['Title'] == 'Movie 1'

    user_preferences = ['Comedy']
    filtered_movies = filter_movies_based_on_preferences(movies_df, user_preferences)
    assert len(filtered_movies) == 1
    assert filtered_movies.iloc[0]['Title'] == 'Movie 2'

    user_preferences = ['Drama']
    filtered_movies = filter_movies_based_on_preferences(movies_df, user_preferences)
    assert len(filtered_movies) == 0