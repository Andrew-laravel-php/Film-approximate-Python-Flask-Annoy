import pytest
from flask import Flask

app = Flask(__name__)


@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client


def test_home_page(client):
    response = client.get('/')
    assert response.status_code == 200
    assert b"Welcome to my Flask app" in response.data


def test_search_endpoint_with_query(client):
    response = client.get('/search?query=action')
    assert response.status_code == 200
    assert b"Title" in response.data
    assert b"Poster" in response.data
    assert b"Distance" in response.data
    assert b"Color" in response.data


def test_search_endpoint_without_query(client):
    response = client.get('/search')
    assert response.status_code == 200
    assert b"error" in response.data


def test_search_endpoint_with_preferences(client):
    response = client.get('/search?query=action&preferences=action&preferences=adventure')
    assert response.status_code == 200
    assert b"Title" in response.data
    assert b"Poster" in response.data
    assert b"Distance" in response.data
    assert b"Color" in response.data


def test_search_endpoint_with_invalid_preferences(client):
    response = client.get('/search?query=action&preferences=invalid_genre')
    assert response.status_code == 200
    assert b"message" in response.data


def test_search_by_category_endpoint_with_genre(client):
    response = client.get('/searchbycategory?genre=action')
    assert response.status_code == 200
    assert b"search_by_category.html" in response.data


def test_search_by_category_endpoint_with_genre_and_rating(client):
    response = client.get('/searchbycategory?genre=action&rating=7')
    assert response.status_code == 200
    assert b"search_by_category.html" in response.data


def test_search_by_category_endpoint_without_parameters(client):
    response = client.get('/searchbycategory')
    assert response.status_code == 200
    assert b"search_by_category.html" in response.data


if __name__ == '__main__':
    pytest.main()
