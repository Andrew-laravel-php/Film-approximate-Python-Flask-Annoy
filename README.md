# Film-approximate-Python-Flask-Annoy
 Film-approximate-Python-Flask-Annoy
Для написания поиска ближайших соседей с помощью библиотеки Annoy для сайта с фильмами можно использовать язык программирования Python.

Для начала необходимо установить библиотеку Annoy и ее зависимости. Для этого можно использовать менеджер пакетов pip:

pip install annoy

Затем нужно загрузить данные фильмов и преобразовать их в векторный формат. Для этого можно использовать алгоритм word2vec или другие подходы к векторизации текста.

После этого можно создать объект AnnoyIndex и добавить в него векторы фильмов:
import random
from annoy import AnnoyIndex

# Create AnnoyIndex object
annoy_index = AnnoyIndex(100)

# Add vectors to index
for i in range(len(movie_vectors)):
    annoy_index.add_item(i, movie_vectors[i])

# Build the index
annoy_index.build(50)


Здесь 100 - размерность векторов фильмов, а 50 - число деревьев в построении индекса.

После этого можно использовать метод get_nns_by_vector для поиска ближайших соседей по вектору фильма:

# Get indices of nearest neighbors
nearest_neighbors = annoy_index.get_nns_by_vector(movie_vector, 10)

Здесь movie_vector - вектор искомого фильма, а 10 - число ближайших соседей, которые нужно найти.

Для использования Annoy на сайте можно написать API-метод, который будет принимать запросы на поиск ближайших фильмов и возвращать результаты в формате JSON. Например:

from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/nearest_movies', methods=['GET'])
def nearest_movies():
    # Get query parameters
    movie_id = request.args.get('movie_id')
    num_neighbors = request.args.get('num_neighbors')

    # Get movie vector
    movie_vector = get_movie_vector(movie_id)

    # Get nearest neighbors
    nearest_neighbors = annoy_index.get_nns_by_vector(movie_vector, num_neighbors)

    # Get movie data for nearest neighbors
    nearest_movies = [get_movie_data(movie_id) for movie_id in nearest_neighbors]

    # Return result as JSON
    return jsonify(nearest_movies)

Получаем вектора из movies.csv
import gensim
import pandas as pd

# Load movie data
movies_df = pd.read_csv('movies.csv')

# Convert movie titles to list of TaggedDocument objects
tagged_movies = [gensim.models.doc2vec.TaggedDocument(title.split(), [i]) for i, title in enumerate(movies_df['title'])]

# Train Doc2Vec model
doc2vec_model = gensim.models.Doc2Vec(vector_size=100, min_count=2, epochs=40)
doc2vec_model.build_vocab(tagged_movies)
doc2vec_model.train(tagged_movies, total_examples=doc2vec_model.corpus_count, epochs=doc2vec_model.epochs)

# Get movie vectors
movie_vectors = [doc2vec_model.infer_vector(title.split()) for title in movies_df['title']]

How to test ? 
use route /search 
For example: 
http://localhost:5000/search?query=Harry%20Potter%20and%20the%20Deathly%20Hallows
