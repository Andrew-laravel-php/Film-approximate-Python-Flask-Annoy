from flask import Flask, jsonify, request, render_template
import pandas as pd
import gensim
from gensim.models import Doc2Vec
from annoy import AnnoyIndex
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

# Загрузка данных из файла CSV
data = pd.read_csv('movies.csv')
# Создание и инициализация векторизатора
app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')

annoy_index = AnnoyIndex(26632)
annoy_index.load('films.ann')

vectorizer = TfidfVectorizer(stop_words='english')
vectorized_data = vectorizer.fit_transform(data['Title'])

movies_df = pd.DataFrame(data, columns=['Title', 'Poster'])
movies_df['Genre'] = data['Genre']
movies_df['IMDB Score'] = data['IMDB Score']
@app.route('/search')
def search():
    query = request.args.get('query')
    if not query:
        return jsonify({'error': 'query parameter is missing'})

    movie_vector = vectorizer.transform([query]).toarray().flatten()

    # Пошук найближчого сусіда з використанням моделі ANNOY
    nearest_neighbors, distances = annoy_index.get_nns_by_vector(movie_vector, 5, include_distances=True)

    similar_movies = movies_df.iloc[nearest_neighbors][['Title', 'Poster']]
    similar_movies['Distance'] = distances

    colors = []
    for distance in distances:
        if distance < 0.2:
            colors.append('green')
        elif distance < 0.4:
            colors.append('yellow')
        elif distance < 0.6:
            colors.append('orange')
        else:
            colors.append('red')
    similar_movies['Color'] = colors

    return jsonify(similar_movies.to_dict(orient='records'))

@app.route('/searchbycategory')
def search_by_category():
    category = request.args.get('category')
    if not category:
        return jsonify({'error': 'category parameter is missing'})

    # Получение всех фильмов из указанной категории
    category_movies = movies_df[movies_df['Category'] == category]

    if category_movies.empty:
        return jsonify({'error': 'No movies found in the specified category'})

    # Получение векторов фильмов в указанной категории
    category_vectors = [vectorizer.transform([title]).toarray().flatten() for title in category_movies['Title']]

    # Поиск ближайших соседей для каждого фильма в указанной категории
    similar_movies = []
    for vector in category_vectors:
        nearest_neighbors, distances = annoy_index.get_nns_by_vector(vector, 5, include_distances=True)
        similar_movies.extend(movies_df.iloc[nearest_neighbors][['Title', 'Poster']].to_dict(orient='records'))

    # Добавление информации о расстоянии и цветах
    for movie in similar_movies:
        distance = movie['Distance']
        if distance < 0.2:
            movie['Color'] = 'green'
        elif distance < 0.4:
            movie['Color'] = 'yellow'
        elif distance < 0.6:
            movie['Color'] = 'orange'
        else:
            movie['Color'] = 'red'

    return jsonify(similar_movies)

if __name__ == '__main__':
    app.run(debug=True)

