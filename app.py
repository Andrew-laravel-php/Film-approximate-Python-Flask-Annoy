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

# Загрузка модели ANNOY из файла films.ann
annoy_index = AnnoyIndex(26632)  # Замените vector_dimension на размерность вашего вектора признаков
annoy_index.load('films.ann')
vectorizer = TfidfVectorizer(stop_words='english')
vectorized_data = vectorizer.fit_transform(data['Title'])

# Создание датафрейма movies_df
movies_df = pd.DataFrame(data, columns=['Title', 'Poster'])

# Пример добавления дополнительных столбцов в датафрейм
movies_df['Genre'] = data['Genre']
movies_df['IMDB Score'] = data['IMDB Score']
@app.route('/search')
def search():
    query = request.args.get('query')
    if not query:
        return jsonify({'error': 'query parameter is missing'})

    movie_vector = vectorizer.transform([query]).toarray().flatten()

    # Найти ближайшие соседи с использованием модели ANNOY
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

if __name__ == '__main__':
    app.run(debug=True)

