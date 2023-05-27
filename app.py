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

    # Search for the nearest neighbors using the ANNOY model
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

    # User preferences
    user_preferences = request.args.getlist('preferences')  # Assuming preferences are passed as a list in the request

    if user_preferences:
        # Filter recommendations based on user preferences
        recommended_movies = filter_movies_based_on_preferences(similar_movies, user_preferences)
        return jsonify(recommended_movies.to_dict(orient='records'))
    else:
        return jsonify(similar_movies.to_dict(orient='records'))


def filter_movies_based_on_preferences(movies_df, user_preferences):
    filtered_movies = movies_df[movies_df['Genre'].isin(user_preferences)]
    return filtered_movies


@app.route('/searchbycategory')
def search_by_category():
    return render_template('search_by_category.html')

if __name__ == '__main__':
    app.run(debug=True)

