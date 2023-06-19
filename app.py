from flask import Flask, jsonify, request, render_template, url_for, flash, redirect
from flask_sqlalchemy import SQLAlchemy
from annoy import AnnoyIndex
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import pytest 
import time
from werkzeug.security import generate_password_hash, check_password_hash

# Загрузка данных из файла CSV
data = pd.read_csv('movies.csv')
# Создание и инициализация векторизатора
app = Flask(__name__)
# SQLALCHEMY
app.config['SECRET_KEY'] = 'atoeatoatp1395189kj@'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'  # Используйте вашу базу данных
db = SQLAlchemy(app)

#clases
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(50), unique=True, nullable=False)
    password_hash = db.Column(db.String(128), nullable=False)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

class Movie(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    movie_id = db.Column(db.String(50), nullable=False)
    tfidf_vector = db.Column(db.String(1000), nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)

#register page
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        if not username or not password:
            flash('Пожалуйста, заполните все поля!', 'error')
        else:
            user = User.query.filter_by(username=username).first()
            if user:
                flash('Такой пользователь уже существует!', 'error')
            else:
                new_user = User(username=username, password=password)
                db.session.add(new_user)
                db.session.commit()
                flash('Регистрация прошла успешно!', 'success')
                return redirect('/')
    
    return render_template('register.html')

#dashboard
@app.route('/dashboard')
def dashboard():
    return 'Страница пользователя'

#login page
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        user = User.query.filter_by(username=username).first()
        if user and user.check_password(password):
            # Успешная аутентификация
            return redirect(url_for('dashboard'))
        else:
            # Неправильное имя пользователя или пароль
            return render_template('login.html', error='Неправильное имя пользователя или пароль')

    return render_template('login.html')

#select movie
@app.route('/select_movie/<username>/<movie_id>/<tfidf_vector>', methods=['GET'])
def select_movie(username, movie_id, tfidf_vector):
    user = User.query.filter_by(username=username).first()
    if user:
        # Пример выбора фильма для пользователя
        movie = Movie(movie_id=movie_id, tfidf_vector=tfidf_vector, user=user)
        db.session.add(movie)
        db.session.commit()
        return 'Фильм выбран для пользователя ' + username
    else:
        return 'Пользователь не найден'

#indexpage
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

#searchrequest
@app.route('/search')
def search():
    start_time = time.time()

    query = request.args.get('query')
    if not query:
        return jsonify({'error': 'query parameter is missing'})

    movie_vector = vectorizer.transform([query]).toarray().flatten()
    end_time = time.time()
    print(f"MOVIE VECTOR 39: {end_time - start_time}")
    nearest_neighbors, distances = annoy_index.get_nns_by_vector(movie_vector, 5, include_distances=True)
    end_time = time.time()
    print(f"Nearest neighboor 40: {end_time - start_time}")
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

    user_preferences = request.args.getlist('preferences') 

    if user_preferences:
        recommended_movies = filter_movies_based_on_preferences(similar_movies, user_preferences)
        if recommended_movies.empty:
            return jsonify({'message': 'No movies found based on your preferences.'})
        else:
            end_time = time.time()
            print(f"Time to run search: {end_time - start_time}")
            return jsonify(recommended_movies.to_dict(orient='records'))
           
    else:
        end_time = time.time()
        print(f"Time to run search: {end_time - start_time}")
        return jsonify(similar_movies.to_dict(orient='records'))
    


def filter_movies_based_on_preferences(movies_df, user_preferences):
    filtered_movies = movies_df[movies_df['Genre'].isin(user_preferences)]
    return filtered_movies

#searchbycategory
@app.route('/searchbycategory')
def search_by_category():
    genre = request.args.get('genre')
    year = request.args.get('year')
    rating = request.args.get('rating')
    filtered_movies = pd.DataFrame()
    titles = pd.DataFrame()
    if genre:
        movies_df['Genre'] = movies_df['Genre'].fillna('')
        filtered_movies = movies_df[movies_df['Genre'].str.contains(str(genre))]
    if rating:
        filtered_movies = filtered_movies[filtered_movies['IMDB Score'] >= float(rating)*2-1]

    if not filtered_movies.empty:
        # num_rows = len(filtered_movies)
        filtered_movies = filtered_movies.sample(n=5)
        titles = filtered_movies['Title']
        posters = filtered_movies['Poster']
        

    if not titles.empty:
        titles = titles.head(10).tolist()
        posters = posters.head(10).tolist()
        return render_template('search_by_category.html', similar_movies=titles,posters=posters)
    else:
        return render_template('search_by_category.html')

#pytests
@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_home_page():
    response = app.test_client().get('/')
    assert response.status_code == 200
    assert b"Welcome to my Flask app" in response.data


if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)

