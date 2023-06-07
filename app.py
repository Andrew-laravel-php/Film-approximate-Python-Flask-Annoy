from flask import Flask, jsonify, request, render_template, url_for, flash, redirect
import pandas as pd
import gensim
from gensim.models import Doc2Vec
from annoy import AnnoyIndex
from sklearn.feature_extraction.text import TfidfVectorizer
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SubmitField
from wtforms.validators import DataRequired, Email, EqualTo

# Загрузка данных из файла CSV
data = pd.read_csv('movies.csv')
# Создание и инициализация векторизатора
app = Flask(__name__)
app.config['SECRET_KEY'] = 'apetiajiet1u351358paiX14u1390@1!'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///filmdb.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Class for form registration
class RegistrationForm(FlaskForm):
    username = StringField('Username', validators=[DataRequired()])
    email = StringField('Email', validators=[DataRequired(), Email()])
    password = PasswordField('Password', validators=[DataRequired()])
    confirm_password = PasswordField('Confirm Password', validators=[DataRequired(), EqualTo('password')])
    submit = SubmitField('Sign Up')

# # Model for user creation
# class User(db.Model):
#     id = db.Column(db.Integer, primary_key=True)
#     username = db.Column(db.String(80), unique=True, nullable=False)
#     email = db.Column(db.String(120), unique=True, nullable=False)
#     password = db.Column(db.String(80), nullable=False)
#     pass_hash = db.Column(db.String(80), nullable=False)

# db.create_all()


@app.route('/register', methods=['GET', 'POST'])
def register():
    form = RegistrationForm()
    if form.validate_on_submit():
        # Добавьте здесь код для обработки данных формы регистрации
        flash('Registration successful!', 'success')
        return redirect(url_for('login'))
    return render_template('register.html', form=form)

@app.route('/login')
def login():
    return render_template('index.html')


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

    user_preferences = request.args.getlist('preferences') 

    if user_preferences:
        recommended_movies = filter_movies_based_on_preferences(similar_movies, user_preferences)
        if recommended_movies.empty:
            return jsonify({'message': 'No movies found based on your preferences.'})
        else:
            return jsonify(recommended_movies.to_dict(orient='records'))
    else:
        return jsonify(similar_movies.to_dict(orient='records'))


def filter_movies_based_on_preferences(movies_df, user_preferences):
    filtered_movies = movies_df[movies_df['Genre'].isin(user_preferences)]
    return filtered_movies

@app.route('/searchbycategory')
def search_by_category():
    genre = request.args.get('genre')
    year = request.args.get('year')
    rating = request.args.get('rating')

    #print(f'{genre}, {year}, {rating}')

    # print(movies_df.columns)

    # filtered_movies = movies_df[['Title', 'Genre']]

    # filtered_movies = filtered_movies.dropna(subset=['Genre'])
    # filtered_movies = filtered_movies[filtered_movies['Genre'].str.contains(genre, case=False)]

    filtered_movies = pd.DataFrame()
    titles = pd.DataFrame()

    if genre:
        movies_df['Genre'] = movies_df['Genre'].fillna('')
        filtered_movies = movies_df[movies_df['Genre'].str.contains(str(genre))]
    
    # if year:
    #     filtered_movies = filtered_movies[filtered_movies['Year'] == year]

    if rating:
        filtered_movies = filtered_movies[filtered_movies['IMDB Score'] >= float(rating)*2-1]
        print(filtered_movies)

    if not filtered_movies.empty:
        titles = filtered_movies['Title']

    if not titles.empty:
        titles = titles.head(10)
        return render_template('search_by_category.html', similar_movies=titles)
    else:
        return render_template('search_by_category.html')

    # if genre:
    #     filtered_movies = filtered_movies[filtered_movies['Genre'].str.contains(genre, case=False)]

    # if year:
    #     filtered_movies = filtered_movies[filtered_movies['Year'] == int(year)]

    # if rating:
    #     filtered_movies = filtered_movies[filtered_movies['IMDB Score'] >= float(rating)]

    # if filtered_movies.empty:
    #     return jsonify({'message': 'No movies found based on the specified category.'})
    # else:
    #     similar_movies = filtered_movies['Title'].tolist()
    #     return render_template('search_by_category.html', similar_movies=similar_movies)



if __name__ == '__main__':
    app.run(debug=True)

