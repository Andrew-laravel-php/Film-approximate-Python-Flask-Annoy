from flask import Flask, jsonify, request, render_template
import pandas as pd
import gensim
from gensim.models import Doc2Vec
from annoy import AnnoyIndex

import spacy

nlp = spacy.load("en_core_web_sm")
# from asgiref.wsgi import WsgiToAsgi

app = Flask(__name__)

# app = WsgiToAsgi(app)
# Load movie data and create Annoy index
movies_df = pd.read_csv('movies.csv')
tagged_movies = [gensim.models.doc2vec.TaggedDocument(title.split(), [i]) for i, title in enumerate(movies_df['Title'])]

model = Doc2Vec.load('doc2vec.model')

# Load Doc2Vec model
doc2vec_model = gensim.models.Doc2Vec.load('doc2vec.model')

# Create AnnoyIndex object
movie_vectors = [doc2vec_model.infer_vector(title.split()) for title in movies_df['Title']]
annoy_index = AnnoyIndex(doc2vec_model.vector_size, metric='angular')
for i in range(len(movie_vectors)):
    annoy_index.add_item(i, movie_vectors[i])
annoy_index.build(50)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/searchbycategory')
def sbc():
    return render_template('search_by_category.html')

@app.route('/suggest')
def suggest():
  query = request.args.get('q')
  suggestions = movies_df[movies_df['Title'].str.contains(query, case=False)]['Title'].tolist()
  return jsonify(suggestions)

@app.route('/search')
def search():
    query = request.args.get('query')
    if not query:
        return jsonify({'error': 'query parameter is missing'})
    movie_vector = doc2vec_model.infer_vector(query.split())
    nearest_neighbors, distances = annoy_index.get_nns_by_vector(movie_vector, 10, include_distances=True)
    similar_movies = movies_df.iloc[nearest_neighbors][['Title', 'Poster']]
    similar_movies['Distance'] = distances
    return jsonify(similar_movies.to_dict(orient='records'))



@app.route('/search_by_category')
def search_by_category():
    genre = request.args.get('genre')
    year = request.args.get('year')
    rating = request.args.get('rating')
    
    query = f"{genre} {year} {rating}"
    
    movie_vector = doc2vec_model.infer_vector(query.split())
    nearest_neighbors = annoy_index.get_nns_by_vector(movie_vector, 10)
    similar_movies = movies_df.iloc[nearest_neighbors]['Title']
    
    return render_template('search_by_category.html', similar_movies=similar_movies.tolist())




if __name__ == '__main__':
    app.run(debug=True)

