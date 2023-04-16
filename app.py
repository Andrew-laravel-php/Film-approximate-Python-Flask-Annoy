from flask import Flask, jsonify, request, render_template
import pandas as pd
import gensim
from annoy import AnnoyIndex

app = Flask(__name__)

# Load movie data and create Annoy index
movies_df = pd.read_csv('movies.csv')
tagged_movies = [gensim.models.doc2vec.TaggedDocument(title.split(), [i]) for i, title in enumerate(movies_df['title'])]

# Load Doc2Vec model
doc2vec_model = gensim.models.Doc2Vec.load('doc2vec.model')

# Create AnnoyIndex object
movie_vectors = [doc2vec_model.infer_vector(title.split()) for title in movies_df['title']]
annoy_index = AnnoyIndex(doc2vec_model.vector_size, metric='angular')
for i in range(len(movie_vectors)):
    annoy_index.add_item(i, movie_vectors[i])
annoy_index.build(50)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/suggest')
def suggest():
  query = request.args.get('q')
  suggestions = movies_df[movies_df['title'].str.contains(query, case=False)]['title'].tolist()
  return jsonify(suggestions)

@app.route('/search')
def search():
    query = request.args.get('query')
    if not query:
        return jsonify({'error': 'query parameter is missing'})
    movie_vector = doc2vec_model.infer_vector(query.split())
    nearest_neighbors = annoy_index.get_nns_by_vector(movie_vector, 10)
    similar_movies = movies_df.iloc[nearest_neighbors]['title']
    return jsonify(similar_movies.tolist())

if __name__ == '__main__':
    app.run()
