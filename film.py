import random
from annoy import AnnoyIndex
import gensim
import pandas as pd

# Load movie data
movies_df = pd.read_csv('movies.csv')

# Convert movie titles to list of TaggedDocument objects
tagged_movies = [gensim.models.doc2vec.TaggedDocument(title.split(), [i]) for i, title in enumerate(movies_df['title'])]

# Train Doc2Vec model
doc2vec_model = gensim.models.Doc2Vec(vector_size=200, min_count=2, epochs=40)
doc2vec_model.build_vocab(tagged_movies)
doc2vec_model.train(tagged_movies, total_examples=doc2vec_model.corpus_count, epochs=doc2vec_model.epochs)

# Get movie vectors
movie_vectors = [doc2vec_model.infer_vector(title.split()) for title in movies_df['title']]

# Create AnnoyIndex object
annoy_index = AnnoyIndex(200, metric='angular')

# Add vectors to index
for i in range(len(movie_vectors)):
    annoy_index.add_item(i, movie_vectors[i])

# Build the index
annoy_index.build(50)

# Get indices of nearest neighbors for a random movie vector
random_movie_vector = doc2vec_model.infer_vector("The Dark Knight".split())
nearest_neighbors = annoy_index.get_nns_by_vector(random_movie_vector, 10)
print(nearest_neighbors)
doc2vec_model.save('doc2vec.model')