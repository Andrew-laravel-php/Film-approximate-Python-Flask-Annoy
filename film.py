import pandas as pd
import numpy as np
from annoy import AnnoyIndex
from sklearn.feature_extraction.text import TfidfVectorizer

data = pd.read_csv('movies.csv')

vectorizer = TfidfVectorizer(stop_words='english')
vectorized_data = vectorizer.fit_transform(data['Title'])
vector_dimension = vectorized_data.shape[1]
print(vector_dimension)
index = AnnoyIndex(vectorized_data.shape[1], metric='angular')
for i in range(vectorized_data.shape[0]):
    vector = vectorized_data[i].toarray().flatten()
    index.add_item(i, vector)
index.build(n_trees=50)
index.save('films.ann')
