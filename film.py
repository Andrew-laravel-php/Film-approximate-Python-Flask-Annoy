import pandas as pd
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize

# Загрузка данных из CSV-файла
df = pd.read_csv('movies.csv')

# Создание документов для обучения модели
documents = [TaggedDocument(words=word_tokenize(title.lower()), tags=[str(index)]) for index, title in df['Title'].items()]

# Обучение модели Doc2Vec
model = Doc2Vec(documents, vector_size=300, window=5, min_count=1, workers=4, epochs=20)

# Сохранение модели в файл
model.save('doc2vec.model')
