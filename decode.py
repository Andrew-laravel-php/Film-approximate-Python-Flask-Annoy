import pandas as pd

df = pd.read_csv('movies.csv', encoding='ISO-8859-1')

with open('movies.csv', 'rb') as f:
    content = f.read()
content = content.replace(b'\xa9', b'')
with open('movies.csv', 'wb') as f:
    f.write(content)

df = pd.read_csv('movies.csv', encoding='ISO-8859-1')
