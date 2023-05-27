import pandas as pd
import matplotlib.pyplot as plt

# Загрузка данных из CSV-файла
data = pd.read_csv('movies.csv')

# Создание графика
plt.figure(figsize=(10, 6))
plt.bar(data['Title'], data['IMDB Score'])
plt.xticks(rotation='vertical')
plt.xlabel('Название фильма')
plt.ylabel('Рейтинг IMDB')
plt.title('Рекомендации фильмов')
plt.subplots_adjust(bottom=0.15, top=0.9)  # Изменение параметров границ фигуры

# Отображение графика
plt.show()
