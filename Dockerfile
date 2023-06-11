# Використовуємо базовий образ Python
FROM python:3.11.2-slim-buster

# Встановлюємо залежності
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Копіюємо файли додатку
COPY . .

# Встановлюємо змінні середовища
ENV FLASK_APP=app.py

# Запускаємо додаток
CMD ["flask", "run", "--host=0.0.0.0"]
