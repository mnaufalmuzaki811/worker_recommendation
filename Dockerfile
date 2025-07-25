FROM python:3.11-slim

WORKDIR /code

RUN pip install nltk && python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt')"

COPY ./requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

COPY ./app /code/app
COPY ./model /code/model

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]