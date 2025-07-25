import pandas as pd
import numpy as np
import re
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import nltk
from nltk.corpus import stopwords
import pickle

try:
    stopwords.words('indonesian')
except (LookupError, OSError):
    nltk.download('stopwords')
    nltk.download('punkt') 

def process_text_series(text_series):
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()
    stopwords_id = set(stopwords.words('indonesian'))

    def clean_noise(text):
        text = re.sub(r'<.*?>', ' ', text)
        text = re.sub(r'[^\\w\\s]', ' ', text)
        text = re.sub(r'\\d+', ' ', text)
        text = re.sub(r'\\s+', ' ', text).strip()
        return text

    def remove_stopwords(text):
        words = text.split()
        return ' '.join([word for word in words if word not in stopwords_id])

    def stem_text(text):
        return stemmer.stem(text)

    processed_texts = []
    for text in text_series:
        text = text.lower()
        text = clean_noise(text)
        text = remove_stopwords(text)
        text = stem_text(text)
        processed_texts.append(text)
    return processed_texts

def run_processing():
    print("Starting offline processing...")

    df_app = pd.read_csv('data/lamaran_pekerjaan_informal.csv')
    df_job = pd.read_csv('data/pekerjaan_informal_indonesia.csv')

    df_app['text'] = df_app['name'] + ' ' + df_app['description'] + ' ' + df_app['tags']
    df_job['text'] = df_job['title'] + ' ' + df_job['description'] + ' ' + df_job['categories']

    df_app['processed_text'] = process_text_series(df_app['text'])
    df_job['processed_text'] = process_text_series(df_job['text'])

    print("Text processing complete.")

    model_bert = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    print("Encoding applicant vectors...")
    app_vectors = model_bert.encode(df_app['processed_text'])
    print("Encoding job vectors...")
    job_vectors = model_bert.encode(df_job['processed_text'])

    print("Calculating cosine similarity...")
    similarity_matrix = cosine_similarity(app_vectors, job_vectors)

    print("Saving artifacts to model/ directory...")
    np.save('model/similarity_matrix.npy', similarity_matrix)

    df_app[['seekerEmail', 'name']].to_pickle('model/df_app.pkl')
    df_job[['title', 'location', 'providerEmail']].to_pickle('model/df_job.pkl')

    print("Offline processing finished successfully!")

if __name__ == '__main__':
    run_processing()
