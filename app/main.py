from fastapi import FastAPI, HTTPException
import pandas as pd
import numpy as np

app = FastAPI(title="Worker Recommendation API")

try:
    similarity_matrix = np.load("model/similarity_matrix.npy")
    df_app = pd.read_pickle("model/df_app.pkl")
    df_job = pd.read_pickle("model/df_job.pkl")
except FileNotFoundError:
    raise RuntimeError("Model artifacts not found. Please run 'offline_processing.py' first.")

@app.get("/")
def read_root():
    return {"message": "Welcome to the Worker Recommendation API"}

@app.get("/recommend/{app_id}")
def get_recommendations(app_id: int, top_n: int = 5):
    """
    Get job recommendations for a specific applicant ID.
    """
    if app_id not in df_app.index:
        raise HTTPException(status_code=404, detail=f"Applicant with ID {app_id} not found.")

    similarity_scores = similarity_matrix[app_id]

    top_job_indices = similarity_scores.argsort()[::-1]

    results = df_job.iloc[top_job_indices].copy()
    results['similarity_score'] = similarity_scores[top_job_indices]

    unique_results = results.drop_duplicates(subset=['title']).head(top_n)

    applicant_name = df_app.loc[app_id, 'name']

    return {
        "applicant_id": app_id,
        "applicant_name": applicant_name,
        "recommendations": unique_results[['title', 'location']].to_dict(orient='records')
    }