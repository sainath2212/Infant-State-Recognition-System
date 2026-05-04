from fastapi import FastAPI

app = FastAPI(title="CryNet API")

@app.get("/api/health")
def health_check():
    return {"status": "ok", "message": "CryNet Backend is healthy"}

@app.get("/api/predict")
def predict():
    return {"message": "Prediction endpoint placeholder"}

# In production, uvicorn will run this file on port 5001.
