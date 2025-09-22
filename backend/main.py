from fastapi import FastAPI
import pandas as pd
from .pipelines.run_inference import predict
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Study Status Prediction API")

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/predict")
def predict_endpoint(data: dict | list[dict]):
    # Convert single row or multiple rows to DataFrame
    df = pd.DataFrame(data if isinstance(data, list) else [data])
    predictions = predict(df)
    return predictions

if __name__ == "__main__":
    import uvicorn

    # Pass the app as an import string: "module_name:app_variable_name"
    # In this case, "main:app" because your file is main.py and your app object is named 'app'
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)