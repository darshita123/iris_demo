# app.py
from fastapi import FastAPI
import pickle

app = FastAPI()

# Load model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

@app.get("/")
def read_root():
    return {"message": "Welcome to FastAPI with a trained model!"}

@app.get("/predict")
def predict(sepal_length: float, sepal_width: float, petal_length: float, petal_width: float):
    prediction = model.predict([[sepal_length, sepal_width, petal_length, petal_width]])
    return {"prediction": int(prediction[0])}
