from fastapi import FastAPI
from pydantic import BaseModel
from model import load_and_predict

app = FastAPI(title="Medical Insurance Cost Predictor")


class Person(BaseModel):
    age: int
    sex: str       
    bmi: float
    children: int
    smoker: str   
    region: str    


@app.get("/")
def root():
    return {"message": "Insurance predictor is running. POST to /predict"}


@app.post("/predict")
def predict(person: Person):
    charge = load_and_predict(person.model_dump(), path="model.pkl")
    return {"predicted_charge": round(charge, 2)}