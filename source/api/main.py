"""
Creator: Ivanovitch Silva / José Marcos Leal B. Filho / Lucas Ismael Campos Medeiros
Date: 22 Julho 2022
Create API
"""
# from typing import Union
import tensorflow as tf
from tensorflow import keras
from pydantic import BaseModel
from fastapi.responses import HTMLResponse
from fastapi import FastAPI
import pandas as pd
import joblib
import os
import wandb
import sys
from source.api.pipeline import FeatureSelector, CategoricalTransformer, NumericalTransformer

# global variables
setattr(sys.modules["__main__"], "FeatureSelector", FeatureSelector)
setattr(sys.modules["__main__"], "CategoricalTransformer", CategoricalTransformer)
setattr(sys.modules["__main__"], "NumericalTransformer", NumericalTransformer)

# name of the model artifact
artifact_model_name = "churn_prediction_project_nn/model_export:latest"

# name of the pipe for data transform
data_transform_name = "churn_prediction_project_nn/data_transform:latest"

# initiate the wandb project
run = wandb.init(project="churn_prediction_project_nn",job_type="api")

# create the api
app = FastAPI()

# declare request example data using pydantic
# a person in our dataset has the following attributes
class Person(BaseModel):
    CreditScore: int
    Geography: str
    Gender: str
    Age: int
    Tenure: int
    Balance: float
    NumOfProducts: int
    HasCrCard: int
    IsActiveMember: int
    EstimatedSalary: float 

    class Config:
        schema_extra = {
            "example": {
                "CreditScore": 850,
                "Geography": 'Spain',
                "Gender": 'Female',
                "Age": 43,
                "Tenure": 2,
                "Balance": 125510.82,
                "NumOfProducts": 1,
                "HasCrCard": 1,
                "IsActiveMember": 1,
                "EstimatedSalary": 79084.1
            }
        }

# give a greeting using GET
@app.get("/", response_class=HTMLResponse)
async def root():
    return """
    <p><span style="font-size:28px"><strong>Customer Churn Prediction</strong></span></p>"""\
    """<p><span style="font-size:20px">It's a Neural Network machine learning project to predict customer churn."""\
        """The dataset contains 10.000 rows, each representing an unique customer with 10 caracteristics. """\
        """(The dataset is avalible """\
        """<a href="https://drive.google.com/file/d/12G9RpQauml0QOUAB3aaPaJVduyEnnMzR/view"> here.)</a>.</span></p>"""

# run the model inference and use a Person data structure via POST to the API.
@app.post("/predict")
async def get_inference(person: Person):
    
    # Download inference artifact
    #model_export_path = run.use_artifact(artifact_model_name).file()
    #model = joblib.load(model_export_path)
    model_export_path = run.use_artifact(artifact_model_name)
    model_dir= model_export_path.download()
    model = keras.models.load_model(model_dir)

    # Download the Pipe for Data Transform
    pipe_data_transform_path = run.use_artifact(data_transform_name).file()
    pipe = joblib.load(pipe_data_transform_path)
    
    # Create a dataframe from the input feature
    # note that we could use pd.DataFrame.from_dict
    # but due be only one instance, it would be necessary to
    # pass the Index.
    df = pd.DataFrame([person.dict()])
    
    # Transforming DataFrame
    df_transformed = pipe.transform(df)
    
    # Predict test data
    predict = model.predict(df_transformed)

    return "Continued" if predict[0] <= 0.5 else "Exited"