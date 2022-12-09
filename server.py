from fastapi import FastAPI,Request
import uvicorn
import numpy as np
import pandas as pd
import pickle
from schema import *

app = FastAPI()

with open('rf.pkl','rb') as f:
    model = pickle.load(f)
 
@app.get('/')
def home():
    return {"status":"server is running"}

@app.post('/predict')
async def predict_data(x:ModelPred,request: Request):
    result = await request.json()  
    print(result)
    x = [v for _,v in result.items()]
    x = np.array(x).reshape(1,-1)
    pred = model.predict(x)
    return {"prediction":pred[0]}


if __name__=="__main__":
    uvicorn.run('server:app',port=1234,reload=True)