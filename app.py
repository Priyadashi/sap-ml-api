{\rtf1\ansi\ansicpg1252\cocoartf2822
\cocoatextscaling0\cocoaplatform0{\fonttbl\f0\fswiss\fcharset0 Helvetica;}
{\colortbl;\red255\green255\blue255;}
{\*\expandedcolortbl;;}
\paperw11900\paperh16840\margl1440\margr1440\vieww11520\viewh8400\viewkind0
\pard\tx720\tx1440\tx2160\tx2880\tx3600\tx4320\tx5040\tx5760\tx6480\tx7200\tx7920\tx8640\pardirnatural\partightenfactor0

\f0\fs24 \cf0 from fastapi import FastAPI\
import joblib\
import pandas as pd\
\
app = FastAPI()\
\
# Load your trained model\
model = joblib.load("optimized_sap_ar_model.pkl")\
\
@app.get("/")\
async def home():\
    return \{"status": "ML API is running"\}\
\
@app.post("/predict")\
async def predict(input_data: dict):\
    try:\
        input_df = pd.DataFrame([input_data])\
        input_encoded = pd.get_dummies(input_df).reindex(columns=model.get_booster().feature_names, fill_value=0)\
        prediction = model.predict(input_encoded)[0]\
        prediction_proba = model.predict_proba(input_encoded)[0, 1]\
\
        return \{\
            "prediction": int(prediction),\
            "probability": float(prediction_proba)\
        \}\
\
    except Exception as e:\
        return \{"error": str(e)\}\
}