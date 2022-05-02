import joblib
import numpy as np
import json
import pandas as pd

model_file = '/opt/ml/model/NN_model.pkl'
model = joblib.load(model_file)


def lambda_handler(event, context):
    body = event['body']
    body = json.loads(body)
    df = pd.DataFrame(body)
    prediction = list(model.predict(df))
    return {
        'statusCode': 200,
        'body': json.dumps(
            {
                "prediction": prediction,
            }
        )
   }
