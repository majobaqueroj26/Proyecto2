import joblib
import tensorflow as tf
from keras.utils.data_utils import pad_sequences
import numpy as np
import json
import pandas as pd

tokenizer_file = '/opt/ml/model/tokenizer.pkl'
model_file = '/opt/ml/model/model-lstm.h5'
tokenizer = joblib.load(tokenizer_file)
model = tf.keras.models.load_model(model_file)
clases = ['Action', 'Adventure', 'Animation', 'Biography', 'Comedy', 'Crime',
       'Documentary', 'Drama', 'Family', 'Fantasy', 'Film-Noir',
       'History', 'Horror', 'Music', 'Musical', 'Mystery', 'News',
       'Romance', 'Sci-Fi', 'Short', 'Sport', 'Thriller', 'War',
       'Western']

def lambda_handler(event, context):
    body = event['body']
    body = json.loads(body)
    txt = str(body)
    seq = tokenizer.texts_to_sequences([txt])
    padded = pad_sequences(seq, maxlen=200)
    prediction = model.predict(padded)
    prediction = list(prediction[0])
    index = [prediction.index(m) for m in prediction if m >= 0.40]
    response = [clases[i] for i in index]
    # df = pd.DataFrame(body)

    #prediction = list(model.predict(df))
    return {
        'statusCode': 200,
        'body': json.dumps({'clases': response}),
   }
