#!/usr/bin/python
from flask import Flask, jsonify
from sklearn.externals import joblib 
import pandas as pd
import os

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
     json_ = request.json

     query_df = pd.DataFrame(json_)
     query = pd.get_dummies(query_df)
     prediction = clf.predict(query)
     return jsonify({'prediction': list(prediction)})
     
if __name__ == '__main__':
     clf = joblib.load(os.path.dirname(__file__) + '/NN_model.pkl')
     app.run(debug=True, use_reloader=False, host='0.0.0.0', port=5000)