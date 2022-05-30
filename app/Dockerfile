FROM public.ecr.aws/lambda/python:3.9

COPY requirements.txt config_nltk.py ./
COPY model /opt/ml/model

RUN python3.9 -m pip install -r requirements.txt -t .
RUN python3.9 config_nltk.py
COPY app.py ./

CMD ["app.lambda_handler"]
