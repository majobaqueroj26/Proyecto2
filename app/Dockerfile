FROM public.ecr.aws/lambda/python:3.9

COPY requirements.txt ./
COPY model /opt/ml/model

RUN python3.9 -m pip install -r requirements.txt -t .
COPY app.py ./

CMD ["app.lambda_handler"]
