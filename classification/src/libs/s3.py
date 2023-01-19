import boto3
import pandas as pd
from io import StringIO
import tempfile
import pickle
import json
import os

def get_bucket_env_var():
    # looks like this --> arn:aws:s3:::automatedmodelgenerationstac-modelstorage623766aa-5i7uwrnn9rc9
    bucket_name = os.getenv("MODEL_STORAGE_NAME")
    return bucket_name

def read_csv_from_s3(bucket, filename):
    client = boto3.client('s3')
    csv_obj = client.get_object(Bucket=bucket, Key=filename)
    csv_string = csv_obj['Body'].read().decode('utf-8')
    df = pd.read_csv(StringIO(csv_string))
    return df

