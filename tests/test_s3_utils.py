import os
import boto3
import pytest
from moto import mock_s3

from utils.s3_utils import download_model_from_s3, upload_model_to_s3

BUCKET_NAME = "test-bucket"
S3_KEY = "model/latest_model.json"
TMP_DIR = "tmp_test_model"
LOCAL_FILE = os.path.join(TMP_DIR, "latest_model.json")

@pytest.fixture
def setup_s3_bucket():
    with mock_s3():
        s3 = boto3.client("s3", region_name="us-east-1")
        s3.create_bucket(Bucket=BUCKET_NAME)
        yield s3

def test_upload_and_download_model(setup_s3_bucket):
    # 1. Create a local dummy model file
    os.makedirs(TMP_DIR, exist_ok=True)
    with open(LOCAL_FILE, "w") as f:
        f.write("hello model")

    # 2. Upload the model (to the in-memory mock S3 provided by moto)
    upload_model_to_s3(LOCAL_FILE, BUCKET_NAME, S3_KEY, with_history=False)

    # 3. Remove the local file to prepare for the download test
    os.remove(LOCAL_FILE)

    # 4. Download the model
    download_model_from_s3(BUCKET_NAME, S3_KEY, TMP_DIR)

    # 5. Verify the downloaded file's content
    assert os.path.exists(LOCAL_FILE)
    with open(LOCAL_FILE) as f:
        assert f.read() == "hello model"

    # 6. Cleanup
    os.remove(LOCAL_FILE)
    os.rmdir(TMP_DIR)
