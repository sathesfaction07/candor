import logging
import time
from pathlib import Path

import boto3

from sme_bridge.learning.classifier import AbstentionClassifier

logger = logging.getLogger(__name__)


class CalibrationJobConfig:
    def __init__(
        self,
        training_data_s3_uri: str,
        output_s3_uri: str,
        role_arn: str,
        job_name_prefix: str = "sme-bridge-calibration",
    ):
        self.training_data_s3_uri = training_data_s3_uri
        self.output_s3_uri = output_s3_uri
        self.role_arn = role_arn
        self.job_name_prefix = job_name_prefix


class CalibrationJobRunner:
    def __init__(self, region: str = "us-east-1"):
        self.sagemaker = boto3.client("sagemaker", region_name=region)
        self.s3 = boto3.client("s3", region_name=region)

    def submit_calibration_job(self, config: CalibrationJobConfig) -> str:
        from datetime import datetime

        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        job_name = f"{config.job_name_prefix}-{timestamp}"

        processing_job_params = {
            "ProcessingJobName": job_name,
            "RoleArn": config.role_arn,
            "AppSpecification": {
                "ImageUri": (
                    "123456789012.dkr.ecr.us-east-1.amazonaws.com/"
                    "sme-bridge-calibration:latest"
                ),
            },
            "ProcessingInputs": [
                {
                    "InputName": "training-data",
                    "S3Input": {
                        "S3Uri": config.training_data_s3_uri,
                        "LocalPath": "/opt/ml/processing/input",
                        "S3DataType": "S3Prefix",
                        "S3InputMode": "File",
                    },
                }
            ],
            "ProcessingOutputConfig": {
                "Outputs": [
                    {
                        "OutputName": "model",
                        "S3Output": {
                            "S3Uri": config.output_s3_uri,
                            "LocalPath": "/opt/ml/processing/output",
                            "S3UploadMode": "EndOfJob",
                        },
                    }
                ]
            },
            "ProcessingResources": {
                "ClusterConfig": {
                    "InstanceCount": 1,
                    "InstanceType": "ml.m5.xlarge",
                    "VolumeSizeInGB": 30,
                }
            },
            "StoppingCondition": {"MaxRuntimeInSeconds": 3600},
        }

        self.sagemaker.create_processing_job(**processing_job_params)
        logger.info(f"Submitted calibration job: {job_name}")
        return job_name

    def poll_job_status(
        self, job_name: str, poll_interval_seconds: float = 10.0, timeout_seconds: float = 1800.0
    ) -> str:
        start_time = time.time()
        while True:
            response = self.sagemaker.describe_processing_job(ProcessingJobName=job_name)
            status = response["ProcessingJobStatus"]

            if status in ["Completed", "Failed", "Stopped"]:
                logger.info(f"Calibration job {job_name} finished with status: {status}")
                return status

            elapsed = time.time() - start_time
            if elapsed > timeout_seconds:
                logger.warning(
            f"Calibration job {job_name} polling timed out after {timeout_seconds}s"
        )
                return "TimedOut"

            time.sleep(poll_interval_seconds)

    def retrieve_trained_classifier(self, s3_uri: str, local_path: str) -> AbstentionClassifier:
        parts = s3_uri.replace("s3://", "").split("/", 1)
        bucket = parts[0]
        key = parts[1] if len(parts) > 1 else "classifier.pkl"

        local_p = Path(local_path)
        local_p.parent.mkdir(parents=True, exist_ok=True)

        self.s3.download_file(bucket, key, str(local_p))
        logger.info(f"Downloaded classifier from {s3_uri} to {local_path}")

        classifier = AbstentionClassifier.load(local_path)
        return classifier
