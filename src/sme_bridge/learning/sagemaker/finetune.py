import json
import logging
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import boto3

logger = logging.getLogger(__name__)


@dataclass
class FineTuneConfig:
    model_id: str
    training_data_s3_uri: str
    output_s3_uri: str
    role_arn: str
    hyperparameters: dict | None = None
    job_name_prefix: str = "candor-finetune"


@dataclass
class ModelVersion:
    version_id: str
    model_arn: str
    created_at: datetime
    status: str
    is_active: bool


class SageMakerClient:
    def __init__(self, region: str = "us-east-1"):
        self.sagemaker = boto3.client("sagemaker", region_name=region)
        self.s3 = boto3.client("s3", region_name=region)

    def upload_training_data(self, data: list[dict], s3_uri: str) -> None:
        parts = s3_uri.replace("s3://", "").split("/", 1)
        bucket = parts[0]
        key = parts[1] if len(parts) > 1 else "data.jsonl"

        jsonl_content = "\n".join(json.dumps(record) for record in data)

        self.s3.put_object(Bucket=bucket, Key=key, Body=jsonl_content.encode("utf-8"))
        logger.info(f"Uploaded {len(data)} training records to {s3_uri}")

    def submit_finetune_job(self, config: FineTuneConfig) -> str:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        job_name = f"{config.job_name_prefix}-{timestamp}"

        hyperparameters = config.hyperparameters or {
            "epochs": "3",
            "learning_rate": "2e-5",
            "batch_size": "8",
        }

        training_job_params = {
            "TrainingJobName": job_name,
            "RoleArn": config.role_arn,
            "AlgorithmSpecification": {
                "TrainingImage": config.model_id,
                "TrainingInputMode": "File",
            },
            "InputDataConfig": [
                {
                    "ChannelName": "training",
                    "DataSource": {
                        "S3DataSource": {
                            "S3DataType": "S3Prefix",
                            "S3Uri": config.training_data_s3_uri,
                        }
                    },
                }
            ],
            "OutputDataConfig": {"S3OutputPath": config.output_s3_uri},
            "ResourceConfig": {
                "InstanceType": "ml.g5.2xlarge",
                "InstanceCount": 1,
                "VolumeSizeInGB": 100,
            },
            "StoppingCondition": {"MaxRuntimeInSeconds": 86400},
            "HyperParameters": hyperparameters,
        }

        self.sagemaker.create_training_job(**training_job_params)
        logger.info(f"Submitted fine-tuning job: {job_name}")
        return job_name

    def poll_job_status(
        self, job_name: str, poll_interval_seconds: float = 30.0, timeout_seconds: float = 3600.0
    ) -> str:
        start_time = time.time()
        while True:
            response = self.sagemaker.describe_training_job(TrainingJobName=job_name)
            status = response["TrainingJobStatus"]

            if status in ["Completed", "Failed", "Stopped"]:
                logger.info(f"Job {job_name} finished with status: {status}")
                return status

            elapsed = time.time() - start_time
            if elapsed > timeout_seconds:
                logger.warning(f"Job {job_name} polling timed out after {timeout_seconds}s")
                return "TimedOut"

            time.sleep(poll_interval_seconds)


class ModelRegistry:
    def __init__(self, model_package_group: str, region: str = "us-east-1"):
        self.model_package_group = model_package_group
        self.sagemaker = boto3.client("sagemaker", region_name=region)

    def register_model(self, model_s3_uri: str, version_description: str) -> ModelVersion:
        timestamp = datetime.now()
        response = self.sagemaker.create_model_package(
            ModelPackageGroupName=self.model_package_group,
            ModelPackageDescription=version_description,
            InferenceSpecification={
                "Containers": [
                    {
                        "Image": "123456789012.dkr.ecr.us-east-1.amazonaws.com/candor:latest",
                        "ModelDataUrl": model_s3_uri,
                    }
                ],
                "SupportedContentTypes": ["application/json"],
                "SupportedResponseMIMETypes": ["application/json"],
            },
        )

        model_arn = response["ModelPackageArn"]
        version_id = model_arn.split("/")[-1]

        logger.info(f"Registered model version {version_id}")

        return ModelVersion(
            version_id=version_id,
            model_arn=model_arn,
            created_at=timestamp,
            status="Pending",
            is_active=False,
        )

    def list_versions(self) -> list[ModelVersion]:
        response = self.sagemaker.list_model_packages(
            ModelPackageGroupName=self.model_package_group,
            SortBy="CreationTime",
            SortOrder="Descending",
        )

        versions = []
        for pkg in response.get("ModelPackageSummaryList", []):
            versions.append(
                ModelVersion(
                    version_id=pkg["ModelPackageArn"].split("/")[-1],
                    model_arn=pkg["ModelPackageArn"],
                    created_at=pkg["CreationTime"],
                    status=pkg["ModelPackageStatus"],
                    is_active=pkg.get("ModelApprovalStatus") == "Approved",
                )
            )

        return versions

    def promote_version(self, version_id: str) -> None:
        versions = self.list_versions()
        target_version = next((v for v in versions if v.version_id == version_id), None)

        if not target_version:
            raise ValueError(f"Version {version_id} not found")

        self.sagemaker.update_model_package(
            ModelPackageArn=target_version.model_arn,
            ModelApprovalStatus="Approved",
        )
        logger.info(f"Promoted version {version_id} to active")

    def rollback_to_version(self, version_id: str) -> None:
        versions = self.list_versions()

        for v in versions:
            if v.is_active:
                self.sagemaker.update_model_package(
                    ModelPackageArn=v.model_arn,
                    ModelApprovalStatus="PendingManualApproval",
                )

        self.promote_version(version_id)
        logger.info(f"Rolled back to version {version_id}")


def export_qa_pairs_for_finetuning(qa_pairs: list[dict], output_path: str) -> None:
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)

    with p.open("w") as f:
        for qa in qa_pairs:
            f.write(json.dumps(qa) + "\n")

    logger.info(f"Exported {len(qa_pairs)} QA pairs to {output_path}")
