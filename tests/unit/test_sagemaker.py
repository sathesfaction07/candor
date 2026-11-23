import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

from sme_bridge.learning.sagemaker.calibration_job import (
    CalibrationJobConfig,
    CalibrationJobRunner,
)
from sme_bridge.learning.sagemaker.finetune import (
    FineTuneConfig,
    ModelRegistry,
    SageMakerClient,
    export_qa_pairs_for_finetuning,
)


@patch("boto3.client")
def test_sagemaker_client_upload_training_data(mock_boto):
    mock_s3 = MagicMock()
    mock_boto.return_value = mock_s3

    client = SageMakerClient()
    client.s3 = mock_s3

    data = [{"query": "test", "answer": "response"}]
    client.upload_training_data(data, "s3://bucket/path/data.jsonl")

    mock_s3.put_object.assert_called_once()
    call_args = mock_s3.put_object.call_args
    assert call_args[1]["Bucket"] == "bucket"
    assert call_args[1]["Key"] == "path/data.jsonl"


@patch("boto3.client")
def test_sagemaker_client_submit_finetune_job(mock_boto):
    mock_sagemaker = MagicMock()
    mock_sagemaker.create_training_job.return_value = {"TrainingJobArn": "arn:aws:sagemaker:job"}
    mock_boto.return_value = mock_sagemaker

    client = SageMakerClient()
    client.sagemaker = mock_sagemaker

    config = FineTuneConfig(
        model_id="model123",
        training_data_s3_uri="s3://bucket/train.jsonl",
        output_s3_uri="s3://bucket/output",
        role_arn="arn:aws:iam::123456789012:role/SageMakerRole",
    )

    job_name = client.submit_finetune_job(config)

    assert "sme-bridge-finetune" in job_name
    mock_sagemaker.create_training_job.assert_called_once()
    call_args = mock_sagemaker.create_training_job.call_args[1]
    assert call_args["RoleArn"] == config.role_arn
    assert call_args["AlgorithmSpecification"]["TrainingImage"] == config.model_id


@patch("boto3.client")
def test_sagemaker_client_poll_job_status_completed(mock_boto):
    mock_sagemaker = MagicMock()
    mock_sagemaker.describe_training_job.return_value = {"TrainingJobStatus": "Completed"}
    mock_boto.return_value = mock_sagemaker

    client = SageMakerClient()
    client.sagemaker = mock_sagemaker

    status = client.poll_job_status("test-job", poll_interval_seconds=0.001)

    assert status == "Completed"


@patch("boto3.client")
def test_sagemaker_client_poll_job_status_transitions(mock_boto):
    mock_sagemaker = MagicMock()
    statuses = ["InProgress", "InProgress", "Completed"]
    call_count = 0

    def side_effect(*args, **kwargs):
        nonlocal call_count
        result = {"TrainingJobStatus": statuses[min(call_count, len(statuses) - 1)]}
        call_count += 1
        return result

    mock_sagemaker.describe_training_job.side_effect = side_effect
    mock_boto.return_value = mock_sagemaker

    client = SageMakerClient()
    client.sagemaker = mock_sagemaker

    status = client.poll_job_status("test-job", poll_interval_seconds=0.001)

    assert status == "Completed"
    assert call_count == 3


@patch("boto3.client")
def test_sagemaker_client_poll_job_status_failed(mock_boto):
    mock_sagemaker = MagicMock()
    mock_sagemaker.describe_training_job.return_value = {"TrainingJobStatus": "Failed"}
    mock_boto.return_value = mock_sagemaker

    client = SageMakerClient()
    client.sagemaker = mock_sagemaker

    status = client.poll_job_status("test-job", poll_interval_seconds=0.001)

    assert status == "Failed"


@patch("boto3.client")
def test_model_registry_register_model(mock_boto):
    mock_sagemaker = MagicMock()
    mock_sagemaker.create_model_package.return_value = {
        "ModelPackageArn": "arn:aws:sagemaker:::model-package/group/v1"
    }
    mock_boto.return_value = mock_sagemaker

    registry = ModelRegistry("test-group")
    registry.sagemaker = mock_sagemaker

    version = registry.register_model("s3://bucket/model.tar.gz", "Test model")

    assert version.version_id == "v1"
    assert version.model_arn == "arn:aws:sagemaker:::model-package/group/v1"
    mock_sagemaker.create_model_package.assert_called_once()


@patch("boto3.client")
def test_model_registry_list_versions(mock_boto):
    mock_sagemaker = MagicMock()
    mock_sagemaker.list_model_packages.return_value = {
        "ModelPackageSummaryList": [
            {
                "ModelPackageArn": "arn:aws:sagemaker:::model-package/group/v2",
                "CreationTime": datetime.now(),
                "ModelPackageStatus": "Completed",
                "ModelApprovalStatus": "Approved",
            },
            {
                "ModelPackageArn": "arn:aws:sagemaker:::model-package/group/v1",
                "CreationTime": datetime.now(),
                "ModelPackageStatus": "Completed",
                "ModelApprovalStatus": "PendingManualApproval",
            },
        ]
    }
    mock_boto.return_value = mock_sagemaker

    registry = ModelRegistry("test-group")
    registry.sagemaker = mock_sagemaker

    versions = registry.list_versions()

    assert len(versions) == 2
    assert versions[0].version_id == "v2"
    assert versions[0].is_active is True
    assert versions[1].version_id == "v1"
    assert versions[1].is_active is False


@patch("boto3.client")
def test_model_registry_promote_version(mock_boto):
    mock_sagemaker = MagicMock()
    mock_sagemaker.list_model_packages.return_value = {
        "ModelPackageSummaryList": [
            {
                "ModelPackageArn": "arn:aws:sagemaker:::model-package/group/v1",
                "CreationTime": datetime.now(),
                "ModelPackageStatus": "Completed",
                "ModelApprovalStatus": "PendingManualApproval",
            }
        ]
    }
    mock_boto.return_value = mock_sagemaker

    registry = ModelRegistry("test-group")
    registry.sagemaker = mock_sagemaker

    registry.promote_version("v1")

    mock_sagemaker.update_model_package.assert_called_once()
    call_args = mock_sagemaker.update_model_package.call_args[1]
    assert call_args["ModelApprovalStatus"] == "Approved"


@patch("boto3.client")
def test_model_registry_rollback_to_version(mock_boto):
    mock_sagemaker = MagicMock()
    mock_sagemaker.list_model_packages.return_value = {
        "ModelPackageSummaryList": [
            {
                "ModelPackageArn": "arn:aws:sagemaker:::model-package/group/v2",
                "CreationTime": datetime.now(),
                "ModelPackageStatus": "Completed",
                "ModelApprovalStatus": "Approved",
            },
            {
                "ModelPackageArn": "arn:aws:sagemaker:::model-package/group/v1",
                "CreationTime": datetime.now(),
                "ModelPackageStatus": "Completed",
                "ModelApprovalStatus": "PendingManualApproval",
            },
        ]
    }
    mock_boto.return_value = mock_sagemaker

    registry = ModelRegistry("test-group")
    registry.sagemaker = mock_sagemaker

    registry.rollback_to_version("v1")

    assert mock_sagemaker.update_model_package.call_count == 2


@patch("boto3.client")
def test_calibration_job_submit(mock_boto):
    mock_sagemaker = MagicMock()
    mock_sagemaker.create_processing_job.return_value = {
        "ProcessingJobArn": "arn:aws:sagemaker:job"
    }
    mock_boto.return_value = mock_sagemaker

    runner = CalibrationJobRunner()
    runner.sagemaker = mock_sagemaker

    config = CalibrationJobConfig(
        training_data_s3_uri="s3://bucket/train.jsonl",
        output_s3_uri="s3://bucket/output",
        role_arn="arn:aws:iam::123456789012:role/SageMakerRole",
    )

    job_name = runner.submit_calibration_job(config)

    assert "sme-bridge-calibration" in job_name
    mock_sagemaker.create_processing_job.assert_called_once()


@patch("boto3.client")
def test_calibration_job_poll_status(mock_boto):
    mock_sagemaker = MagicMock()
    mock_sagemaker.describe_processing_job.return_value = {"ProcessingJobStatus": "Completed"}
    mock_boto.return_value = mock_sagemaker

    runner = CalibrationJobRunner()
    runner.sagemaker = mock_sagemaker

    status = runner.poll_job_status("test-job", poll_interval_seconds=0.001)

    assert status == "Completed"


def test_export_qa_pairs():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "qa_pairs.jsonl"
        qa_pairs = [
            {"query": "What is X?", "answer": "X is Y"},
            {"query": "What is Z?", "answer": "Z is W"},
        ]

        export_qa_pairs_for_finetuning(qa_pairs, str(path))

        assert path.exists()
        with path.open("r") as f:
            lines = f.readlines()
        assert len(lines) == 2
