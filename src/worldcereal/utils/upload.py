from __future__ import (
    annotations,  # Required because of type annotations for return results
)

import datetime
from typing import TYPE_CHECKING, Optional

from catboost import CatBoostClassifier
from loguru import logger

if TYPE_CHECKING:
    from openeo.rest.connection import Connection

import hashlib
import os
from dataclasses import dataclass

import boto3
from boto3.s3.transfer import TransferConfig


@dataclass(frozen=True)
class S3URI:
    bucket: str
    key: str

    @classmethod
    def from_str(cls, uri: str) -> S3URI:
        s3_prefix = "s3://"
        if uri.startswith(s3_prefix):
            without_prefix = uri[len(s3_prefix) :]
            without_prefix_parts = without_prefix.split("/")
            bucket = without_prefix_parts[0]
            if len(without_prefix_parts) == 1:
                return S3URI(bucket, "")
            else:
                return S3URI(bucket, "/".join(without_prefix_parts[1:]))
        else:
            raise ValueError(
                "Input {uri} is not a valid S3 URI should be of form s3://<bucket>/<key>"
            )


@dataclass(frozen=True)
class AWSSTSCredentials:
    AWS_ACCESS_KEY_ID: str
    AWS_SECRET_ACCESS_KEY: str
    AWS_SESSION_TOKEN: str
    subject_from_web_identity_token: str
    STS_ENDPOINT = "https://sts.prod.warsaw.openeo.dataspace.copernicus.eu"

    @classmethod
    def _from_assume_role_response(cls, resp: dict) -> AWSSTSCredentials:
        d = resp["Credentials"]

        return AWSSTSCredentials(
            AWS_ACCESS_KEY_ID=d["AccessKeyId"],
            AWS_SECRET_ACCESS_KEY=d["SecretAccessKey"],
            AWS_SESSION_TOKEN=d["SessionToken"],
            subject_from_web_identity_token=resp["SubjectFromWebIdentityToken"],
        )

    def set_as_environment_variables(self) -> None:
        """If temporary credentials are to be used elsewhere in the notebook"""
        os.environ["AWS_ACCESS_KEY_ID"] = self.AWS_ACCESS_KEY_ID
        os.environ["AWS_SECRET_ACCESS_KEY"] = self.AWS_SECRET_ACCESS_KEY
        os.environ["AWS_SESSION_TOKEN"] = self.AWS_SESSION_TOKEN

    def as_kwargs(self) -> dict:
        return {
            "aws_access_key_id": self.AWS_ACCESS_KEY_ID,
            "aws_secret_access_key": self.AWS_SECRET_ACCESS_KEY,
            "aws_session_token": self.AWS_SESSION_TOKEN,
        }

    @classmethod
    def from_openeo_connection(cls, conn: Connection) -> AWSSTSCredentials:
        """
        Takes an OpenEO connection object and returns temporary credentials to interact with S3
        """
        auth_token = conn.auth.bearer.split("/")
        os.environ["AWS_ENDPOINT_URL_STS"] = cls.STS_ENDPOINT
        sts = boto3.client("sts")
        return AWSSTSCredentials._from_assume_role_response(
            sts.assume_role_with_web_identity(
                RoleArn="arn:aws:iam::000000000000:role/S3Access",
                RoleSessionName=auth_token[1],
                WebIdentityToken=auth_token[2],
                DurationSeconds=43200,
            )
        )

    def get_user_hash(self) -> str:
        hash_object = hashlib.sha1(self.subject_from_web_identity_token.encode())
        return hash_object.hexdigest()


class OpenEOArtifactHelper:
    BUCKET_NAME = "OpenEO-artifacts"
    S3_ENDPOINT = "https://s3.prod.warsaw.openeo.dataspace.copernicus.eu"
    # From what size will we switch to multi-part-upload
    MULTIPART_THRESHOLD_IN_MB = 50

    def __init__(self, creds: AWSSTSCredentials):
        self._creds = creds
        self.session = boto3.Session(**creds.as_kwargs())

    @classmethod
    def from_openeo_connection(cls, conn: Connection) -> OpenEOArtifactHelper:
        creds = AWSSTSCredentials.from_openeo_connection(conn)
        return OpenEOArtifactHelper(creds)

    def get_s3_client(self):
        return self.session.client("s3", endpoint_url=self.S3_ENDPOINT)

    def set_env(self):
        os.environ["AWS_ENDPOINT_URL_S3"] = self.S3_ENDPOINT

    def user_prefix(self) -> str:
        """Each user has its own prefix retrieve it"""
        return self._creds.get_user_hash()

    def get_upload_prefix(self):
        return (
            f"{self.user_prefix()}/{datetime.datetime.utcnow().strftime('%Y/%m/%d')}/"
        )

    def get_upload_key(self, object_name: str) -> str:
        return f"{self.get_upload_prefix()}{object_name}"

    def upload_bytes(self, object_name: str, blob: bytes) -> str:
        """Upload a bunch of bytes into an object and return an S3 URI to it"""
        bucket = self.BUCKET_NAME
        key = self.get_upload_key(object_name)
        self.get_s3_client().put_object(Body=blob, Bucket=bucket, Key=key)
        return f"s3://{bucket}/{key}"

    def upload_string(self, object_name: str, s: str, encoding: str = "utf-8") -> str:
        """Upload a string into an object"""
        return self.upload_bytes(object_name, s.encode(encoding))

    def upload_file(self, object_name: str, src_file_path: str) -> str:
        MB = 1024**2
        config = TransferConfig(multipart_threshold=self.MULTIPART_THRESHOLD_IN_MB * MB)

        bucket = self.BUCKET_NAME
        key = self.get_upload_key(object_name)

        self.get_s3_client().upload_file(src_file_path, bucket, key, Config=config)
        return f"s3://{bucket}/{key}"

    def get_presigned_url(
        self, s3_uri: str, expires_in_seconds: int = 3600 * 24
    ) -> str:
        typed_s3_uri = S3URI.from_str(s3_uri)
        return self.get_s3_client().generate_presigned_url(
            "get_object",
            Params={"Bucket": typed_s3_uri.bucket, "Key": typed_s3_uri.key},
            ExpiresIn=expires_in_seconds,
        )


def deploy_model(
    conn: Connection, model: CatBoostClassifier, pattern: Optional[str] = None
) -> str:
    """Method to deploy a CatBoost model to s3 using an existing openEO connection.

    Parameters
    ----------
    conn : Connection
        openEO connection to use for deployment
    model : CatBoostClassifier
        CatBoost model to deploy
    pattern : str, optional
        identifier to add to the model name, by default None

    Returns
    -------
    str
        presigned URL to the deployed model. Note that this URL is only valid for a limited time.
    """

    import tempfile

    if pattern is None:
        targetpath = "custommodel.onnx"
    else:
        targetpath = f"{pattern}_custommodel.onnx"

    # Use a temporary file to save the ONNX model
    with tempfile.NamedTemporaryFile(delete=False, suffix=".onnx") as temp_file:
        model_file_path = temp_file.name

        model.save_model(
            f"{model_file_path}",
            format="onnx",
            export_parameters={
                "onnx_domain": "ai.catboost",
                "onnx_model_version": 1,
                "onnx_doc_string": "custom model for crop classification using CatBoost",
                "onnx_graph_name": "CatBoostModel_for_MulticlassClassification",
            },
        )

        logger.info("Deploying model ...")

        artifact_helper = OpenEOArtifactHelper.from_openeo_connection(conn)
        normal_s3_uri = artifact_helper.upload_file(targetpath, model_file_path)
        presigned_uri = artifact_helper.get_presigned_url(normal_s3_uri)

    logger.info(f"Deployed to: {normal_s3_uri}")

    return presigned_uri
