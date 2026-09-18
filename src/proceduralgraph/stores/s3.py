# Copyright 2026 Vikash Ranjan (CTO, styls.ai). Licensed under the Apache License, Version 2.0.
"""S3-compatible ``BlobStore`` (``pip install proceduralgraph[s3]``). Conditional writes use ``If-None-Match: *`` and
``If-Match`` on PutObject; the bucket's provider must support them (AWS S3 does; check others)."""

from __future__ import annotations

import asyncio


class S3BlobStore:
    def __init__(self, bucket: str, *, client=None, **client_kwargs):
        import boto3

        self.bucket = bucket
        self.client = client or boto3.client("s3", **client_kwargs)

    @staticmethod
    def _precondition_failed(exc) -> bool:
        code = getattr(exc, "response", {}).get("Error", {}).get("Code")
        status = getattr(exc, "response", {}).get("ResponseMetadata", {}).get("HTTPStatusCode")
        return code in {"PreconditionFailed", "ConditionalRequestConflict"} or status in {412, 409}

    async def get(self, key: str) -> tuple[bytes, str] | None:
        from botocore.exceptions import ClientError

        def call():
            try:
                response = self.client.get_object(Bucket=self.bucket, Key=key)
            except ClientError as exc:
                if exc.response.get("Error", {}).get("Code") in {"NoSuchKey", "404"}:
                    return None
                raise
            return response["Body"].read(), response["ETag"]

        return await asyncio.to_thread(call)

    async def put_if_absent(self, key: str, data: bytes) -> bool:
        from botocore.exceptions import ClientError

        def call():
            try:
                self.client.put_object(Bucket=self.bucket, Key=key, Body=data, IfNoneMatch="*")
                return True
            except ClientError as exc:
                if self._precondition_failed(exc):
                    return False
                raise

        return await asyncio.to_thread(call)

    async def put_if_match(self, key: str, data: bytes, *, tag: str) -> bool:
        from botocore.exceptions import ClientError

        def call():
            try:
                self.client.put_object(Bucket=self.bucket, Key=key, Body=data, IfMatch=tag)
                return True
            except ClientError as exc:
                if self._precondition_failed(exc):
                    return False
                raise

        return await asyncio.to_thread(call)

    async def list(self, prefix: str) -> list[str]:
        def call():
            keys = []
            paginator = self.client.get_paginator("list_objects_v2")
            for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix):
                keys.extend(item["Key"] for item in page.get("Contents", []))
            return sorted(keys)

        return await asyncio.to_thread(call)


__all__ = ["S3BlobStore"]
