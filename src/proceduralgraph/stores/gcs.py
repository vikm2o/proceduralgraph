# Copyright 2026 Vikash Ranjan (CTO, styls.ai). Licensed under the Apache License, Version 2.0.
"""Google Cloud Storage ``BlobStore`` (``pip install proceduralgraph[gcs]``). Conditional writes use object generations:
``if_generation_match=0`` creates only when absent, ``if_generation_match=<generation>`` replaces one known version."""

from __future__ import annotations

import asyncio


class GCSBlobStore:
    def __init__(self, bucket: str, *, client=None):
        from google.cloud import storage

        self.client = client or storage.Client()
        self.bucket = self.client.bucket(bucket)

    async def get(self, key: str) -> tuple[bytes, str] | None:
        def call():
            blob = self.bucket.get_blob(key)
            if blob is None:
                return None
            return blob.download_as_bytes(), str(blob.generation)

        return await asyncio.to_thread(call)

    async def _put(self, key: str, data: bytes, generation: int) -> bool:
        from google.api_core.exceptions import PreconditionFailed

        def call():
            try:
                self.bucket.blob(key).upload_from_string(data, content_type="application/json", if_generation_match=generation)
                return True
            except PreconditionFailed:
                return False

        return await asyncio.to_thread(call)

    async def put_if_absent(self, key: str, data: bytes) -> bool:
        return await self._put(key, data, 0)

    async def put_if_match(self, key: str, data: bytes, *, tag: str) -> bool:
        return await self._put(key, data, int(tag))

    async def list(self, prefix: str) -> list[str]:
        return await asyncio.to_thread(lambda: sorted(blob.name for blob in self.client.list_blobs(self.bucket, prefix=prefix)))


__all__ = ["GCSBlobStore"]
