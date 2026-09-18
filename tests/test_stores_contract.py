# Copyright 2026 Vikash Ranjan (CTO, styls.ai). Licensed under the Apache License, Version 2.0.
"""Every RevisionStore adapter passes the same contract (REQUIREMENTS §6.5, verbatim): ordered chain, digest identity,
refusal to fork. Acceptance §8.3. Postgres and S3 run under testcontainers / moto and are skipped, visibly, without
Docker or the extra."""

import os

import pytest

from proceduralgraph.documents import HeadMoved
from proceduralgraph.graph import Graph
from proceduralgraph.rejections import RejectionMemory
from proceduralgraph.stores.blob import BlobRevisionStore, MemoryBlobStore
from proceduralgraph.stores.file import FileRevisionStore, export_workspace
from proceduralgraph.stores.memory import MemoryRevisionStore


async def contract(store):
    assert await store.head("ws", "graph") is None
    first = await store.append("ws", "graph", {"n": 1, "text": "é"}, expected_head=None, meta={"iteration": 1})
    assert first.seq == 0 and first.parent_digest is None and first.meta == {"iteration": 1}
    second = await store.append("ws", "graph", {"n": 2}, expected_head=first.digest)
    assert second.seq == 1 and second.parent_digest == first.digest
    head = await store.head("ws", "graph")
    assert head.digest == second.digest and head.document == {"n": 2}
    head.verify()
    with pytest.raises(HeadMoved):
        await store.append("ws", "graph", {"n": 3}, expected_head=first.digest)
    with pytest.raises(HeadMoved):
        await store.append("ws", "graph", {"n": 3}, expected_head=None)
    assert [r.seq for r in await store.list("ws", "graph")] == [1, 0]
    assert [r.seq for r in await store.list("ws", "graph", limit=1)] == [1]
    assert (await store.get("ws", "graph", first.digest)).document == {"n": 1, "text": "é"}
    assert await store.get("ws", "graph", "0" * 64) is None
    assert await store.head("ws", "rejections") is None and await store.head("other", "graph") is None
    other = await store.append("other", "graph", {"n": 1}, expected_head=None)
    assert other.seq == 0


async def test_memory_store_contract():
    await contract(MemoryRevisionStore())


async def test_file_store_contract_and_export(tmp_path):
    store = FileRevisionStore(tmp_path / "revisions")
    await contract(store)
    assert (tmp_path / "revisions" / "ws" / "graph" / "HEAD").read_text().startswith("000001-")
    written = export_workspace(tmp_path / "workspace", graph=Graph.skeleton(), rejections=RejectionMemory())
    names = sorted(str(p.relative_to(tmp_path / "workspace")) for p in written)
    assert names == ["graph.json", "graph.md", "graph.mmd", "rejections.md"]
    assert (tmp_path / "workspace" / "graph.mmd").read_text().startswith("flowchart LR")


async def test_blob_store_contract_and_race():
    blobs = MemoryBlobStore()
    store = BlobRevisionStore(blobs, prefix="p")
    await contract(store)
    head = await store.head("ws", "graph")
    racer = BlobRevisionStore(blobs, prefix="p")
    await racer.append("ws", "graph", {"n": 99}, expected_head=head.digest)
    with pytest.raises(HeadMoved):
        await store.append("ws", "graph", {"n": 100}, expected_head=head.digest)
    assert (await store.head("ws", "graph")).document == {"n": 99}


@pytest.mark.postgres
async def test_postgres_store_contract():
    pytest.importorskip("asyncpg")
    pytest.importorskip("testcontainers")
    try:
        from testcontainers.community.postgres import PostgresContainer
    except ImportError:  # older testcontainers
        from testcontainers.postgres import PostgresContainer

    from proceduralgraph.stores.postgres import PostgresRevisionStore

    if os.environ.get("PROCEDURALGRAPH_SKIP_DOCKER"):
        pytest.skip("PROCEDURALGRAPH_SKIP_DOCKER set")
    try:
        import docker

        docker.from_env().ping()
    except Exception as exc:  # noqa: BLE001 - any Docker failure means "skip, and say so"
        pytest.skip(f"Docker is not available: {exc}")
    with PostgresContainer("postgres:16-alpine", driver="asyncpg") as pg:
        store = PostgresRevisionStore.from_url(pg.get_connection_url(), table="custom_revisions")
        await store.create_tables()
        await contract(store)
        # The primary key refuses the fork even when the head check is bypassed by a racing writer.
        head = await store.head("ws", "graph")
        racer = PostgresRevisionStore(store.engine, table="custom_revisions")
        await racer.append("ws", "graph", {"n": 99}, expected_head=head.digest)
        with pytest.raises(HeadMoved):
            await store.append("ws", "graph", {"n": 100}, expected_head=head.digest)
        await store.engine.dispose()


@pytest.mark.s3
async def test_s3_store_contract():
    pytest.importorskip("boto3")
    moto = pytest.importorskip("moto")
    from proceduralgraph.stores.s3 import S3BlobStore

    with moto.mock_aws():
        import boto3

        client = boto3.client("s3", region_name="us-east-1")
        client.create_bucket(Bucket="pg-test")
        store = BlobRevisionStore(S3BlobStore("pg-test", client=client), prefix="p")
        await contract(store)
