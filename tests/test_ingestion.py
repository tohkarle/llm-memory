"""The ingestion pipeline: detection, lane selection, provenance, caching."""

from __future__ import annotations

import struct
import zlib
from collections.abc import Iterator
from pathlib import Path

import pytest

from memory import media, workqueue
from memory.api import MemoryAPI
from memory.ingestion import IngestionError, detect_media_type, sniff


@pytest.fixture
def api(tmp_path: Path) -> Iterator[MemoryAPI]:
    instance = MemoryAPI.initialize(tmp_path / "root")
    try:
        yield instance
    finally:
        instance.close()


@pytest.fixture(autouse=True)
def clear_probes() -> Iterator[None]:
    media.reset_probes()
    yield
    media.reset_probes()


def notes(tmp_path: Path, name: str = "notes.md") -> Path:
    path = tmp_path / name
    path.write_text(
        "\n\n".join(f"Paragraph {i} about uv packaging and pytest." for i in range(8)),
        encoding="utf-8",
    )
    return path


def png(tmp_path: Path, name: str = "shot.png") -> Path:
    def chunk(tag: bytes, data: bytes) -> bytes:
        body = tag + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body))

    path = tmp_path / name
    path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", 16, 8, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(b"".join(b"\x00" + b"\xff\x00\x00" * 16 for _ in range(8))))
        + chunk(b"IEND", b"")
    )
    return path


def wav(tmp_path: Path, name: str = "clip.wav") -> Path:
    data = b"\x00\x00" * 4000
    path = tmp_path / name
    path.write_bytes(
        b"RIFF"
        + struct.pack("<I", 36 + len(data))
        + b"WAVE"
        + b"fmt "
        + struct.pack("<IHHIIHH", 16, 1, 1, 8000, 16000, 2, 16)
        + b"data"
        + struct.pack("<I", len(data))
        + data
    )
    return path


class TestTypeDetection:
    @pytest.mark.parametrize(
        ("payload", "expected"),
        [
            (b"%PDF-1.7\n", "application/pdf"),
            (b"\x89PNG\r\n\x1a\n", "image/png"),
            (b"\xff\xd8\xff\xe0", "image/jpeg"),
            (b"GIF89a", "image/gif"),
            (b"OggS\x00", "audio/ogg"),
            (b"fLaC\x00", "audio/flac"),
            (b"ID3\x04", "audio/mpeg"),
        ],
    )
    def test_sniffs_by_magic_number(self, tmp_path: Path, payload: bytes, expected: str) -> None:
        path = tmp_path / "input.bin"
        path.write_bytes(payload + b"\x00" * 64)
        assert sniff(path) == expected

    def test_sniffs_riff_subtypes(self, tmp_path: Path) -> None:
        for marker, expected in ((b"WAVE", "audio/wav"), (b"AVI ", "video/x-msvideo")):
            path = tmp_path / f"{expected.replace('/', '-')}.bin"
            path.write_bytes(b"RIFF" + b"\x00" * 4 + marker + b"\x00" * 32)
            assert sniff(path) == expected

    def test_sniffs_mp4_brands(self, tmp_path: Path) -> None:
        path = tmp_path / "v.bin"
        path.write_bytes(b"\x00\x00\x00\x18ftypisom" + b"\x00" * 32)
        assert sniff(path) == "video/mp4"

    def test_content_beats_a_lying_extension(self, tmp_path: Path) -> None:
        """A .txt that is really a PDF must not reach the text processor."""
        path = tmp_path / "actually.txt"
        path.write_bytes(b"%PDF-1.4\n" + b"\x00" * 32)
        assert detect_media_type(path) == "application/pdf"

    def test_caller_declaration_wins(self, tmp_path: Path) -> None:
        path = tmp_path / "a.png"
        path.write_bytes(b"\x89PNG\r\n\x1a\n")
        assert detect_media_type(path, declared="image/custom") == "image/custom"

    def test_falls_back_to_the_extension(self, tmp_path: Path) -> None:
        path = tmp_path / "plain.md"
        path.write_text("# heading", encoding="utf-8")
        assert detect_media_type(path) in ("text/markdown", "text/x-markdown")

    def test_unknown_is_octet_stream(self, tmp_path: Path) -> None:
        path = tmp_path / "mystery.zzz"
        path.write_bytes(b"\x07\x08\x09")
        assert detect_media_type(path) == "application/octet-stream"

    def test_empty_file_is_not_sniffable(self, tmp_path: Path) -> None:
        path = tmp_path / "empty.bin"
        path.write_bytes(b"")
        assert sniff(path) is None


class TestInlineIngestion:
    def test_text_is_processed_inline(self, api: MemoryAPI, tmp_path: Path) -> None:
        result = api.ingest(notes(tmp_path))
        assert result["status"] == "complete"
        assert result["processor"] == "text"
        assert result["cost"] == "cheap"
        assert result["segments"] >= 1
        assert result["queued_task"] is None

    def test_the_original_is_stored_unmodified(self, api: MemoryAPI, tmp_path: Path) -> None:
        source = notes(tmp_path)
        original = source.read_bytes()
        result = api.ingest(source)
        blob = api.paths.asset_blob(result["sha256"])
        assert blob.read_bytes() == original
        assert source.read_bytes() == original, "the source file must not be touched"

    def test_segments_are_searchable(self, api: MemoryAPI, tmp_path: Path) -> None:
        api.ingest(notes(tmp_path))
        hits = api.search_segments("packaging")
        assert hits and "packaging" in hits[0]["text"]

    def test_a_hit_carries_its_provenance(self, api: MemoryAPI, tmp_path: Path) -> None:
        result = api.ingest(notes(tmp_path))
        hit = api.search_segments("packaging")[0]
        assert hit["asset_id"] == result["asset_id"]
        assert hit["derivation_id"] == result["derivation_id"]
        assert hit["processor"] == "text"
        assert hit["media_type"] == result["media_type"]

    def test_no_process_stores_without_deriving(self, api: MemoryAPI, tmp_path: Path) -> None:
        result = api.ingest(notes(tmp_path), process=False)
        assert result["status"] == "stored"
        assert result["segments"] == 0
        assert api.get_asset(result["asset_id"]) is not None

    def test_an_unsupported_type_is_still_stored(self, api: MemoryAPI, tmp_path: Path) -> None:
        path = tmp_path / "mystery.zzz"
        path.write_bytes(b"\x07\x08\x09binary payload")
        result = api.ingest(path)
        assert result["status"] == "unsupported"
        assert api.get_asset(result["asset_id"]) is not None
        assert result["warnings"]

    def test_tags_and_metadata_reach_the_asset(self, api: MemoryAPI, tmp_path: Path) -> None:
        result = api.ingest(notes(tmp_path), tags=["research"], meta={"project": "phase3"})
        asset = api.get_asset(result["asset_id"])
        assert asset is not None
        assert asset["tags"] == ["research"]
        assert asset["meta"]["project"] == "phase3"

    def test_missing_file_raises(self, api: MemoryAPI, tmp_path: Path) -> None:
        with pytest.raises(IngestionError, match="not a file"):
            api.ingest(tmp_path / "nope.md")

    def test_image_degrades_but_still_registers(self, api: MemoryAPI, tmp_path: Path) -> None:
        result = api.ingest(png(tmp_path))
        assert result["processor"] == "image"
        assert result["status"] in ("complete", "degraded")
        assert result["derivation_id"] is not None


class TestDerivationCache:
    def test_reingesting_the_same_file_reuses_the_derivation(
        self, api: MemoryAPI, tmp_path: Path
    ) -> None:
        """What keeps a 40-minute transcription from running twice."""
        source = notes(tmp_path)
        first = api.ingest(source)
        asset_id = first["asset_id"]
        again = api.process_asset(asset_id)
        assert again["status"] == "cached"
        assert again["derivation_id"] == first["derivation_id"]

    def test_force_bypasses_the_cache(self, api: MemoryAPI, tmp_path: Path) -> None:
        first = api.ingest(notes(tmp_path))
        again = api.process_asset(first["asset_id"], force=True)
        assert again["status"] in ("complete", "degraded")
        assert again["derivation_id"] == first["derivation_id"]

    def test_different_params_are_a_different_derivation(
        self, api: MemoryAPI, tmp_path: Path
    ) -> None:
        first = api.ingest(notes(tmp_path))
        tuned = api.process_asset(first["asset_id"], params={"target_chars": 200})
        assert tuned["derivation_id"] != first["derivation_id"]

    def test_reprocessing_does_not_leave_phantom_segments(
        self, api: MemoryAPI, tmp_path: Path
    ) -> None:
        """Fewer segments the second time must not leave the surplus indexed."""
        first = api.ingest(notes(tmp_path), params={"target_chars": 200})
        many = first["segments"]
        assert many > 1
        second = api.process_asset(first["asset_id"], params={"target_chars": 200}, force=True)
        rows = api.conn.execute(
            "SELECT count(*) c FROM segments WHERE derivation_id = ?",
            (second["derivation_id"],),
        ).fetchone()["c"]
        assert rows == second["segments"]


class TestDeferredLane:
    def test_an_expensive_processor_is_queued(self, api: MemoryAPI, tmp_path: Path) -> None:
        result = api.ingest(wav(tmp_path))
        assert result["status"] == "queued"
        assert result["cost"] == "expensive"
        assert result["queued_task"]
        assert api.queue_stats()["pending"] == 1

    def test_inline_overrides_the_cost_hint(self, api: MemoryAPI, tmp_path: Path) -> None:
        result = api.ingest(wav(tmp_path), defer=False)
        assert result["status"] != "queued"
        assert api.queue_stats()["pending"] == 0

    def test_defer_overrides_a_cheap_processor(self, api: MemoryAPI, tmp_path: Path) -> None:
        result = api.ingest(notes(tmp_path), defer=True)
        assert result["status"] == "queued"

    def test_requeueing_one_asset_does_not_queue_twice(
        self, api: MemoryAPI, tmp_path: Path
    ) -> None:
        """The dedupe key is per asset+processor+version.

        Note that two ``ingest`` calls on the same file are two *assets* by
        design (``store_asset`` mints a new id per import, because the same
        PDF can be two different things in the user's world), so deduping is
        tested by re-processing one asset rather than re-ingesting the file.
        """
        first = api.ingest(wav(tmp_path))
        again = api.process_asset(first["asset_id"], defer=True)
        assert api.queue_stats()["pending"] == 1
        assert again["queued_task"] == first["queued_task"]

    def test_two_imports_of_one_file_are_two_assets(self, api: MemoryAPI, tmp_path: Path) -> None:
        source = wav(tmp_path)
        first = api.ingest(source)
        second = api.ingest(source)
        assert first["asset_id"] != second["asset_id"]
        assert first["sha256"] == second["sha256"], "identical bytes share one blob"
        assert api.queue_stats()["pending"] == 2

    def test_running_the_queue_produces_the_derivation(
        self, api: MemoryAPI, tmp_path: Path
    ) -> None:
        api.ingest(notes(tmp_path), defer=True)
        report = api.queue_run()
        assert report["processed"] == 1
        assert len(report["completed"]) == 1
        assert api.queue_stats()["pending"] == 0
        assert api.search_segments("packaging")


class TestDirectoryIngestion:
    def test_ingests_every_file(self, api: MemoryAPI, tmp_path: Path) -> None:
        folder = tmp_path / "docs"
        folder.mkdir()
        for i in range(3):
            (folder / f"n{i}.md").write_text(f"Note {i} about uv.", encoding="utf-8")
        results = api.ingest_directory(folder)
        assert len(results) == 3
        assert all(r["status"] == "complete" for r in results)

    def test_skips_hidden_files(self, api: MemoryAPI, tmp_path: Path) -> None:
        folder = tmp_path / "docs"
        folder.mkdir()
        (folder / "visible.md").write_text("content", encoding="utf-8")
        (folder / ".hidden.md").write_text("secret", encoding="utf-8")
        assert len(api.ingest_directory(folder)) == 1

    def test_one_bad_file_does_not_abort_the_batch(
        self, api: MemoryAPI, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        folder = tmp_path / "docs"
        folder.mkdir()
        for i in range(3):
            (folder / f"n{i}.md").write_text(f"Note {i}.", encoding="utf-8")

        from memory import ingestion

        real = ingestion.ingest
        calls = {"n": 0}

        def flaky(api_: MemoryAPI, path: Path, **kw: object) -> object:
            calls["n"] += 1
            if calls["n"] == 2:
                raise RuntimeError("simulated failure")
            return real(api_, path, **kw)  # type: ignore[arg-type]

        monkeypatch.setattr(ingestion, "ingest", flaky)
        results = ingestion.ingest_directory(api, folder)
        assert len(results) == 3
        assert sum(r.status == "error" for r in results) == 1

    def test_not_a_directory_raises(self, api: MemoryAPI, tmp_path: Path) -> None:
        with pytest.raises(IngestionError, match="not a directory"):
            api.ingest_directory(notes(tmp_path))


class TestSegmentRetrieval:
    def test_lists_an_assets_segments_in_order(self, api: MemoryAPI, tmp_path: Path) -> None:
        result = api.ingest(notes(tmp_path), params={"target_chars": 200})
        rows = api.asset_segments(result["asset_id"])
        assert [r["ordinal"] for r in rows] == list(range(len(rows)))

    def test_get_segment_returns_the_provenance_chain(self, api: MemoryAPI, tmp_path: Path) -> None:
        api.ingest(notes(tmp_path))
        hit = api.search_segments("packaging")[0]
        full = api.get_segment(hit["segment_id"])
        assert full is not None
        assert full["processor"] == "text"
        assert full["sha256"]

    def test_unknown_segment_is_none(self, api: MemoryAPI) -> None:
        assert api.get_segment("00000000-0000-4000-8000-000000000000") is None

    def test_search_can_be_scoped_to_one_asset(self, api: MemoryAPI, tmp_path: Path) -> None:
        first = api.ingest(notes(tmp_path, "a.md"))
        second_path = tmp_path / "b.md"
        second_path.write_text("Totally different packaging content here.", encoding="utf-8")
        api.ingest(second_path)
        scoped = api.search_segments("packaging", asset_id=first["asset_id"])
        assert scoped and all(h["asset_id"] == first["asset_id"] for h in scoped)

    def test_a_query_matching_nothing_returns_nothing(self, api: MemoryAPI, tmp_path: Path) -> None:
        api.ingest(notes(tmp_path))
        assert api.search_segments("zzzznotpresent") == []


class TestQueueDurability:
    """A worker dying must never lose or strand a task."""

    def test_a_task_survives_a_closed_connection(self, tmp_path: Path) -> None:
        root = tmp_path / "root"
        first = MemoryAPI.initialize(root)
        task_id = first.ingest(wav(tmp_path))["queued_task"]
        first.close()

        with MemoryAPI(root) as reopened:
            assert workqueue.get(reopened.conn, str(task_id)) is not None
            assert reopened.queue_stats()["pending"] == 1

    def test_an_expired_lease_is_reclaimed(self, api: MemoryAPI, tmp_path: Path) -> None:
        api.ingest(notes(tmp_path), defer=True)
        claimed = workqueue.claim(api.conn, lease_s=3600)
        assert claimed is not None

        # Simulate a worker killed mid-task: the row stays 'running' with a
        # lease that then expires.
        api.conn.execute(
            "UPDATE work_queue SET lease_expires_at = '2000-01-01T00:00:00Z' WHERE id = ?",
            (claimed.id,),
        )
        api.conn.commit()
        assert workqueue.reap_expired(api.conn) == 1
        assert workqueue.get(api.conn, claimed.id).status == "pending"  # type: ignore[union-attr]

    def test_attempts_increment_at_claim_time(self, api: MemoryAPI, tmp_path: Path) -> None:
        """So a task that hard-crashes the interpreter still counts its try."""
        api.ingest(notes(tmp_path), defer=True)
        claimed = workqueue.claim(api.conn)
        assert claimed is not None and claimed.attempts == 1

    def test_a_failure_schedules_a_backed_off_retry(self, api: MemoryAPI, tmp_path: Path) -> None:
        api.ingest(notes(tmp_path), defer=True)
        claimed = workqueue.claim(api.conn)
        assert claimed is not None
        after = workqueue.fail(api.conn, claimed.id, "boom")
        assert after is not None
        assert after.status == "pending"
        assert after.not_before is not None
        assert after.last_error == "boom"

    def test_attempts_are_exhausted_then_it_fails(self, api: MemoryAPI, tmp_path: Path) -> None:
        api.ingest(notes(tmp_path), defer=True)
        task = workqueue.listing(api.conn)[0]
        api.conn.execute("UPDATE work_queue SET max_attempts = 1 WHERE id = ?", (task.id,))
        api.conn.commit()
        claimed = workqueue.claim(api.conn)
        assert claimed is not None
        after = workqueue.fail(api.conn, claimed.id, "boom")
        assert after is not None and after.status == "failed"

    def test_retry_resets_a_failed_task(self, api: MemoryAPI, tmp_path: Path) -> None:
        api.ingest(notes(tmp_path), defer=True)
        task = workqueue.listing(api.conn)[0]
        api.conn.execute(
            "UPDATE work_queue SET status='failed', attempts=3 WHERE id = ?", (task.id,)
        )
        api.conn.commit()
        report = api.queue_retry(all_failed=True)
        assert report["count"] == 1
        revived = workqueue.get(api.conn, task.id)
        assert revived is not None and revived.status == "pending" and revived.attempts == 0

    def test_a_poison_task_does_not_stop_the_drain(self, api: MemoryAPI, tmp_path: Path) -> None:
        workqueue.enqueue(
            api.conn,
            kind="process_asset",
            asset_id="00000000-0000-4000-8000-000000000000",
            payload={"asset_id": "00000000-0000-4000-8000-000000000000"},
        )
        api.ingest(notes(tmp_path), defer=True)
        report = api.queue_run()
        assert report["processed"] == 2
        assert len(report["completed"]) == 1
        assert len(report["errors"]) == 1

    def test_cancel_stops_a_pending_task(self, api: MemoryAPI, tmp_path: Path) -> None:
        task_id = api.ingest(wav(tmp_path))["queued_task"]
        assert api.queue_cancel(str(task_id))["cancelled"] is True
        assert api.queue_stats()["pending"] == 0

    def test_purge_keeps_unfinished_work(self, api: MemoryAPI, tmp_path: Path) -> None:
        api.ingest(wav(tmp_path))
        assert api.queue_purge(keep_days=0)["removed"] == 0
        assert api.queue_stats()["pending"] == 1

    def test_an_unknown_task_kind_fails_loudly(self, api: MemoryAPI) -> None:
        workqueue.enqueue(api.conn, kind="not_a_real_kind", payload={})
        report = api.queue_run()
        assert len(report["errors"]) == 1
        assert "unknown task kind" in report["errors"][0]["error"]
