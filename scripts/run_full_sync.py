"""
CI / scheduled-run entrypoint: Google Drive -> chunk -> embed -> FAISS in one shot.

GitHub Actions invokes this script on a schedule; it must run unattended with no
prompts or interactive input.

skip_deletion is hardcoded to False here (production sync always purges stale
chunks). The only place skip_deletion=True is used is
scripts/regression_stale_chunk_demo.py (demo / regression mode).
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import gdrive_kb  # noqa: E402
from chunk_store import get_chunk_count, init_db  # noqa: E402
from dotenv import load_dotenv  # noqa: E402
from embedder import get_embedding_dim  # noqa: E402
from faiss_store import load_index  # noqa: E402
from sync_pipeline import sync_to_faiss  # noqa: E402

DEFAULT_FAISS_INDEX_PATH = "faiss_index.bin"
_LOG = logging.getLogger("run_full_sync")


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _faiss_index_path() -> str:
    return os.environ.get("FAISS_INDEX_PATH", DEFAULT_FAISS_INDEX_PATH).strip() or (
        DEFAULT_FAISS_INDEX_PATH
    )


def _configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [run_full_sync] %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )


def _log_config() -> None:
    _LOG.info("Config: GDRIVE_SA_FILE=%s", os.environ.get("GDRIVE_SA_FILE", "(default)"))
    _LOG.info("Config: GDRIVE_FOLDER_ID=%s", os.environ.get("GDRIVE_FOLDER_ID", "(unset)"))
    _LOG.info("Config: SYNC_STATE_FILE=%s", os.environ.get("SYNC_STATE_FILE", "(default)"))
    _LOG.info("Config: FAISS_INDEX_PATH=%s", _faiss_index_path())
    _LOG.info("Config: CHUNK_STORE_DB=%s", os.environ.get("CHUNK_STORE_DB", "(default)"))


def _format_files_summary(results: list[dict]) -> str:
    if not results:
        return "(none)"
    return ", ".join(
        f"{entry.get('name', entry.get('file_id'))} ({entry.get('status', '?')})"
        for entry in results
    )


def _verify_store_faiss_parity() -> bool:
    init_db()
    store_count = get_chunk_count()
    index = load_index(_faiss_index_path(), dim=get_embedding_dim())
    faiss_count = int(index.ntotal)
    if store_count == faiss_count:
        _LOG.info(
            "Sanity check PASS: chunk_store rows=%d, FAISS ntotal=%d",
            store_count,
            faiss_count,
        )
        return True

    _LOG.error(
        "Sanity check FAIL: chunk_store rows=%d != FAISS ntotal=%d — "
        "index and metadata store are out of sync; refusing to treat run as successful",
        store_count,
        faiss_count,
    )
    return False


def main() -> int:
    load_dotenv(_project_root() / ".env")
    _configure_logging()

    _LOG.info("=== Full sync started ===")
    _log_config()

    try:
        results, removals, text_lookup = gdrive_kb.run()
    except SystemExit as exc:
        _LOG.error("Google Drive ingestion failed: %s", exc)
        return 1
    except Exception:
        _LOG.exception("Google Drive ingestion failed with an unexpected error")
        return 1

    if not results and not removals:
        _LOG.info("No changes detected, nothing to sync.")
        return 0

    _LOG.info(
        "Drive ingestion complete: %d file(s) to add/update, %d file(s) to remove",
        len(results),
        len(removals),
    )
    if results:
        _LOG.info("Files to index: %s", _format_files_summary(results))
    if removals:
        _LOG.info("File IDs to remove: %s", ", ".join(removals))

    try:
        sync_to_faiss(results, removals, text_lookup, skip_deletion=False)
    except SystemExit as exc:
        _LOG.error(
            "FAISS sync failed while processing [%s]: %s",
            _format_files_summary(results),
            exc,
        )
        return 1
    except Exception:
        _LOG.exception(
            "FAISS sync failed while processing [%s]",
            _format_files_summary(results),
        )
        return 1

    by_status: dict[str, int] = {}
    for entry in results:
        status = entry.get("status", "UNKNOWN")
        by_status[status] = by_status.get(status, 0) + 1
    status_parts = ", ".join(f"{count} {status}" for status, count in sorted(by_status.items()))

    _LOG.info(
        "Sync finished: %d file(s) added/updated (%s), %d file(s) removed",
        len(results),
        status_parts or "n/a",
        len(removals),
    )

    if not _verify_store_faiss_parity():
        return 1

    _LOG.info("=== Full sync completed successfully ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
