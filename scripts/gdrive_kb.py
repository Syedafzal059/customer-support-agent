"""
Phase 1–2 — Google Drive PDF ingestion + paragraph-aware chunking.
Lists PDFs, downloads changed files, extracts text, and chunks into stable
chunk dicts. Returns metadata + text for sync_pipeline.py to embed and index.

Setup:
    pip install google-api-python-client google-auth pypdf

Usage (from repo root):
    python scripts/gdrive_kb.py
    python scripts/gdrive_kb.py --check

Returns from run():
    (results, removal_ids, text_lookup)
    - results: files needing re-embedding (NEW / CHANGED / REPLACED)
    - removal_ids: file_ids whose old chunks must be deleted
    - text_lookup: {file_id: extracted_text} for sync_pipeline.sync_to_faiss

Requires:
    - scripts/customer-support-agent-501616-534fe1748725.json (or GDRIVE_SA_FILE)
    - GDRIVE_FOLDER_ID env var (folder shared with the service account)
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from chunker import chunk_text, make_chunk_id  # noqa: E402
from dotenv import load_dotenv
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from pypdf import PdfReader

SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]
DEFAULT_SA_FILENAME = "customer-support-agent-501616-534fe1748725.json"
DEFAULT_SYNC_STATE_FILE = "data/gdrive_sync_state.json"


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _scripts_dir() -> Path:
    return Path(__file__).resolve().parent


def _default_service_account_path() -> Path:
    return _scripts_dir() / DEFAULT_SA_FILENAME


def _service_account_path() -> Path:
    override = os.environ.get("GDRIVE_SA_FILE", "").strip()
    if override:
        path = Path(override)
        if not path.is_absolute():
            path = _project_root() / path
        return path
    return _default_service_account_path()


def _folder_id() -> str:
    return os.environ.get("GDRIVE_FOLDER_ID", "").strip()


def _sync_state_path() -> Path:
    override = os.environ.get("SYNC_STATE_FILE", DEFAULT_SYNC_STATE_FILE).strip()
    path = Path(override)
    if not path.is_absolute():
        path = _project_root() / path
    return path


def load_sync_state() -> dict[str, Any]:
    path = _sync_state_path()
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def save_sync_state(state: dict[str, Any]) -> None:
    path = _sync_state_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2), encoding="utf-8")


def hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def chunk_ids_for_file(file_id: str, chunk_count: int) -> list[int]:
    """Reconstruct deterministic chunk IDs from a prior sync entry."""
    return [make_chunk_id(file_id, i) for i in range(chunk_count)]


def get_drive_service():
    sa_path = _service_account_path()
    if not sa_path.is_file():
        raise SystemExit(
            f"Service account file not found: {sa_path}\n"
            "Set GDRIVE_SA_FILE or place the JSON under scripts/."
        )
    creds = service_account.Credentials.from_service_account_file(str(sa_path), scopes=SCOPES)
    return build("drive", "v3", credentials=creds), sa_path


def check_connection() -> int:
    """Verify credentials and Drive API access without needing a folder ID."""
    service, sa_path = get_drive_service()
    print(f"Service account file: {sa_path}")
    try:
        about = service.about().get(fields="user,storageQuota").execute()
    except HttpError as exc:
        if exc.resp.status == 403 and "accessNotConfigured" in str(exc):
            print(
                "FAIL  Google Drive API is not enabled for this GCP project.\n"
                "      Enable it: https://console.cloud.google.com/apis/library/drive.googleapis.com"
                "?project=customer-support-agent-501616\n"
                "      Wait 1–2 minutes, then rerun: python scripts/gdrive_kb.py --check",
                file=sys.stderr,
            )
            return 1
        print(f"FAIL  Drive API auth/connection: {exc}", file=sys.stderr)
        return 1

    user = about.get("user", {})
    email = user.get("emailAddress", "(unknown)")
    print(f"OK    Authenticated as: {email}")

    try:
        response = (
            service.files()
            .list(pageSize=5, fields="files(id, name, mimeType)", orderBy="modifiedTime desc")
            .execute()
        )
    except HttpError as exc:
        print(f"FAIL  Drive files.list: {exc}", file=sys.stderr)
        return 1

    files = response.get("files", [])
    print(f"OK    files.list returned {len(files)} file(s) visible to this account")
    for item in files:
        print(f"      - {item.get('name')} ({item.get('mimeType')})")
    return 0


def list_pdfs(service, folder_id: str):
    """Paginate through every PDF in the folder. Never assumes one page is enough."""
    query = f"'{folder_id}' in parents and mimeType='application/pdf' and trashed=false"
    files, page_token = [], None
    while True:
        response = (
            service.files()
            .list(
                q=query,
                fields="nextPageToken, files(id, name, modifiedTime)",
                pageToken=page_token,
            )
            .execute()
        )
        files.extend(response.get("files", []))
        page_token = response.get("nextPageToken")
        if not page_token:
            break
    return files


def download_pdf(service, file_id: str):
    return service.files().get_media(fileId=file_id).execute()


def extract_text(pdf_bytes: bytes, filename: str):
    """Returns extracted text, or None if the file is unusable (scanned, corrupt, empty)."""
    try:
        reader = PdfReader(io.BytesIO(pdf_bytes))
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
        if not text.strip():
            print(f"  SKIP  {filename}: extracted empty text (likely scanned/image-only)")
            return None
        return text
    except Exception as exc:
        print(f"  SKIP  {filename}: failed to parse ({exc})")
        return None


def run() -> tuple[list[dict], list[str], dict[str, str]]:
    folder_id = _folder_id()
    if not folder_id:
        raise SystemExit(
            "Set GDRIVE_FOLDER_ID in .env or the environment.\n"
            "  Copy the ID from your Drive folder URL: "
            "https://drive.google.com/drive/folders/<FOLDER_ID>\n"
            "Share the folder with: drive-sync-bot@customer-support-agent-501616.iam.gserviceaccount.com"
        )

    service, sa_path = get_drive_service()
    print(f"Service account file: {sa_path}")
    print(f"Sync state file: {_sync_state_path()}")
    print(f"Listing PDFs in folder {folder_id} ...")

    try:
        files = list_pdfs(service, folder_id)
    except HttpError as exc:
        raise SystemExit(f"Drive API error while listing files: {exc}") from exc

    print(f"Found {len(files)} PDF(s).\n")

    sync_state = load_sync_state()
    results: list[dict] = []
    removal_ids: list[str] = []
    text_lookup: dict[str, str] = {}
    unchanged_count = 0

    current_file_ids = {file_meta["id"] for file_meta in files}
    orphaned_ids = {file_id for file_id in sync_state if file_id not in current_file_ids}
    orphaned_by_name: dict[str, str] = {
        sync_state[file_id]["name"]: file_id
        for file_id in orphaned_ids
        if sync_state[file_id].get("name")
    }

    for file_meta in files:
        file_id = file_meta["id"]
        name = file_meta["name"]
        modified_time = file_meta["modifiedTime"]
        prior = sync_state.get(file_id)

        if prior and prior.get("modified_time") == modified_time:
            unchanged_count += 1
            print(f"SKIP  {name}: unchanged (modifiedTime)")
            continue

        print(f"Processing: {name}")
        try:
            pdf_bytes = download_pdf(service, file_id)
        except HttpError as exc:
            print(f"  SKIP  {name}: download failed ({exc})")
            continue

        text = extract_text(pdf_bytes, name)
        if text is None:
            continue

        new_hash = hash_text(text)

        if prior and prior.get("hash") == new_hash:
            unchanged_count += 1
            sync_state[file_id] = {
                "hash": new_hash,
                "name": name,
                "modified_time": modified_time,
                "chunk_count": int(prior.get("chunk_count", 0)),
            }
            print(f"  SKIP  {name}: unchanged (hash {new_hash[:10]}..., modifiedTime drift)")
            continue

        chunks = chunk_text(text, file_id=file_id, source_name=name)

        if prior is None and name in orphaned_by_name:
            old_file_id = orphaned_by_name.pop(name)
            orphaned_ids.discard(old_file_id)
            old_entry = sync_state.pop(old_file_id)
            removal_ids.append(old_file_id)
            status = "REPLACED"
            print(
                f"  OK    {name}: REPLACED (old file_id {old_file_id} reconciled by name), "
                f"extracted {len(text)} chars, {len(chunks)} chunk(s), hash {new_hash[:10]}..."
            )
        elif prior is None:
            status = "NEW"
            print(
                f"  OK    {status}, extracted {len(text)} chars, "
                f"{len(chunks)} chunk(s), hash {new_hash[:10]}..."
            )
        else:
            status = "CHANGED"
            removal_ids.append(file_id)
            print(
                f"  OK    {status}, extracted {len(text)} chars, "
                f"{len(chunks)} chunk(s), hash {new_hash[:10]}..."
            )

        text_lookup[file_id] = text
        results.append(
            {
                "file_id": file_id,
                "name": name,
                "modified_time": modified_time,
                "char_count": len(text),
                "hash": new_hash,
                "status": status,
                "chunk_count": len(chunks),
            }
        )
        sync_state[file_id] = {
            "hash": new_hash,
            "name": name,
            "modified_time": modified_time,
            "chunk_count": len(chunks),
        }

    for file_id in orphaned_ids:
        entry = sync_state.pop(file_id)
        name = entry.get("name", file_id)
        chunk_count = int(entry.get("chunk_count", 0))
        removal_ids.append(file_id)
        print(
            f"DELETED  {name}: removed from sync (file_id {file_id}, {chunk_count} stale chunk(s))"
        )

    save_sync_state(sync_state)

    total_chunks = sum(r["chunk_count"] for r in results)
    print(
        f"\nDone. {len(results)} new/changed/replaced ({total_chunks} chunk(s)), "
        f"{len(removal_ids)} removed, {unchanged_count} unchanged (skipped), "
        f"out of {len(files)} total PDF(s)."
    )
    return results, removal_ids, text_lookup


def main() -> int:
    load_dotenv(_project_root() / ".env")
    if len(sys.argv) > 1 and sys.argv[1] == "--check":
        return check_connection()
    run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
