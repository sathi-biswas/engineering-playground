"""Google Drive authentication, document loading, and FAISS/Chroma RAG retriever."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from config import get_settings
from utils.logger import get_logger
from utils.secrets import is_usable_secret

logger = get_logger(__name__)

_DRIVE_SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]


class GoogleDriveRAGError(RuntimeError):
    """Raised when Drive auth, download, or indexing fails."""


class GoogleDriveRAG:
    """Load bug-report documents from Google Drive and expose a RAG retriever.

    Authentication modes (first match wins):
    1. Service-account JSON (``type: service_account``)
    2. Existing OAuth user token (``credentials/token.json`` or authorized-user JSON)
    3. OAuth Desktop/Web client JSON (``installed`` / ``web``) → interactive consent

    When Drive credentials are unavailable, falls back to reading a local
    Markdown/text file whose path is passed as ``gdrive_file_id`` (useful for
    offline / demo runs).
    """

    def __init__(self, persist_dir: Path | None = None) -> None:
        settings = get_settings()
        self.persist_dir = Path(persist_dir or settings.vector_store_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.credentials_path = settings.google_application_credentials
        self.folder_id = settings.gdrive_folder_id
        self.project_root = settings.project_root
        self.token_path = self.project_root / "credentials" / "token.json"
        self._vectorstore: Any = None
        self._docs: list[Any] = []

    # ------------------------------------------------------------------
    # Auth
    # ------------------------------------------------------------------

    def _build_credentials(self) -> Any:
        """Build Google API credentials from service account or OAuth client."""
        from google.auth.transport.requests import Request
        from google.oauth2 import service_account
        from google.oauth2.credentials import Credentials
        from google_auth_oauthlib.flow import InstalledAppFlow

        # 1) Reuse previously saved user token
        if self.token_path.exists():
            logger.info("Using cached OAuth token: %s", self.token_path)
            creds = Credentials.from_authorized_user_file(str(self.token_path), _DRIVE_SCOPES)
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
                self._persist_token(creds)
            if creds and creds.valid:
                return creds

        cred_path = Path(self.credentials_path) if self.credentials_path else None
        if not cred_path or not cred_path.exists():
            raise GoogleDriveRAGError(
                "No Google credentials found. Set GOOGLE_APPLICATION_CREDENTIALS "
                "to a service-account JSON or OAuth Desktop client JSON. "
                "For offline demos, pass a local file path as gdrive_file_id."
            )

        raw = json.loads(cred_path.read_text(encoding="utf-8"))

        # 2) Service account
        if raw.get("type") == "service_account" or "client_email" in raw:
            logger.info("Using service-account credentials: %s", cred_path)
            return service_account.Credentials.from_service_account_file(
                str(cred_path),
                scopes=_DRIVE_SCOPES,
            )

        # 3) Authorized-user token stored directly in GOOGLE_APPLICATION_CREDENTIALS
        if "refresh_token" in raw and "token" in raw:
            logger.info("Using authorized-user credentials file: %s", cred_path)
            return Credentials.from_authorized_user_file(str(cred_path), _DRIVE_SCOPES)

        # 4) OAuth Desktop / Web client → interactive consent (first run)
        if "installed" in raw or "web" in raw:
            logger.info(
                "OAuth client detected (%s). Starting browser consent flow…",
                "installed" if "installed" in raw else "web",
            )
            flow = InstalledAppFlow.from_client_secrets_file(str(cred_path), _DRIVE_SCOPES)
            creds = flow.run_local_server(port=0)
            self._persist_token(creds)
            return creds

        raise GoogleDriveRAGError(
            f"Unrecognized Google credentials format in {cred_path}. "
            "Expected service_account, authorized-user token, or OAuth installed/web client."
        )

    def _persist_token(self, creds: Any) -> None:
        """Save OAuth token for subsequent non-interactive runs."""
        self.token_path.parent.mkdir(parents=True, exist_ok=True)
        self.token_path.write_text(creds.to_json(), encoding="utf-8")
        logger.info("Saved OAuth token to %s", self.token_path)

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load_file(self, file_id: str) -> str:
        """Download a Drive file (or local path) and return its text content."""
        local_candidate = Path(file_id)
        if local_candidate.exists() and local_candidate.is_file():
            logger.info("Loading local bug report file: %s", local_candidate)
            return local_candidate.read_text(encoding="utf-8", errors="replace")

        try:
            return self._download_from_drive(file_id)
        except GoogleDriveRAGError:
            raise
        except Exception as exc:
            raise GoogleDriveRAGError(f"Failed to load Drive file {file_id}: {exc}") from exc

    def _download_from_drive(self, file_id: str) -> str:
        """Fetch file content via the Google Drive REST API."""
        from googleapiclient.discovery import build
        from googleapiclient.http import MediaIoBaseDownload
        import io

        creds = self._build_credentials()
        service = build("drive", "v3", credentials=creds, cache_discovery=False)

        meta = (
            service.files()
            .get(fileId=file_id, fields="id,name,mimeType", supportsAllDrives=True)
            .execute()
        )
        mime = meta.get("mimeType", "")
        name = meta.get("name", file_id)
        logger.info("Downloading Drive file '%s' (%s)", name, mime)

        # Google Docs → export as plain text
        if mime.startswith("application/vnd.google-apps"):
            export_mime = "text/plain"
            request = service.files().export_media(fileId=file_id, mimeType=export_mime)
        else:
            request = service.files().get_media(fileId=file_id, supportsAllDrives=True)

        buffer = io.BytesIO()
        downloader = MediaIoBaseDownload(buffer, request)
        done = False
        while not done:
            _, done = downloader.next_chunk()

        content = buffer.getvalue().decode("utf-8", errors="replace")
        logger.info("Downloaded %d bytes from Drive file %s", len(content), file_id)
        return content

    def load_via_langchain(self, folder_id: str | None = None) -> list[Any]:
        """Use LangChain ``GoogleDriveLoader`` when available."""
        try:
            from langchain_community.document_loaders import GoogleDriveLoader
        except ImportError as exc:
            raise GoogleDriveRAGError(
                "langchain-community GoogleDriveLoader unavailable"
            ) from exc

        fid = folder_id or self.folder_id
        if not fid:
            raise GoogleDriveRAGError("GDRIVE_FOLDER_ID is required for folder loading")

        loader_kwargs: dict[str, Any] = {"folder_id": fid}
        if self.credentials_path:
            loader_kwargs["service_account_key"] = Path(self.credentials_path)

        loader = GoogleDriveLoader(**loader_kwargs)
        docs = loader.load()
        self._docs = docs
        logger.info("Loaded %d documents via GoogleDriveLoader", len(docs))
        return docs

    # ------------------------------------------------------------------
    # Vector store / RAG
    # ------------------------------------------------------------------

    def build_index(self, texts: list[str], metadatas: list[dict] | None = None) -> Any:
        """Chunk texts and build a FAISS vector store (FakeEmbeddings offline)."""
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        docs_text = splitter.create_documents(texts, metadatas=metadatas)
        embeddings = self._get_embeddings()

        try:
            from langchain_community.vectorstores import FAISS

            self._vectorstore = FAISS.from_documents(docs_text, embeddings)
            index_path = self.persist_dir / "faiss_index"
            self._vectorstore.save_local(str(index_path))
            logger.info("Built FAISS index with %d chunks", len(docs_text))
            return self._vectorstore
        except Exception as faiss_exc:
            logger.warning("FAISS build failed (%s); retrying with FakeEmbeddings", faiss_exc)

        from langchain_community.embeddings import FakeEmbeddings
        from langchain_community.vectorstores import FAISS

        self._vectorstore = FAISS.from_documents(docs_text, FakeEmbeddings(size=384))
        logger.info("Built FAISS index with FakeEmbeddings (%d chunks)", len(docs_text))
        return self._vectorstore

    def _get_embeddings(self) -> Any:
        """Return Gemini embeddings (free), OpenAI embeddings, or FakeEmbeddings."""
        settings = get_settings()

        gemini_key = (
            settings.gemini_api_key.get_secret_value() if settings.gemini_api_key else None
        )
        if is_usable_secret(gemini_key, min_length=20):
            try:
                from langchain_google_genai import GoogleGenerativeAIEmbeddings

                logger.info("Using GoogleGenerativeAIEmbeddings for RAG")
                return GoogleGenerativeAIEmbeddings(
                    model="models/text-embedding-004",
                    google_api_key=gemini_key,
                )
            except Exception as exc:
                logger.warning("Gemini embeddings unavailable (%s)", exc)

        openai_key = (
            settings.openai_api_key.get_secret_value() if settings.openai_api_key else None
        )
        if is_usable_secret(openai_key):
            from langchain_openai import OpenAIEmbeddings

            return OpenAIEmbeddings(api_key=openai_key)

        logger.warning("No usable embedding key — using FakeEmbeddings for offline RAG")
        from langchain_community.embeddings import FakeEmbeddings

        return FakeEmbeddings(size=384)

    def as_retriever(self, k: int = 4) -> Any:
        """Return a LangChain retriever over the built index."""
        if self._vectorstore is None:
            raise GoogleDriveRAGError("Call build_index() before as_retriever()")
        return self._vectorstore.as_retriever(search_kwargs={"k": k})

    def retrieve(self, query: str, k: int = 4) -> list[str]:
        """Similarity-search the index and return page_content strings."""
        if self._vectorstore is None:
            return []
        docs = self._vectorstore.similarity_search(query, k=k)
        return [d.page_content for d in docs]

    def ingest_bug_report(self, file_id: str) -> tuple[str, list[str]]:
        """End-to-end: load a bug report, index it, return (raw_text, top chunks)."""
        raw = self.load_file(file_id)
        self.build_index([raw], metadatas=[{"source": file_id, "type": "bug_report"}])
        chunks = self.retrieve("bug error stacktrace failure root cause", k=6)
        return raw, chunks


def dump_credentials_hint() -> str:
    """Return a human-readable credentials setup hint."""
    return json.dumps(
        {
            "service_account": "Set GOOGLE_APPLICATION_CREDENTIALS to sa.json (type=service_account)",
            "oauth_desktop": "Use OAuth Desktop client JSON; first run opens browser, token saved to credentials/token.json",
            "offline": "Pass a local .md/.txt path as --gdrive-file-id",
        },
        indent=2,
    )
