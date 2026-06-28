"""Seed Azure AI Search with SAP sales business/context knowledge."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from azure.core.credentials import AzureKeyCredential
from azure.core.exceptions import AzureError
from azure.search.documents import SearchClient
from django.conf import settings
from django.core.management.base import BaseCommand, CommandError
from openai import AzureOpenAI

from agent.rag.indexing import add_content_vectors, merge_or_upload_in_batches
from agent.rag.knowledge_seed import build_sales_knowledge_documents


class Command(BaseCommand):
    """Build and optionally upload SAP sales knowledge documents."""

    help = "Build and upload SAP sales business/context documents to Azure AI Search."

    def add_arguments(self, parser):
        parser.add_argument(
            "--upload",
            action="store_true",
            help="Upload generated documents to Azure AI Search. Without this flag the command is a dry run.",
        )
        parser.add_argument(
            "--index-name",
            default=None,
            help="Azure AI Search index name. Defaults to AZURE_SEARCH_INDEX_NAME from settings/env.",
        )
        parser.add_argument(
            "--output",
            default=None,
            help="Optional path to write generated documents as JSON before vector generation.",
        )
        parser.add_argument(
            "--sample-size",
            type=int,
            default=3,
            help="Number of sample documents to print during dry run.",
        )
        parser.add_argument(
            "--batch-size",
            type=int,
            default=100,
            help="Azure Search upload batch size.",
        )
        parser.add_argument(
            "--no-vectors",
            action="store_true",
            help="Upload text-only documents without generating contentVector.",
        )

    def handle(self, *args, **options):
        documents = build_sales_knowledge_documents()
        self.stdout.write(f"Built {len(documents)} SAP sales knowledge documents.")
        self._write_output_file(documents, options.get("output"))

        if not options["upload"]:
            self._write_dry_run_sample(documents, options["sample_size"])
            self.stdout.write(
                self.style.WARNING(
                    "Dry run only. Re-run with --upload to write to Azure AI Search."
                )
            )
            return

        index_name = options["index_name"] or _setting("AZURE_SEARCH_INDEX")
        endpoint = _setting("AZURE_SEARCH_ENDPOINT")
        search_key = _setting("AZURE_SEARCH_KEY")
        self._require_settings(
            {
                "AZURE_SEARCH_ENDPOINT": endpoint,
                "AZURE_SEARCH_KEY": search_key,
                "AZURE_SEARCH_INDEX_NAME": index_name,
            }
        )
        self.stdout.write(f"Target Azure AI Search index: {index_name}")

        upload_documents = documents
        if not options["no_vectors"]:
            upload_documents = add_content_vectors(
                documents,
                openai_client=self._build_openai_client(),
                embedding_deployment=self._embedding_deployment(),
            )

        search_client = SearchClient(
            endpoint=endpoint,
            index_name=index_name,
            credential=AzureKeyCredential(search_key),
        )
        try:
            results = merge_or_upload_in_batches(
                search_client,
                upload_documents,
                batch_size=options["batch_size"],
            )
        except AzureError as exc:
            raise CommandError(f"Azure AI Search upload failed: {exc}") from exc

        failed = [result for result in results if not getattr(result, "succeeded", False)]
        if failed:
            errors = [
                f"{getattr(result, 'key', '<unknown>')}: {getattr(result, 'error_message', '')}"
                for result in failed[:10]
            ]
            raise CommandError(
                "Azure Search rejected some documents: " + "; ".join(errors)
            )

        self.stdout.write(
            self.style.SUCCESS(
                f"Uploaded {len(upload_documents)} documents to Azure AI Search index {index_name}."
            )
        )

    def _build_openai_client(self) -> AzureOpenAI:
        endpoint = _setting("AZURE_OPENAI_ENDPOINT")
        api_key = _setting("AZURE_OPENAI_KEY")
        self._require_settings(
            {
                "AZURE_OPENAI_ENDPOINT": endpoint,
                "AZURE_OPENAI_KEY": api_key,
                "AZURE_OPENAI_EMBED_DEPLOYMENT": self._embedding_deployment(),
            }
        )
        api_version = (
            getattr(settings, "AZURE_OPENAI_API_VERSION", None)
            or os.getenv("AZURE_OPENAI_API_VERSION")
            or "2025-01-01-preview"
        )
        return AzureOpenAI(
            azure_endpoint=endpoint,
            api_key=api_key,
            api_version=api_version,
        )

    def _embedding_deployment(self) -> str:
        return _setting("AZURE_OPENAI_EMBED_DEPLOYMENT")

    def _write_output_file(self, documents: list[dict[str, Any]], output_path: str | None) -> None:
        if not output_path:
            return
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(documents, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        self.stdout.write(f"Wrote generated documents to {path}.")

    def _write_dry_run_sample(self, documents: list[dict[str, Any]], sample_size: int) -> None:
        if sample_size <= 0:
            return
        sample = documents[:sample_size]
        self.stdout.write("Sample documents:")
        self.stdout.write(json.dumps(sample, ensure_ascii=False, indent=2, default=str))

    def _require_settings(self, values: dict[str, str | None]) -> None:
        missing = [name for name, value in values.items() if not value]
        if missing:
            raise CommandError("Missing required settings/env values: " + ", ".join(missing))


def _setting(name: str) -> str:
    return str(getattr(settings, name, None) or os.getenv(name) or "").strip()
