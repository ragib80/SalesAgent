import os
import requests
from dotenv import load_dotenv
import pyodbc
import tempfile
import csv
from datetime import datetime
import decimal

from django.core.management.base import BaseCommand
from django.conf import settings
from functools import lru_cache

# ADX SDK imports
from azure.kusto.data import (
    KustoClient,
    KustoConnectionStringBuilder,
    DataFormat,
)
from azure.kusto.ingest import (
    QueuedIngestClient,
    IngestionProperties,
    FileDescriptor,
)

from sales_analyzer.models import DataIngestionTracker  

# Load environment variables from .env
load_dotenv()

class ADXTool:
    def __init__(self, cluster: str, database: str):
        if not cluster or not database:
            raise ValueError("Azure ADX cluster and database must be provided.")
        kcsb = KustoConnectionStringBuilder.with_az_cli_authentication(cluster)
        self.client = KustoClient(kcsb)
        self.database = database

    def run(self, kql: str):
        resp = self.client.execute(self.database, kql)
        tbl  = resp.primary_results[0]
        cols = [c.column_name for c in tbl.columns]
        rows = [list(r) for r in tbl]
        return cols, rows

@lru_cache(maxsize=1)
def adx() -> ADXTool:
    cluster  = getattr(settings, "ADX_CLUSTER_DEV", os.getenv("ADX_CLUSTER_DEV"))
    database = getattr(settings, "ADX_DATABASE_DEV", os.getenv("ADX_DATABASE_DEV"))
    if not cluster or not database:
        raise ValueError("ADX_CLUSTER_DEV and ADX_DATABASE_DEV must be set in settings or env")
    return ADXTool(cluster, database)

class Command(BaseCommand):
    help = 'Imports sales data from MS SQL to ADX (via Blob Storage staging)'

    def add_arguments(self, parser):
        parser.add_argument(
            '--start_date',
            type=str,
            help="ISO-format start date for incremental load (defaults to last watermark)"
        )
        parser.add_argument(
            '--end_date',
            type=str,
            help="ISO-format end date for incremental load (defaults to now)"
        )

    def handle(self, *args, **options):
        # ─── ADX SETUP ──────────────────────────────────────────────────────────
        adx_cluster = os.getenv("ADX_CLUSTER_DEV", settings.ADX_CLUSTER_DEV)
        adx_db      = os.getenv("ADX_DATABASE_DEV", settings.ADX_DATABASE_DEV)
        if not adx_cluster or not adx_db:
            raise ValueError("Missing ADX_CLUSTER_DEV or ADX_DATABASE_DEV")
        adx_client = adx()

        # ─── MSSQL SETUP ────────────────────────────────────────────────────────
        server = os.getenv("DB_HOST", settings.DB_HOST)
        dbname = os.getenv("DB_NAME", settings.DB_NAME)
        user = os.getenv("DB_USER", settings.DB_USER)  # This might be empty
        pwd = os.getenv("DB_PASSWORD", settings.DB_PASSWORD)  # This might be empty
        driver = os.getenv("DB_DRIVER", settings.DB_DRIVER)
        extra = os.getenv("DB_EXTRA_PARAMS", settings.DB_EXTRA_PARAMS)

        # Validate host and database name
        if not server or not dbname:
            raise ValueError("Both DB_HOST and DB_NAME must be set")

        # Start building connection string
        parts = [
            f"DRIVER={{{driver}}}",
            f"SERVER={server}",
            f"DATABASE={dbname}",
        ]

        # If user and password are empty, use Windows Authentication
        if user and pwd:
            parts.append(f"UID={user}")
            parts.append(f"PWD={pwd}")
        else:
            # Add Windows authentication option
            parts.append(f"Trusted_Connection=yes")  # Use Windows Authentication

        # Append extra params (Trusted_Connection should be there already, but you can override)
        if extra:
            parts.append(extra)

        # Final connection string
        start_date = '2025-06-01'
        end_date = '2025-06-15'
        conn_str = ";".join(parts)
        self.stdout.write(f"Connecting with: {conn_str}")

        # Establish the connection
        conn = pyodbc.connect(conn_str)
        cursor = conn.cursor()
        query = """
            SELECT * 
              FROM SAPSalesInfos 
             WHERE FKDAT_TEMP BETWEEN ? AND ?
        """
        cursor.execute(query, (start_date, end_date))
        rows = cursor.fetchall()

        if not rows:
            self.stdout.write(self.style.SUCCESS("No new data to ingest."))
            conn.close()
            return

        # ─── UPDATE WATERMARK ───────────────────────────────────────────────────
        new_watermark = max(r.FKDAT_TEMP for r in rows)
        DataIngestionTracker.objects.create(last_ingested_timestamp=new_watermark)

        # ─── DUMP TO CSV & STAGE IN BLOB ────────────────────────────────────────
        with tempfile.NamedTemporaryFile(delete=False, mode="w", newline="", suffix=".csv") as tmp:
            writer = csv.writer(tmp)
            writer.writerow([col[0] for col in cursor.description])
            for row in rows:
                rec = list(row)
                for i, v in enumerate(rec):
                    if isinstance(v, datetime):
                        rec[i] = v.isoformat()
                    elif isinstance(v, decimal.Decimal):
                        rec[i] = float(v)
                writer.writerow(rec)
            tmp_path = tmp.name
        conn.close()

        # ─── DOWNLOAD FILE FROM BLOB STORAGE USING SAS URL ────────────────────
        # Dynamically create the filename based on the current date or watermark
        file_name = f"sales_data_{new_watermark}.csv"
        
        # Construct SAS URL with the dynamic filename
        blob_sas_url = f"https://bpblaistorageaccount.blob.core.windows.net/aicontainer/{file_name}?se=2025-07-23T10%3A00%3A00Z&sp=r&sv=2022-11-02&sr=b&sig=pTGnPScZbt0RCexEZoRxR9LwGLcxBgUfhojRvmj7bpA%3D"
        
        # Download the file to a temporary local path
        response = requests.get(blob_sas_url)
        if response.status_code == 200:
            with open(tmp_path, "wb") as f:
                f.write(response.content)
            self.stdout.write(self.style.SUCCESS(f"Downloaded file to {tmp_path}"))
        else:
            self.stdout.write(self.style.ERROR(f"Failed to download the file: {response.status_code}"))
            return

        # ─── INGEST THE FILE FROM LOCAL PATH INTO ADX ─────────────────────────
        ingest_client = QueuedIngestClient(
            KustoConnectionStringBuilder.with_az_cli_authentication(adx_cluster)
        )
        ingestion_props = IngestionProperties(
            database=adx_db,
            table="YSales",
            data_format=DataFormat.CSV  # Using DataFormat from azure.kusto.data
        )
        file_desc = FileDescriptor(tmp_path, 0)
        ingest_client.ingest_from_file(file_desc, ingestion_properties=ingestion_props)

        self.stdout.write(self.style.SUCCESS("✓ Data successfully staged and ingested into ADX."))
