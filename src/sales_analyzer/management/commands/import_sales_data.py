import os
import requests
import logging
from dotenv import load_dotenv
import pyodbc
import tempfile
import csv
import time
from datetime import datetime, timedelta
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
    ReportLevel,
)
from azure.kusto.ingest.status import KustoIngestStatusQueues

from sales_analyzer.models import DataIngestionTracker  

# Azure Blob Storage imports
from azure.storage.blob import BlobServiceClient

# Load environment variables from .env
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO, filename="ingestion_log.log", format="%(asctime)s - %(message)s")
logger = logging.getLogger()

def get_setting_or_env(name):
    return getattr(settings, name, None) or os.getenv(name)

def get_adx_table_name():
    return get_setting_or_env("ADX_DATABASE_TABLE") or get_setting_or_env("ADX_TABLE") or "SAPSalesInfos"

def get_adx_mapping_name():
    return get_setting_or_env("ADX_DATABASE_TABLE_MAPPING") or "SAPSalesInfos_mapping"

def build_adx_connection(cluster):
    client_id = get_setting_or_env("AZURE_CLIENT_ID")
    client_secret = get_setting_or_env("AZURE_CLIENT_SECRET")
    tenant_id = get_setting_or_env("AZURE_TENANT_ID")
    missing = [
        name for name, value in {
            "AZURE_CLIENT_ID": client_id,
            "AZURE_CLIENT_SECRET": client_secret,
            "AZURE_TENANT_ID": tenant_id,
        }.items()
        if not value
    ]
    if missing:
        raise ValueError(f"Missing Azure ADX service principal config: {', '.join(missing)}")

    return KustoConnectionStringBuilder.with_aad_application_key_authentication(
        cluster,
        aad_app_id=client_id,
        app_key=client_secret,
        authority_id=tenant_id,
    )

def wait_for_ingestion_result(ingest_client, source_id, timeout_seconds=120, poll_interval=5):
    status_queues = KustoIngestStatusQueues(ingest_client)
    source_id = str(source_id).lower()
    deadline = time.time() + timeout_seconds

    while time.time() < deadline:
        for message in status_queues.failure.peek(32):
            if str(getattr(message, "IngestionSourceId", "")).lower() == source_id:
                return False, str(message)

        for message in status_queues.success.peek(32):
            if str(getattr(message, "IngestionSourceId", "")).lower() == source_id:
                return True, str(message)

        time.sleep(poll_interval)

    return None, f"No ADX ingestion status received within {timeout_seconds} seconds for source_id={source_id}"

class ADXTool:
    def __init__(self, cluster: str, database: str):
        if not cluster or not database:
            raise ValueError("Azure ADX cluster and database must be provided.")
        kcsb = build_adx_connection(cluster)
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
    cluster  = getattr(settings, "ADX_CLUSTER", os.getenv("ADX_CLUSTER"))
    database = getattr(settings, "ADX_DATABASE", os.getenv("ADX_DATABASE"))
    if not cluster or not database:
        raise ValueError("ADX_CLUSTER and ADX_DATABASE must be set in settings or env")
    return ADXTool(cluster, database)

class Command(BaseCommand):
    help = 'Imports sales data from MS SQL to ADX (via Blob Storage staging) - Fetches yesterday only'

    def add_arguments(self, parser):
        parser.add_argument(
            '--start_date',
            type=str,
            default=None,
            help="ISO-format start date for incremental load (defaults to yesterday)"
        )
        parser.add_argument(
            '--end_date',
            type=str,
            default=None,
            help="ISO-format end date for incremental load (defaults to yesterday)"
        )

    def get_dates(self, start_date_arg, end_date_arg):
        """
        Get start and end dates intelligently.
        
        Logic:
        - If both dates provided: Use them
        - If no dates provided: Use yesterday only (skip today's incomplete data)
        - If only one date provided: Raise error (require both or neither)
        
        Returns:
            Tuple of (start_date, end_date) as ISO format strings
        """
        if start_date_arg and end_date_arg:
            # User provided both dates
            start_date = start_date_arg
            end_date = end_date_arg
            self.stdout.write(f"📅 Using provided dates: {start_date} to {end_date}")
        
        elif not start_date_arg and not end_date_arg:
            # No dates provided: default to yesterday only
            today = datetime.now().date()
            yesterday = today - timedelta(days=1)
            start_date = yesterday.isoformat()
            end_date = yesterday.isoformat()  # ✅ Same as start_date - yesterday only!
            self.stdout.write(self.style.SUCCESS(f"📅 No dates provided. Using default: {start_date} (Yesterday only)"))
        
        else:
            # Only one date provided: ERROR
            raise ValueError("❌ Both --start_date and --end_date must be provided together, or neither")
        
        return start_date, end_date

    def check_data_exists_in_tracker(self, watermark):
        """
        Check if data with the given watermark already exists in local DataIngestionTracker.
        
        This is the first layer of duplicate detection (local database).
        
        Args:
            watermark: The watermark/date to check
        
        Returns:
            True if data already tracked, False otherwise
        """
        try:
            last_ingested_timestamp = DataIngestionTracker.objects.latest('created_at').last_ingested_timestamp
        except DataIngestionTracker.DoesNotExist:
            return False
        
        # ─── NORMALIZE DATES FOR COMPARISON ─────────────────────────────────
        # Convert watermark to datetime if it's a string
        if isinstance(watermark, str):
            watermark = datetime.fromisoformat(watermark)
        
        # Convert last_ingested_timestamp to datetime if it's a string
        if isinstance(last_ingested_timestamp, str):
            last_ingested_timestamp = datetime.fromisoformat(last_ingested_timestamp)
        
        # Strip timezone info for comparison (compare naive datetimes)
        if watermark.tzinfo is not None:
            watermark = watermark.replace(tzinfo=None)
        
        if last_ingested_timestamp.tzinfo is not None:
            last_ingested_timestamp = last_ingested_timestamp.replace(tzinfo=None)
        
        # Compare dates only (ignore time component)
        watermark_date = watermark.date() if isinstance(watermark, datetime) else watermark
        last_ingested_date = last_ingested_timestamp.date() if isinstance(last_ingested_timestamp, datetime) else last_ingested_timestamp
        
        # If we've already ingested this date or later, skip
        if last_ingested_date >= watermark_date:
            logger.warning(f"Data for watermark {watermark_date} already tracked as ingested.")
            return True
        
        return False

    def check_data_exists_in_adx(self, adx_client, watermark, table_name=None):
        """
        Check if data with the given watermark already exists in ADX table.
        
        This is the second layer of duplicate detection (cloud database).
        Queries the actual ADX table to ensure data isn't already there.
        
        Args:
            adx_client: ADXTool instance
            watermark: The watermark/date to check (datetime or string)
            table_name: The table name in ADX
        
        Returns:
            True if data exists in ADX, False otherwise
        """
        try:
            table_name = table_name or get_adx_table_name()

            # Convert watermark to ISO date string
            if isinstance(watermark, datetime):
                watermark_str = watermark.date().isoformat()
            else:
                watermark_str = watermark
            
            # Build KQL query to count records for this date
            kql = f"""
            {table_name}
            | where fkdat == datetime({watermark_str})
            | count
            """
            
            self.stdout.write(f"🔍 Checking if data exists in ADX for watermark: {watermark_str}")
            cols, rows = adx_client.run(kql)
            
            # Extract count from result (rows[0][0] is the count)
            if rows and rows[0][0] > 0:
                count = rows[0][0]
                logger.warning(f"Data for watermark {watermark_str} already exists in ADX. Count: {count}")
                self.stdout.write(self.style.WARNING(f"⚠️  Found {count} records in ADX for {watermark_str}"))
                return True
            else:
                self.stdout.write(self.style.SUCCESS(f"✓ No existing data found for watermark: {watermark_str}"))
                return False
                
        except Exception as e:
            logger.error(f"Error checking ADX for existing data: {str(e)}")
            self.stdout.write(self.style.ERROR(f"⚠️  Error checking ADX: {str(e)}"))
            # Return False to allow ingestion to proceed (fail-open approach)
            # Better to ingest than to fail completely
            return False

    def handle(self, *args, **options):
        """
        Main handler for the ingestion command.
        
        Flow:
        1. Get dates (auto or provided)
        2. Connect to MS SQL Server
        3. Query sales data
        4. Determine watermark (max date)
        5. Check Tracker (local DB)
        6. Check ADX (cloud DB)
        7. Check Blob Storage
        8. Convert to CSV
        9. Upload to Blob
        10. Ingest to ADX
        11. Update Tracker
        """
        
        # ─── STEP 1: GET START AND END DATES ────────────────────────────────
        start_date_arg = options.get('start_date')
        end_date_arg = options.get('end_date')
        
        start_date, end_date = self.get_dates(start_date_arg, end_date_arg)
        self.stdout.write(f"📆 Ingesting data from {start_date} to {end_date}")

        # ─── STEP 2: ADX SETUP ──────────────────────────────────────────────
        adx_cluster = os.getenv("ADX_CLUSTER", settings.ADX_CLUSTER)
        adx_db      = os.getenv("ADX_DATABASE", settings.ADX_DATABASE)
        adx_table   = get_adx_table_name()
        adx_mapping = get_adx_mapping_name()
        if not adx_cluster or not adx_db:
            raise ValueError("Missing ADX_CLUSTER or ADX_DATABASE")
        self.stdout.write(f"ADX target: database={adx_db}, table={adx_table}, mapping={adx_mapping}")
        adx_client = adx()

        # ─── STEP 3: MSSQL SETUP ───────────────────────────────────────────
        server = os.getenv("DB_HOST", settings.DB_HOST)
        dbname = os.getenv("DB_NAME", settings.DB_NAME)
        user = os.getenv("DB_USER", settings.DB_USER)
        pwd = os.getenv("DB_PASSWORD", settings.DB_PASSWORD)
        driver = os.getenv("DB_DRIVER", settings.DB_DRIVER)
        extra = os.getenv("DB_EXTRA_PARAMS", settings.DB_EXTRA_PARAMS)

        # Validate host and database name
        if not server or not dbname:
            raise ValueError("Both DB_HOST and DB_NAME must be set")

        # Build connection string
        parts = [
            f"DRIVER={{{driver}}}",
            f"SERVER={server}",
            f"DATABASE={dbname}",
        ]

        # Add authentication (SQL or Windows)
        if user and pwd:
            parts.append(f"UID={user}")
            parts.append(f"PWD={pwd}")
        else:
            parts.append(f"Trusted_Connection=yes")

        # Append extra parameters
        if extra:
            parts.append(extra)

        conn_str = ";".join(parts)
        self.stdout.write(f"🔗 Connecting to MS SQL Server...")

        # ─── STEP 4: QUERY SQL SERVER ──────────────────────────────────────
        conn = pyodbc.connect(conn_str)
        cursor = conn.cursor()
        
        query = """
            SELECT * 
              FROM SAPSalesInfos 
             WHERE fkdat BETWEEN ? AND ?
        """
        cursor.execute(query, (start_date, end_date))
        rows = cursor.fetchall()

        # Check if we got any data
        if not rows:
            self.stdout.write(self.style.SUCCESS("✓ No new data to ingest."))
            conn.close()
            return

        self.stdout.write(f"📊 Retrieved {len(rows)} rows from MS SQL Server")

        # ─── STEP 5: DETERMINE WATERMARK ───────────────────────────────────
        new_watermark = max(r.fkdat for r in rows)
        self.stdout.write(f"💧 New watermark: {new_watermark}")

        # ─── STEP 6: CHECK TRACKER DATABASE (Layer 1) ──────────────────────
        self.stdout.write("🔍 Checking DataIngestionTracker...")
        tracker_has_data = self.check_data_exists_in_tracker(new_watermark)
        if tracker_has_data:
            logger.warning(f"Data for watermark {new_watermark} already exists in tracker. Skipping ingestion.")
            self.stdout.write(self.style.WARNING(f"Data for watermark {new_watermark} already tracked. Verifying ADX before skipping."))

        # ─── STEP 7: CHECK ADX DATABASE (Layer 2) ──────────────────────────
        self.stdout.write("🔍 Checking Azure Data Explorer for existing data...")
        if self.check_data_exists_in_adx(adx_client, new_watermark, table_name=adx_table):
            logger.warning(f"Data for watermark {new_watermark} already exists in ADX. Skipping ingestion.")
            self.stdout.write(self.style.WARNING(f"Data for watermark {new_watermark} already in ADX. Skipping."))
            conn.close()
            return
        if tracker_has_data:
            logger.warning(f"Tracker has watermark {new_watermark}, but ADX table {adx_table} has no matching rows. Retrying ingestion.")
            self.stdout.write(self.style.WARNING(f"Tracker has {new_watermark}, but ADX has no matching rows. Retrying ingestion."))

        # ─── STEP 8: CONVERT TO CSV ────────────────────────────────────────
        self.stdout.write("📝 Converting data to CSV format...")
        with tempfile.NamedTemporaryFile(delete=False, mode="w", newline="", suffix=".csv") as tmp:
            writer = csv.writer(tmp)
            # Write header row
            writer.writerow([col[0] for col in cursor.description])
            # Write data rows with type conversion
            for row in rows:
                rec = list(row)
                for i, v in enumerate(rec):
                    # Convert datetime to ISO format string
                    if isinstance(v, datetime):
                        rec[i] = v.isoformat()
                    # Convert Decimal to float
                    elif isinstance(v, decimal.Decimal):
                        rec[i] = float(v)
                writer.writerow(rec)
            tmp_path = tmp.name
        
        self.stdout.write(f"✓ CSV file created: {tmp_path}")
        conn.close()

        # ─── STEP 9: CHECK BLOB STORAGE (Layer 3) ──────────────────────────
        file_name = f"sales_data_{new_watermark}.csv"
        self.stdout.write(f"☁️  Checking Azure Blob Storage...")
        
        try:
            blob_svc = BlobServiceClient.from_connection_string(os.getenv("AZURE_STORAGE_CONNECTION_STRING"))
            container_client = blob_svc.get_container_client("aicontainer")
            blob_client = container_client.get_blob_client(file_name)
            blob_exists = blob_client.exists()
            
            # Skip if file already exists
            if blob_exists:
                logger.warning(f"File {file_name} already exists in Blob Storage. Skipping upload.")
                self.stdout.write(self.style.WARNING(f"File {file_name} already in Blob Storage. Re-uploading and retrying ADX ingestion."))
        except Exception as e:
            logger.error(f"Error checking Blob Storage: {str(e)}")
            self.stdout.write(self.style.ERROR(f"Error checking Blob Storage: {str(e)}"))
            return

        # ─── STEP 10: UPLOAD TO BLOB STORAGE ───────────────────────────────
        self.stdout.write(f"⬆️  Uploading {file_name} to Blob Storage...")
        try:
            with open(tmp_path, "rb") as data:
                blob_client.upload_blob(data, overwrite=True)
                logger.info(f"File uploaded successfully to Blob Storage.")
                self.stdout.write(self.style.SUCCESS(f"✓ File uploaded to Blob Storage"))
        except Exception as e:
            logger.error(f"Error uploading to Blob Storage: {str(e)}")
            self.stdout.write(self.style.ERROR(f"Error uploading to Blob Storage: {str(e)}"))
            return
        
        # ─── STEP 11: INGEST INTO ADX ──────────────────────────────────────
        self.stdout.write("⬆️  Ingesting data into Azure Data Explorer...")
        try:
            ingest_client = QueuedIngestClient(
                build_adx_connection(adx_cluster)
            )
            ingestion_props = IngestionProperties(
                database=adx_db,
                table=adx_table,
                data_format=DataFormat.CSV,
                ingestion_mapping_reference=adx_mapping,
                ingestion_mapping_kind=DataFormat.CSV.ingestion_mapping_kind,
                ignore_first_record=True,
                report_level=ReportLevel.FailuresAndSuccesses,
            )
            file_desc = FileDescriptor(tmp_path)
            ingest_result = ingest_client.ingest_from_file(file_desc, ingestion_properties=ingestion_props)
            logger.info(f"ADX ingestion queued: {ingest_result}")
            self.stdout.write(f"Queued ADX ingestion. Source ID: {ingest_result.source_id}")

            status, status_message = wait_for_ingestion_result(ingest_client, ingest_result.source_id)
            if status is False:
                logger.error(f"ADX ingestion failed: {status_message}")
                self.stdout.write(self.style.ERROR(f"ADX ingestion failed: {status_message}"))
                return
            if status is None:
                logger.warning(status_message)
                self.stdout.write(self.style.WARNING(status_message))
                self.stdout.write(self.style.WARNING("ADX may still finish later. Watermark was not updated because insertion was not confirmed."))
                return

            # ─── STEP 12: UPDATE WATERMARK ─────────────────────────────────
            DataIngestionTracker.objects.create(last_ingested_timestamp=new_watermark)
            
            logger.info(f"Data successfully ingested. Watermark: {new_watermark}")
            self.stdout.write(self.style.SUCCESS("✅ Data successfully ingested into ADX!"))
            self.stdout.write(self.style.SUCCESS(f"✓ Watermark updated: {new_watermark}"))
            
        except Exception as e:
            logger.error(f"Failed to ingest data: {str(e)}")
            self.stdout.write(self.style.ERROR(f"❌ Failed to ingest data: {str(e)}"))
