# Script to run Django import_sales_data command
# Location: C:\inetpub\wwwroot\VoiceOfSales\run_import_sales.ps1

# Set the directories
$VenvDir = "C:\inetpub\wwwroot\VoiceOfSales"
$SrcDir = "C:\inetpub\wwwroot\VoiceOfSales\src"
$LogsDir = "C:\inetpub\wwwroot\VoiceOfSales\src\logs"

# Change to src directory (where manage.py is located)
cd $SrcDir

# Activate virtual environment
& "$VenvDir\venv\Scripts\Activate.ps1"

# Run the Django management command
python manage.py import_sales_data

# Log the result
$timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
if ($LASTEXITCODE -eq 0) {
    Add-Content -Path "$LogsDir\import_sales.log" -Value "$timestamp - Success: import_sales_data completed"
} else {
    Add-Content -Path "$LogsDir\import_sales.log" -Value "$timestamp - Error: import_sales_data failed with exit code $LASTEXITCODE"
}

exit $LASTEXITCODE