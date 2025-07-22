from django.db import models

# Create your models here.
class DataIngestionTracker(models.Model):
    last_ingested_timestamp = models.DateTimeField()
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Ingested up to {self.last_ingested_timestamp}"