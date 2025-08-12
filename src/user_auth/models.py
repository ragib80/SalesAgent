from django.db import models

class TokenBlacklist(models.Model):
    refresh_token = models.CharField(max_length=450, unique=True)  # <= key change
    blacklisted_at = models.DateTimeField(auto_now_add=True)
