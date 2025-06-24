from django.db import models

# Create your models here.


class TokenBlacklist(models.Model):
    refresh_token = models.TextField(unique=True)
    blacklisted_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Blacklisted Token {self.refresh_token[:20]}..."
