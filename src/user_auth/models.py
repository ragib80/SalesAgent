import hashlib
import secrets
from datetime import timedelta

from django.conf import settings
from django.db import models
from django.utils import timezone


class TokenBlacklist(models.Model):
    refresh_token = models.CharField(max_length=450, unique=True)  # <= key change
    blacklisted_at = models.DateTimeField(auto_now_add=True)


class OTPToken(models.Model):
    """
    Stores a one-time password tied to a login session.

    Flow:
      1. LoginInitiateView creates one of these after LDAP verifies credentials.
      2. The plain OTP code is emailed to the user.
      3. OTPVerifyView looks up the record by session_token, calls .verify(), then issues JWT.
    """

    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="otp_tokens",
    )
    # Opaque token returned to the client so it can identify the session at /otp/verify/
    session_token = models.CharField(max_length=64, unique=True, db_index=True)
    # SHA-256 of the plain OTP code (never stored in clear text)
    otp_hash = models.CharField(max_length=64)
    created_at = models.DateTimeField(auto_now_add=True)
    expires_at = models.DateTimeField()
    is_used = models.BooleanField(default=False)

    class Meta:
        db_table = "otp_tokens"

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def create_for_user(cls, user, otp_length: int = 6, expiry_minutes: int = 5):
        """
        Invalidate any pending OTPs, generate a new one, persist, and return
        (OTPToken instance, plain_otp_code).  The caller is responsible for
        emailing the plain code.
        """
        # Invalidate previous pending OTPs for this user
        cls.objects.filter(user=user, is_used=False).update(is_used=True)

        plain_otp = "".join(str(secrets.randbelow(10)) for _ in range(otp_length))
        session_token = secrets.token_urlsafe(32)
        otp_hash = hashlib.sha256(plain_otp.encode()).hexdigest()
        expires_at = timezone.now() + timedelta(minutes=expiry_minutes)

        obj = cls.objects.create(
            user=user,
            session_token=session_token,
            otp_hash=otp_hash,
            expires_at=expires_at,
        )
        return obj, plain_otp

    # ------------------------------------------------------------------
    # Verification
    # ------------------------------------------------------------------

    def verify(self, plain_otp: str) -> bool:
        """Return True only if the code is correct, unused, and not expired."""
        if self.is_used or timezone.now() > self.expires_at:
            return False
        return hashlib.sha256(plain_otp.encode()).hexdigest() == self.otp_hash


# models.py



# Existing tables (read-only from Django migrations point of view)
class Depo(models.Model):
    id = models.BigIntegerField(db_column='id', primary_key=True)
    code = models.CharField(db_column='Code', max_length=255)
    name = models.CharField(db_column='Name', max_length=255)

    def __str__(self):
        if self.code and self.name:
            return f"{self.code} — {self.name}"
        return self.name or self.code or str(self.id)

    class Meta:
        db_table = 'Depo'
        managed = False


class Zone(models.Model):
    id = models.BigIntegerField(db_column='id', primary_key=True)
    code = models.CharField(db_column='Code', max_length=255)
    name = models.CharField(db_column='Name', max_length=255)

    def __str__(self):
        if self.code and self.name:
            return f"{self.code} — {self.name}"
        return self.name or self.code or str(self.id)

    class Meta:
        db_table = 'Zone'
        managed = False


class Territory(models.Model):
    id = models.BigIntegerField(db_column='id', primary_key=True)
    code = models.CharField(db_column='Code', max_length=255)
    name = models.CharField(db_column='Name', max_length=255)

    def __str__(self):
        if self.code and self.name:
            return f"{self.code} — {self.name}"
        return self.name or self.code or str(self.id)

    class Meta:
        db_table = 'Territory'
        managed = False


# Mapping tables (Django will create/manage these)
class UserDepoMap(models.Model):
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        db_column='user_id',
        on_delete=models.CASCADE,
        related_name='depo_links',
    )
    depo = models.ForeignKey(
        Depo,
        db_column='depo_id',
        on_delete=models.CASCADE,
        related_name='user_links',
    )
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = 'UserDepoMap'
        constraints = [
            models.UniqueConstraint(fields=['user', 'depo'], name='uq_user_depo')
        ]
        indexes = [
            models.Index(fields=['user'], name='ix_udm_user'),
            models.Index(fields=['depo'], name='ix_udm_depo'),
        ]


class UserZoneMap(models.Model):
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        db_column='user_id',
        on_delete=models.CASCADE,
        related_name='zone_links',
    )
    zone = models.ForeignKey(
        Zone,
        db_column='zone_id',
        on_delete=models.CASCADE,
        related_name='user_links',
    )
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = 'UserZoneMap'
        constraints = [
            models.UniqueConstraint(fields=['user', 'zone'], name='uq_user_zone')
        ]
        indexes = [
            models.Index(fields=['user'], name='ix_uzm_user'),
            models.Index(fields=['zone'], name='ix_uzm_zone'),
        ]


class UserTerritoryMap(models.Model):
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        db_column='user_id',
        on_delete=models.CASCADE,
        related_name='territory_links',
    )
    territory = models.ForeignKey(
        Territory,
        db_column='territory_id',
        on_delete=models.CASCADE,
        related_name='user_links',
    )
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = 'UserTerritoryMap'
        constraints = [
            models.UniqueConstraint(fields=['user', 'territory'], name='uq_user_territory')
        ]
        indexes = [
            models.Index(fields=['user'], name='ix_utm_user'),
            models.Index(fields=['territory'], name='ix_utm_territory'),
        ]
