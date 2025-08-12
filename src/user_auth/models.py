from django.db import models
from django.conf import settings

class TokenBlacklist(models.Model):
    refresh_token = models.CharField(max_length=450, unique=True)  # <= key change
    blacklisted_at = models.DateTimeField(auto_now_add=True)


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
