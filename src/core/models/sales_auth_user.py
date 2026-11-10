# # In your SalesAuthUser model

# from django.db import models
# import uuid


# class SalesAuthUser(models.Model):
#     uuid = models.UUIDField(
#         default=uuid.uuid4, editable=False, unique=True, db_index=True)
#     # other fields here...

#     def __str__(self):
#         return f"SalesAuthUser ({self.uuid})"

#     class Meta:
#         db_table = 'sales_auth_user'
#         verbose_name = "SalesAuthUser"
#         verbose_name_plural = "SalesAuthUsers"
#         ordering = ['-created_at']
