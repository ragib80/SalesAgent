from django.contrib import admin

from django.contrib.auth import get_user_model
from django.contrib.auth.admin import UserAdmin as BaseUserAdmin
from django.db.models import Count

from user_auth.models import Depo, Zone, Territory, UserDepoMap, UserZoneMap, UserTerritoryMap

# ---- Display text in autocomplete dropdowns (good UX) ----
# (Add __str__ in your models if not already there)
# class Depo(models.Model): ...
#     def __str__(self): return f"{self.code} — {self.name}"
# Same for Zone/Territory.

# ---- Register area masters so autocomplete_fields works ----
# @admin.register(Depo)
# class DepoAdmin(admin.ModelAdmin):
#     search_fields = ['id', 'code', 'name']
#     list_display = ['id', 'code', 'name']
#     ordering = ['code']
#     list_per_page = 20

# @admin.register(Zone)
# class ZoneAdmin(admin.ModelAdmin):
#     search_fields = ['id', 'code', 'name']
#     list_display = ['id', 'code', 'name']
#     ordering = ['code']
#     list_per_page = 20

# @admin.register(Territory)
# class TerritoryAdmin(admin.ModelAdmin):
#     search_fields = ['id', 'code', 'name']
#     list_display = ['id', 'code', 'name']
#     ordering = ['code']
#     list_per_page = 20

# class UserDepoInline(admin.TabularInline):
#     model = UserDepoMap
#     fk_name = 'user'                    # be explicit
#     extra = 1                           # <-- show 1 blank row
#     autocomplete_fields = ['depo']
#     show_change_link = True
#     classes = []                        # <-- not collapsed


# class UserZoneInline(admin.TabularInline):
#     model = UserZoneMap
#     fk_name = 'user'
#     extra = 1
#     autocomplete_fields = ['zone']
#     show_change_link = True
#     classes = []



# class UserTerritoryInline(admin.TabularInline):
#     model = UserTerritoryMap
#     fk_name = 'user'
#     extra = 1
#     autocomplete_fields = ['territory']
#     show_change_link = True
#     classes = []
# # ---- User admin with inlines + counts ----
# User = get_user_model()

# # If User is already registered, replace it with our enhanced admin.
# try:
#     admin.site.unregister(User)
# except admin.sites.NotRegistered:
#     pass

# @admin.register(User)
# class SalesAuthUserAdmin(BaseUserAdmin):
#     inlines = [UserDepoInline, UserZoneInline, UserTerritoryInline]
#     save_on_top = True

#     # optional: keep your counts here (use the FK field to avoid id-PK assumption)
#     from django.db.models import Count
#     def get_queryset(self, request):
#         qs = super().get_queryset(request)
#         return qs.annotate(
#             _depo_count=Count('depo_links__depo', distinct=True),
#             _zone_count=Count('zone_links__zone', distinct=True),
#             _territory_count=Count('territory_links__territory', distinct=True),
#         )
#     def depo_count(self, obj): return getattr(obj, '_depo_count', 0)
#     depo_count.short_description = 'Depo'
#     def zone_count(self, obj): return getattr(obj, '_zone_count', 0)
#     zone_count.short_description = 'Zone'
#     def territory_count(self, obj): return getattr(obj, '_territory_count', 0)
#     territory_count.short_description = 'Territory'