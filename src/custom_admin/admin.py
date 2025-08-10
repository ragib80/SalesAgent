from django.contrib import admin
from custom_admin.models import SalesAuthUser
from django.contrib.auth.admin import UserAdmin

class SalesAuthUserAdmin(UserAdmin):
    model = SalesAuthUser
    list_display = ('email', 'username', 'first_name', 'last_name', 'is_active', 'is_staff', 'is_superuser', 'date_joined')
    list_filter = ('is_active', 'is_staff', 'is_superuser', 'date_joined')
    search_fields = ('email', 'username', 'first_name', 'last_name')
    ordering = ('-date_joined',)
    filter_horizontal = ()
    
    fieldsets = (
        (None, {'fields': ('email', 'password')}),
        ('Personal info', {'fields': ('username', 'first_name', 'last_name')}),
        ('Permissions', {'fields': ('is_active', 'is_staff', 'is_superuser', 'groups', 'user_permissions')}),
        ('Important dates', {'fields': ('date_joined', 'last_login')}),
    )
    
    add_fieldsets = (
        (None, {
            'classes': ('wide',),
            'fields': (
                'email', 'password1', 'password2', 
                'username', 'first_name', 'last_name', 
                'is_active', 'is_staff', 'is_superuser'
            ),
        }),
    )

admin.site.register(SalesAuthUser, SalesAuthUserAdmin)
