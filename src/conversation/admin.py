from django.contrib import admin


from .models import Conversation, Message

# Read-only inline for messages
class MessageInline(admin.TabularInline):
    model = Message
    fields = ('uuid', 'sender', 'text', 'created_at')
    readonly_fields = ('uuid', 'sender', 'text', 'created_at')
    extra = 0
    can_delete = False
    show_change_link = True
    list_per_page = 20  # <-- Pagination for inline


    # Only show non-deleted messages
    def get_queryset(self, request):
        qs = super().get_queryset(request)
        return qs.filter(is_deleted=False)

# Read-only admin for Conversation
@admin.register(Conversation)
class ConversationAdmin(admin.ModelAdmin):
    list_display = ('uuid', 'user', 'title', 'last_message_at', 'pinned', 'is_deleted', 'is_archive')
    list_filter = ('user', 'is_deleted', 'is_archive', 'pinned')
    search_fields = ('uuid', 'user__username', 'title')
    readonly_fields = ('uuid', 'user', 'title', 'last_message_at', 'pinned', 'is_deleted', 'is_archive', 'created_at', 'updated_at')
    inlines = [MessageInline]
    list_per_page = 25  # <-- Pagination for conversation list

    def has_add_permission(self, request):
        return False
    def has_change_permission(self, request, obj=None):
        return False
    def has_delete_permission(self, request, obj=None):
        return False

# Read-only admin for Message
@admin.register(Message)
class MessageAdmin(admin.ModelAdmin):
    list_display = ('uuid', 'conversation', 'sender', 'created_at', 'is_deleted')
    list_filter = ('sender', 'is_deleted')
    search_fields = ('uuid', 'conversation__uuid', 'conversation__title', 'text')
    readonly_fields = ('uuid', 'conversation', 'sender', 'text', 'created_at', 'image_url', 'token_count', 'response_time_ms', 'is_deleted', 'ai_model_response', 'updated_at')
    list_per_page = 50  # <-- Pagination for message list

    def has_add_permission(self, request):
        return False
    def has_change_permission(self, request, obj=None):
        return False
    def has_delete_permission(self, request, obj=None):
        return False
