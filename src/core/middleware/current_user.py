from contextvars import ContextVar

__all__ = [
    # setters/clearers
    "set_current_admin_user", "clear_current_admin_user",
    "set_current_chat_user",  "clear_current_chat_user",
    # getters
    "get_current_user", "get_current_chat_user",
    # middleware
    "AttachCurrentAdminUserMiddleware",
]

# Separate contexts so you can address them independently
_admin_user_ctx: ContextVar = ContextVar("admin_user_ctx", default=None)
_chat_user_ctx:  ContextVar = ContextVar("chat_user_ctx",  default=None)

# -------- Admin (session-auth / Django admin) ----------
def set_current_admin_user(user):
    """Store the session-auth user (admin/staff) for this request."""
    return _admin_user_ctx.set(user)

def clear_current_admin_user(token=None):
    if token is not None:
        _admin_user_ctx.reset(token)
    else:
        _admin_user_ctx.set(None)

# -------- Chat (frontend JWT) ----------
def set_current_chat_user(user):
    """Store the chat/JWT user for this request (call in your API view)."""
    return _chat_user_ctx.set(user)

def clear_current_chat_user(token=None):
    if token is not None:
        _chat_user_ctx.reset(token)
    else:
        _chat_user_ctx.set(None)

# -------- Getters ----------
def get_current_user():
    """
    Preferred getter for code that expects the *admin/session* user.
    Returns the admin user if present; otherwise falls back to chat user.
    """
    user = _admin_user_ctx.get()
    return user if user is not None else _chat_user_ctx.get()

def get_current_chat_user():
    """
    Preferred getter for code that expects the *chat/JWT* user.
    Returns the chat user if present; otherwise falls back to admin user.
    """
    user = _chat_user_ctx.get()
    return user if user is not None else _admin_user_ctx.get()

# -------- Middleware for admin/session flows ----------
class AttachCurrentAdminUserMiddleware:
    """
    Captures request.user from session-auth (Django admin / staff) into the admin context.
    For JWT chat users, you will set the chat context *inside the DRF view*.
    """
    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        token = set_current_admin_user(getattr(request, "user", None))
        try:
            return self.get_response(request)
        finally:
            clear_current_admin_user(token)




# from contextvars import ContextVar
# import threading
# __all__ = [
#     "set_current_chat_user",
#     "get_current_chat_user",
#     "clear_current_chat_user",
#     "AttachCurrentChatUserMiddleware",
#     # back-compat:
#     "get_current_user",
# ]

# # ContextVar works in both WSGI and ASGI
# _current_chat_user: ContextVar = ContextVar("current_chat_user", default=None)
# _user = threading.local()  # Thread-local storage for the user

# def set_current_chat_user(user):
#     """
#     Set the current request's user into the context var.
#     Returns a token which you should pass to clear_current_chat_user(token).
#     """
#     return _current_chat_user.set(user)

# def get_current_chat_user():
#     """Get the current request's user (or None)."""
#     return _current_chat_user.get()

# # Back-compat alias for older imports
# def get_current_user():
#     return get_current_chat_user()

# def clear_current_chat_user(token=None):
#     """
#     Reset to the previous value if token is provided,
#     otherwise set it to None.
#     """
#     if token is not None:
#         _current_chat_user.reset(token)
#     else:
#         _current_chat_user.set(None)


# class CurrentUserMiddleware:
#     def __init__(self, get_response):
#         self.get_response = get_response

#     def __call__(self, request):
#         # No need to store user in thread-local storage anymore
#         # Directly use request.user in the views to get the current authenticated user
#         response = self.get_response(request)
#         return response

# class AttachCurrentChatUserMiddleware:
#     """
#     Optional: For session-auth (Django admin) this captures request.user early.
#     For JWT (DRF SimpleJWT), you'll still set the user inside the view (see step 2).
#     """
#     def __init__(self, get_response):
#         self.get_response = get_response

#     def __call__(self, request):
#         token = set_current_chat_user(getattr(request, "user", None))
#         try:
#             return self.get_response(request)
#         finally:
#             clear_current_chat_user(token)
