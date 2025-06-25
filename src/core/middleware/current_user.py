import threading

_user = threading.local()  # Thread-local storage for the user

def get_current_user():
    return getattr(_user, 'value', None)

class CurrentUserMiddleware:
    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        # No need to store user in thread-local storage anymore
        # Directly use request.user in the views to get the current authenticated user
        response = self.get_response(request)
        return response