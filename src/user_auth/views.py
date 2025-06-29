from django.db import models
from rest_framework_simplejwt.views import TokenObtainPairView
from rest_framework import serializers
from rest_framework.response import Response

from rest_framework import status
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated
from rest_framework_simplejwt.tokens import RefreshToken
from .models import TokenBlacklist
from rest_framework_simplejwt.exceptions import TokenError
from rest_framework.views import exception_handler
# Custom Token Serializer (optional, to extend with user info)
from django.shortcuts import render


def login_template_view(request):
    return render(request, 'login.html')


class CustomTokenObtainPairSerializer(serializers.Serializer):
    username = serializers.CharField()
    password = serializers.CharField()

    def validate(self, attrs):
        # Here you can validate user credentials manually if needed
        # Otherwise, you can use Django's built-in authentication
        return super().validate(attrs)


class CustomTokenObtainPairView(TokenObtainPairView):
    serializer_class = CustomTokenObtainPairSerializer

    def post(self, request, *args, **kwargs):
        # Call the parent class method for token creation
        response = super().post(request, *args, **kwargs)

        # Get the user associated with the request
        user = request.user

        # Add additional data to the response
        response.data['message'] = 'Login successful!'
        response.data['user_uuid'] = str(user.uuid)  # Add user UUID
        response.data['user_full_name'] = user.get_full_name()  # Add user full name

        return response
# models.py
class CustomTokenObtainPairView(TokenObtainPairView):
    def post(self, request, *args, **kwargs):
        response = super().post(request, *args, **kwargs)
        print(response.data)  # Log the response data
        return response



class LogoutAPIView(APIView):
    def post(self, request):
        try:
            refresh_token = request.data.get("refresh_token") or request.headers.get('Authorization')
            print(f"Received refresh_token: {refresh_token}")

            if not refresh_token:
                return Response({"detail": "Refresh token is required."}, status=status.HTTP_400_BAD_REQUEST)

            if refresh_token.startswith('Bearer '):
                refresh_token = refresh_token[7:]

            token = RefreshToken(refresh_token)
            token.blacklist()
            return Response({"detail": "Successfully logged out."}, status=status.HTTP_200_OK)

        except TokenError as e:
            print("TokenError:", e)
            if "Token is blacklisted" in str(e):
                return Response({"detail": "Already logged out."}, status=status.HTTP_200_OK)
            return Response({"detail": f"Token error: {str(e)}"}, status=status.HTTP_400_BAD_REQUEST)
        except Exception as e:
            # PRINT THE STACK TRACE for debugging
            import traceback; traceback.print_exc()
            return Response({"detail": f"An error occurred during logout: {str(e)}"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


# class LogoutAPIView(APIView):
#     def post(self, request):
#         """
#         This endpoint will blacklist the user's refresh token on logout.
#         """
#         try:
#             # Get the refresh token from the Authorization header
#             refresh_token = request.data.get("refresh_token") or request.headers.get('Authorization')
#             print("Received refresh_token:", refresh_token)

#             if not refresh_token:
#                 return Response({"detail": "Refresh token is required."}, status=status.HTTP_400_BAD_REQUEST)

#             # If the token is prefixed with 'Bearer ', remove it
#             if refresh_token.startswith('Bearer '):  
#                 refresh_token = refresh_token[7:]

#             # Create a RefreshToken instance from the refresh token
#             token = RefreshToken(refresh_token)

#             # Blacklist the refresh token
#             token.blacklist()

#             # Clear the user session by setting request.user to None
#             request.user = None  # Clear the user for this session
#             _user.value = None  # Clear thread-local storage

#             # Return a success message
#             return Response({"detail": "Successfully logged out."}, status=status.HTTP_200_OK)

#         except TokenError as e:
#             return Response({"detail": f"Token error: {str(e)}"}, status=status.HTTP_400_BAD_REQUEST)
#         except Exception as e:
#             return Response({"detail": "An error occurred during logout."}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
