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

        # Optionally, add more data to the response (like user info)
        response.data['message'] = 'Login successful!'
        return response

# models.py


class LogoutAPIView(APIView):
    # permission_classes = [IsAuthenticated]

    def post(self, request):
        """
        Blacklist the user's refresh token on logout.
        """
        try:
            # Get the current user's refresh token from the request
            refresh_token = request.data.get("refresh_token")

            if not refresh_token:
                return Response({"detail": "Refresh token is required."}, status=status.HTTP_400_BAD_REQUEST)

            # Create a RefreshToken instance from the refresh token
            token = RefreshToken(refresh_token)

            # Add the refresh token to the blacklist
            TokenBlacklist.objects.create(refresh_token=str(token))

            # Return a success message
            return Response({"detail": "Successfully logged out."}, status=status.HTTP_200_OK)

        except Exception as e:
            return Response({"detail": str(e)}, status=status.HTTP_400_BAD_REQUEST)
