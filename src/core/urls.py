"""
URL configuration for core project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/5.0/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path, include
from django.views.generic import TemplateView
from user_auth.views import login_template_view

urlpatterns = [
    path('admin/', admin.site.urls),
    path("agent/", include("agent.urls")),
    path('api/sales/', include('sales_analyzer.urls')),
    path('api/', include('conversation.urls')),
    path('api/user/', include('user_auth.urls')),
    path('login/', login_template_view, name='login-template'),

]
