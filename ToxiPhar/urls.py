"""
URL configuration for ToxiPhar project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/6.1/topics/http/urls/
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
# toxpredict/urls.py
from django.urls import path, include
from django.contrib import admin

urlpatterns = [
    path("admin/", admin.site.urls),
    path('accounts/', include('accounts.urls')),
    path("api/", include("api.urls")),
    path("reports/", include("reports.urls")),
    path("", include("web.urls")),
    path('subscriptions/', include('subscriptions.urls')),
]


#curl -X POST http://127.0.0.1:8000/api/predict/ \ -H "Content-Type: application/json" \ -d '{"smiles": "CC(=O)Oc1ccccc1C(=O)O"}'
