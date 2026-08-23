from django.urls import path
from .views import download_report

urlpatterns = [path("download/", download_report, name="download")]