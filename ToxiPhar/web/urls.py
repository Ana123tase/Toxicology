from django.urls import path
from django.shortcuts import render
from .views import home, predict_view, report_view

app_name = "web"

def login_placeholder(request):
    return render(request, "web/coming_soon.html", {"feature": "Login"})

def register_placeholder(request):
    return render(request, "web/coming_soon.html", {"feature": "Registration"})

urlpatterns = [
    path("", home, name="home"),
    path("predict/", predict_view, name="predict"),
    path("report/", report_view, name="report"),
]