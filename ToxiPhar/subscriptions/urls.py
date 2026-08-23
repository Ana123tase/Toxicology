# subscriptions/urls.py
from django.urls import path
from . import views

urlpatterns = [
    path('plans/', views.subscription_plans_view, name='subscription_plans'),
    path('subscribe/<slug:slug>/', views.subscribe_view, name='subscribe_plan'),
    path('checkout/<slug:slug>/', views.subscription_checkout_view, name='subscription_checkout'),
]