from django.contrib import admin
from django.utils import timezone
from datetime import timedelta

from .models import SubscriptionPlan, Subscription


@admin.register(SubscriptionPlan)
class PlanAdmin(admin.ModelAdmin):
    list_display = ('name', 'tier', 'billing_cycle', 'price', 'duration_days', 'daily_prediction_limit', 'is_active')
    list_filter = ('tier', 'billing_cycle', 'is_active')
    prepopulated_fields = {'slug': ('name',)}


@admin.register(Subscription)
class SubscriptionAdmin(admin.ModelAdmin):
    list_display = ('researcher', 'plan', 'is_active', 'expires_at', 'days_remaining_display', 'last_payment_ref')
    list_filter = ('is_active', 'plan')
    search_fields = ('researcher__institution', 'researcher__user__username')
    actions = ['activate_30_days', 'activate_180_days', 'activate_365_days', 'deactivate']

    def days_remaining_display(self, obj):
        return obj.days_remaining
    days_remaining_display.short_description = "Days Left"

    @admin.action(description="Activate for 30 Days")
    def activate_30_days(self, request, queryset):
        # Single bulk UPDATE instead of looping + saving each row —
        # every selected row gets the exact same expires_at value.
        queryset.update(is_active=True, expires_at=timezone.now() + timedelta(days=30))

    @admin.action(description="Activate for 180 Days")
    def activate_180_days(self, request, queryset):
        queryset.update(is_active=True, expires_at=timezone.now() + timedelta(days=180))

    @admin.action(description="Activate for 365 Days")
    def activate_365_days(self, request, queryset):
        queryset.update(is_active=True, expires_at=timezone.now() + timedelta(days=365))

    @admin.action(description="Deactivate / Expire")
    def deactivate(self, request, queryset):
        queryset.update(is_active=False)