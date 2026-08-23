from django.contrib import admin
from .models import SubscriptionPlan, Subscription
from django.utils import timezone
from datetime import timedelta

@admin.register(SubscriptionPlan)
class PlanAdmin(admin.ModelAdmin):
    list_display = ('name', 'price', 'duration_days', 'is_active')
    prepopulated_fields = {'slug': ('name',)}

@admin.register(Subscription)
class SubscriptionAdmin(admin.ModelAdmin):
    list_display = ('pharmacy', 'plan', 'is_active', 'expires_at', 'days_remaining_display', 'last_payment_ref')
    list_filter = ('is_active', 'plan')
    search_fields = ('pharmacy__pharmacy_name', 'pharmacy__user__username')
    actions = ['activate_30_days', 'activate_180_days', 'activate_365_days', 'deactivate']

    def days_remaining_display(self, obj):
        return obj.days_remaining
    days_remaining_display.short_description = "Days Left"

    @admin.action(description="Activate for 30 Days ($2)")
    def activate_30_days(self, request, queryset):
        for sub in queryset:
            sub.is_active = True
            sub.expires_at = timezone.now() + timedelta(days=30)
            sub.save()

    @admin.action(description="Activate for 180 Days ($9)")
    def activate_180_days(self, request, queryset):
        for sub in queryset:
            sub.is_active = True
            sub.expires_at = timezone.now() + timedelta(days=180)
            sub.save()

    @admin.action(description="Activate for 365 Days ($17)")
    def activate_365_days(self, request, queryset):
        for sub in queryset:
            sub.is_active = True
            sub.expires_at = timezone.now() + timedelta(days=365)
            sub.save()

    @admin.action(description="Deactivate / Expire")
    def deactivate(self, request, queryset):
        queryset.update(is_active=False)