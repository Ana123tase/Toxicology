# subscriptions/models.py
from django.db import models
from django.utils import timezone
from datetime import timedelta
from decimal import Decimal

class SubscriptionPlan(models.Model):
    class BillingCycle(models.TextChoices):
        MONTHLY = 'monthly', 'Monthly'
        SEMI_ANNUAL = 'semi_annual', '6 Months'
        YEARLY = 'yearly', 'Yearly'

    name = models.CharField(max_length=50, unique=True)  # e.g. Premium Monthly
    slug = models.SlugField(unique=True)  # e.g. premium-monthly
    billing_cycle = models.CharField(max_length=20, choices=BillingCycle.choices, db_index=True)
    
    price = models.DecimalField(max_digits=10, decimal_places=2)  # 2.00, 9.00, 17.00
    currency = models.CharField(max_length=10, default='USD')
    
    duration_days = models.IntegerField(help_text="How many days this payment gives")
    
    # What features does this plan unlock? Same for all your premium plans for now
    features = models.JSONField(default=list, help_text='e.g. ["insurance", "advanced_reports"]')
    
    is_active = models.BooleanField(default=True, db_index=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['price']
        indexes = [
            models.Index(fields=['is_active', 'billing_cycle']),
        ]

    def has_feature(self, feature_code: str) -> bool:
        return feature_code in self.features

    @property
    def price_per_month(self):
        # For display: 2.00, 1.50, 1.42
        if self.duration_days == 0:
            return self.price
        months = self.duration_days / 30
        return round(self.price / Decimal(months), 2)

    def __str__(self):
        return f"{self.name} - ${self.price}/{self.get_billing_cycle_display()}"

class Subscription(models.Model):
    pharmacy = models.OneToOneField(
        'accounts.PharmacistProfile',
        on_delete=models.CASCADE,
        related_name='subscription'
    )
    plan = models.ForeignKey(SubscriptionPlan, on_delete=models.PROTECT, related_name='subscriptions')
    
    is_active = models.BooleanField(default=True, db_index=True)
    started_at = models.DateTimeField(auto_now_add=True)
    expires_at = models.DateTimeField(db_index=True)

    # For payment proof
    last_payment_ref = models.CharField(max_length=100, blank=True)  # e.g. MoMo TX id, Stripe id

    class Meta:
        indexes = [
            models.Index(fields=['pharmacy', 'is_active']),
            models.Index(fields=['expires_at', 'is_active']),
        ]

    @property
    def is_expired(self):
        return timezone.now() > self.expires_at

    @property
    def days_remaining(self):
        delta = self.expires_at - timezone.now()
        return max(0, delta.days)

    def has_feature(self, feature_code: str) -> bool:
        if not self.is_active or self.is_expired:
            return False
        return self.plan.has_feature(feature_code)

    def renew(self, new_plan=None):
        """Extend subscription"""
        plan = new_plan or self.plan
        self.plan = plan
        # If still active, extend from expiry, otherwise from now
        base = self.expires_at if not self.is_expired else timezone.now()
        self.expires_at = base + timedelta(days=plan.duration_days)
        self.is_active = True
        self.save()

    def __str__(self):
        return f"{self.pharmacy.pharmacy_name} -> {self.plan.name} (expires {self.expires_at.date()})"

    # subscriptions/models.py -> inside Subscription class

    def has_feature(self, feature_code: str) -> bool:
        # Auto-expire: if expired, turn off is_active and save
        if self.expires_at and self.expires_at < timezone.now():
            if self.is_active:
                self.is_active = False
                self.save(update_fields=['is_active'])
            return False
        
        if not self.is_active:
            return False
            
        return self.plan.has_feature(feature_code)

    @property
    def is_expired(self):
        return self.expires_at < timezone.now() if self.expires_at else True