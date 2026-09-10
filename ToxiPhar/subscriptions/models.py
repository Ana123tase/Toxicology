# subscriptions/models.py
from django.db import models, transaction, IntegrityError
from django.utils import timezone
from datetime import timedelta
from decimal import Decimal


class SubscriptionPlan(models.Model):
    class BillingCycle(models.TextChoices):
        FREE = 'free', 'Free'
        MONTHLY = 'monthly', 'Monthly'
        SEMI_ANNUAL = 'semi_annual', '6 Months'
        YEARLY = 'yearly', 'Yearly'

    class Tier(models.TextChoices):
        FREE = 'free', 'Free'
        STARTER = 'starter', 'Starter'
        GROWTH = 'growth', 'Growth'
        UNLIMITED = 'unlimited', 'Unlimited'

    name = models.CharField(max_length=50, unique=True)  # e.g. Starter Monthly
    slug = models.SlugField(unique=True)  # e.g. starter-monthly
    tier = models.CharField(max_length=20, choices=Tier.choices, db_index=True)
    billing_cycle = models.CharField(max_length=20, choices=BillingCycle.choices, db_index=True)

    price = models.DecimalField(max_digits=10, decimal_places=2)
    currency = models.CharField(max_length=10, default='USD')

    duration_days = models.IntegerField(help_text="How many days this payment gives")

    daily_prediction_limit = models.PositiveIntegerField(
        null=True, blank=True,
        help_text="Max predictions per day. Leave blank/null for unlimited."
    )

    features = models.JSONField(default=list, help_text='e.g. ["predictions", "advanced_reports"]')

    is_active = models.BooleanField(default=True, db_index=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['price']
        indexes = [
            models.Index(fields=['is_active', 'billing_cycle']),
            models.Index(fields=['is_active', 'tier']),
        ]

    def has_feature(self, feature_code: str) -> bool:
        return feature_code in self.features

    @property
    def is_unlimited_predictions(self):
        return self.daily_prediction_limit is None

    @property
    def price_per_month(self):
        if self.duration_days == 0:
            return self.price
        months = self.duration_days / 30
        return round(self.price / Decimal(months), 2)

    def __str__(self):
        return f"{self.name} - ${self.price}/{self.get_billing_cycle_display()}"


class Subscription(models.Model):
    researcher = models.OneToOneField(
        'accounts.PharmacistProfile',
        on_delete=models.CASCADE,
        related_name='subscription'
    )
    plan = models.ForeignKey(SubscriptionPlan, on_delete=models.PROTECT, related_name='subscriptions')

    is_active = models.BooleanField(default=True, db_index=True)
    started_at = models.DateTimeField(auto_now_add=True)
    expires_at = models.DateTimeField(db_index=True)

    last_payment_ref = models.CharField(max_length=100, blank=True)  # e.g. MoMo TX id, Stripe id

    class Meta:
        indexes = [
            models.Index(fields=['researcher', 'is_active']),
            models.Index(fields=['expires_at', 'is_active']),
        ]

    @property
    def is_expired(self):
        return self.expires_at < timezone.now() if self.expires_at else True

    # subscriptions/models.py, inside class Subscription

    @property
    def is_pro(self):
        """True only for an active, non-expired, paid-tier subscription.
        Free-tier subscriptions are 'active' by design but are never Pro."""
        if self.is_expired:
            return False
        if not self.is_active:
            return False
        return self.plan.tier != SubscriptionPlan.Tier.FREE


    @property
    def days_remaining(self):
        if self.is_expired:
            return 0
        delta = self.expires_at - timezone.now()
        return max(0, delta.days)

    def has_feature(self, feature_code: str) -> bool:
        if self.is_expired:
            if self.is_active:
                self.is_active = False
                self.save(update_fields=['is_active'])
            return False
        if not self.is_active:
            return False
        return self.plan.has_feature(feature_code)

    def get_daily_prediction_limit(self):
        """Returns None for unlimited, 0 if inactive/expired, otherwise
        the plan's configured daily limit."""
        if self.is_expired:
            if self.is_active:
                self.is_active = False
                self.save(update_fields=['is_active'])
            return 0
        if not self.is_active:
            return 0
        return self.plan.daily_prediction_limit

    def predictions_used_today(self):
        usage = PredictionUsage.objects.filter(
            researcher=self.researcher, date=timezone.localdate()
        ).only('count').first()
        return usage.count if usage else 0

    def can_make_prediction(self):
        limit = self.get_daily_prediction_limit()
        if limit is None:
            return True
        if limit == 0:
            return False
        return self.predictions_used_today() < limit

    def record_prediction(self):
        """Atomically increments today's prediction count. Call this only
        AFTER can_make_prediction() has been checked and the prediction
        actually ran — this call itself doesn't enforce the limit."""
        today = timezone.localdate()
        try:
            with transaction.atomic():
                PredictionUsage.objects.create(researcher=self.researcher, date=today, count=1)
                return
        except IntegrityError:
            pass  # a row for today already exists — fall through to increment it

        PredictionUsage.objects.filter(researcher=self.researcher, date=today).update(
            count=models.F('count') + 1
        )

    def renew(self, new_plan=None):
        plan = new_plan or self.plan
        self.plan = plan
        base = self.expires_at if not self.is_expired else timezone.now()
        self.expires_at = base + timedelta(days=plan.duration_days)
        self.is_active = True
        self.save()

    def __str__(self):
        return f"{self.researcher.institution} -> {self.plan.name} (expires {self.expires_at.date()})"


class PredictionUsage(models.Model):
    """Tracks how many predictions a researcher/institution has made on a
    given day. Used by Subscription.can_make_prediction() to enforce
    SubscriptionPlan.daily_prediction_limit."""
    researcher = models.ForeignKey(
        'accounts.PharmacistProfile',
        on_delete=models.CASCADE,
        related_name='prediction_usage'
    )
    date = models.DateField(db_index=True)
    count = models.PositiveIntegerField(default=0)

    class Meta:
        unique_together = ('researcher', 'date')
        indexes = [
            models.Index(fields=['researcher', 'date']),
        ]

    def __str__(self):
        return f"{self.researcher.institution} - {self.date}: {self.count} predictions"