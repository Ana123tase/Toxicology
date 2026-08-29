# accounts/management/commands/seed_plans.py  (or wherever your command lives)
from decimal import Decimal
from django.core.management.base import BaseCommand
from subscriptions.models import SubscriptionPlan


class Command(BaseCommand):
    help = "Seeds ToxiPhar's subscription plans: Free, Starter, Growth, Unlimited."

    def handle(self, *args, **kwargs):
        plans = [
            {
                "name": "Free",
                "slug": "free",
                "tier": "free",
                "billing_cycle": "free",
                "price": Decimal("0.00"),
                "duration_days": 36500,  # effectively non-expiring
                "daily_prediction_limit": 2,
                "features": ["predictions"],
            },
            {
                "name": "Starter Monthly",
                "slug": "starter-monthly",
                "tier": "starter",
                "billing_cycle": "monthly",
                "price": Decimal("10.00"),
                "duration_days": 30,
                "daily_prediction_limit": 10,
                "features": ["predictions"],
            },
            {
                "name": "Starter 6 Months",
                "slug": "starter-6months",
                "tier": "starter",
                "billing_cycle": "semi_annual",
                "price": Decimal("50.00"),
                "duration_days": 182,
                "daily_prediction_limit": 10,
                "features": ["predictions"],
            },
            {
                "name": "Starter Yearly",
                "slug": "starter-yearly",
                "tier": "starter",
                "billing_cycle": "yearly",
                "price": Decimal("500.00"),
                "duration_days": 365,
                "daily_prediction_limit": 10,
                "features": ["predictions"],
            },
            {
                "name": "Growth Monthly",
                "slug": "growth-monthly",
                "tier": "growth",
                "billing_cycle": "monthly",
                "price": Decimal("20.00"),
                "duration_days": 30,
                "daily_prediction_limit": 30,
                "features": ["predictions"],
            },
            {
                "name": "Growth 6 Months",
                "slug": "growth-6months",
                "tier": "growth",
                "billing_cycle": "semi_annual",
                "price": Decimal("80.00"),
                "duration_days": 182,
                "daily_prediction_limit": 30,
                "features": ["predictions"],
            },
            {
                "name": "Growth Yearly",
                "slug": "growth-yearly",
                "tier": "growth",
                "billing_cycle": "yearly",
                "price": Decimal("800.00"),
                "duration_days": 365,
                "daily_prediction_limit": 30,
                "features": ["predictions"],
            },
            {
                "name": "Unlimited Monthly",
                "slug": "unlimited-monthly",
                "tier": "unlimited",
                "billing_cycle": "monthly",
                "price": Decimal("50.00"),
                "duration_days": 30,
                "daily_prediction_limit": None,
                "features": ["predictions"],
            },
            {
                "name": "Unlimited 6 Months",
                "slug": "unlimited-6months",
                "tier": "unlimited",
                "billing_cycle": "semi_annual",
                "price": Decimal("200.00"),
                "duration_days": 182,
                "daily_prediction_limit": None,
                "features": ["predictions"],
            },
            {
                "name": "Unlimited Yearly",
                "slug": "unlimited-yearly",
                "tier": "unlimited",
                "billing_cycle": "yearly",
                "price": Decimal("1500.00"),
                "duration_days": 365,
                "daily_prediction_limit": None,
                "features": ["predictions"],
            },
        ]

        for p in plans:
            obj, created = SubscriptionPlan.objects.update_or_create(
                slug=p["slug"], defaults=p
            )
            if created:
                self.stdout.write(self.style.SUCCESS(f"Created {obj.name}"))
            else:
                self.stdout.write(f"Updated {obj.name}")

        self.stdout.write(self.style.SUCCESS("Done"))