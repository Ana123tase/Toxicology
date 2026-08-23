from django.core.management.base import BaseCommand
from subscriptions.models import SubscriptionPlan

class Command(BaseCommand):
    def handle(self, *args, **kwargs):
        plans = [
            {
                "name": "Premium Monthly",
                "slug": "premium-monthly",
                "billing_cycle": "monthly",
                "price": 5.00,
                "duration_days": 30,
                "features": ["premium_access", "advanced_reports", "purchases", "edit_payment"],
            },
            {
                "name": "Premium 6 Months",
                "slug": "premium-6months",
                "billing_cycle": "semi_annual",
                "price": 24.00,
                "duration_days": 180,
                "features": ["premium_access", "advanced_reports", "purchases", "edit_payment"],
            },
            {
                "name": "Premium Yearly",
                "slug": "premium-yearly",
                "billing_cycle": "yearly",
                "price": 42.00,
                "duration_days": 365,
                "features": ["premium_access", "advanced_reports", "purchases", "edit_payment", "multi_branch"],
            },
        ]
        for p in plans:
            obj, created = SubscriptionPlan.objects.update_or_create(slug=p['slug'], defaults=p)
            if created:
                self.stdout.write(f"Created {obj.name}")
            else:
                self.stdout.write(f"Updated {obj.name}")
        self.stdout.write("Done")