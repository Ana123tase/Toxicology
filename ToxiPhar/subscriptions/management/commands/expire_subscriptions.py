from django.core.management.base import BaseCommand
from django.utils import timezone
from subscriptions.models import Subscription

class Command(BaseCommand):
    help = 'Deactivates expired subscriptions'

    def handle(self, *args, **kwargs):
        expired = Subscription.objects.filter(
            is_active=True,
            expires_at__lt=timezone.now()
        )
        count = expired.update(is_active=False)
        self.stdout.write(f"Expired {count} subscriptions")