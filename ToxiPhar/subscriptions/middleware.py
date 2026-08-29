import logging
from django.utils import timezone

logger = logging.getLogger(__name__)


class AutoExpireSubscriptionMiddleware:
    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        if request.user.is_authenticated:
            try:
                profile = getattr(request.user, 'profile', None)
                if profile:
                    sub = getattr(profile, 'subscription', None)
                    if sub and sub.is_active and sub.expires_at and sub.expires_at < timezone.now():
                        sub.is_active = False
                        sub.save(update_fields=['is_active'])
            except Exception:
                # Never break the request over a subscription check — but
                # still log it, so a real bug here doesn't go unnoticed.
                logger.exception("AutoExpireSubscriptionMiddleware failed for user_id=%s", request.user.id)

        response = self.get_response(request)
        return response