from django.utils import timezone

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
            except:
                pass  # never break request if subscription check fails

        response = self.get_response(request)
        return response