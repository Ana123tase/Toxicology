# subscriptions/decorators.py
from django.shortcuts import redirect
from functools import wraps
from django.contrib import messages
from django.utils import timezone

def premium_required(view_func):
    """
    Blocks access if user has no active subscription.
    Usage:
    @login_required
    @premium_required
    def my_premium_view(request): ...
    """
    @wraps(view_func)
    def _wrapped_view(request, *args, **kwargs):
        # 1. Must have profile
        profile = getattr(request.user, 'profile', None)
        if not profile:
            messages.error(request, "Please complete your pharmacy profile first.")
            return redirect('profile_setup')  # or your profile url

        # 2. Must have subscription object
        try:
            sub = profile.subscription
        except Exception:
            messages.warning(request, "You need Premium to access this feature.")
            return redirect('subscription_plans')

        # 3. Must be active and not expired
        if not sub.is_active:
            messages.warning(request, "The feature you are requesting requires subscription and yours is inactive. Please renew!")
            return redirect('subscription_plans')

        if sub.expires_at and sub.expires_at < timezone.now():
            sub.is_active = False
            sub.save(update_fields=['is_active'])
            messages.warning(request, "Your Premium has expired. Please renew.")
            return redirect('subscription_plans')

        # 4. All good - allow access
        return view_func(request, *args, **kwargs)
    
    return _wrapped_view

def subscription_feature_required(feature_name):
    """
    For future: if you want different features
    Example: @subscription_feature_required('insurance')
    """
    def decorator(view_func):
        @wraps(view_func)
        def _wrapped_view(request, *args, **kwargs):
            profile = getattr(request.user, 'profile', None)
            if not profile:
                return redirect('subscription_plans')
            try:
                sub = profile.subscription
                if not sub.has_feature(feature_name):
                    messages.warning(request, f"Premium feature '{feature_name}' required.")
                    return redirect('subscription_plans')
            except Exception:
                messages.warning(request, "Premium subscription required.")
                return redirect('subscription_plans')
            
            return view_func(request, *args, **kwargs)
        return _wrapped_view
    return decorator