# subscriptions/decorators.py
from functools import wraps
from django.shortcuts import redirect
from django.contrib import messages
from django.utils import timezone


def _get_active_subscription(request):
    """
    Shared resolution logic for both decorators below — avoids duplicating
    (and potentially drifting from) the same profile/subscription lookup
    and expiry-flip logic in two places.

    Returns (subscription, error_redirect) — error_redirect is None if the
    subscription is valid and active; otherwise it's a redirect response
    the caller should return immediately.
    """
    profile = getattr(request.user, 'profile', None)
    if not profile:
        messages.error(request, "Please complete your profile first.")
        return None, redirect('profile_setup')  # confirm this is the real URL name

    try:
        sub = profile.subscription
    except Exception:
        # Covers Subscription.DoesNotExist (no row yet) and AttributeError
        # (profile.subscription relation misconfigured). If your signup flow
        # auto-creates a Free subscription for every new profile, this
        # branch should rarely fire — if it does, it likely means a profile
        # exists with no subscription row at all.
        messages.warning(request, "You need an active subscription to access this feature.")
        return None, redirect('subscription_plans')

    # Auto-expire check — mirrors Subscription.is_expired / has_feature()
    # in models.py. Kept here too since this decorator runs before those
    # model methods would otherwise be called.
    if sub.expires_at and sub.expires_at < timezone.now():
        if sub.is_active:
            sub.is_active = False
            sub.save(update_fields=['is_active'])
        messages.warning(request, "Your subscription has expired. Please renew.")
        return None, redirect('subscription_plans')

    if not sub.is_active:
        messages.warning(request, "Your subscription is inactive. Please renew.")
        return None, redirect('subscription_plans')

    return sub, None


def premium_required(view_func):
    """
    Requires ANY active, non-expired subscription — including Free tier.
    Per-plan daily prediction limits (Free = 2/day, Starter = 10/day, etc.)
    are NOT enforced here; that happens at the point of use via
    Subscription.can_make_prediction() / record_prediction().
    """
    @wraps(view_func)
    def _wrapped_view(request, *args, **kwargs):
        sub, error_redirect = _get_active_subscription(request)
        if error_redirect:
            return error_redirect
        return view_func(request, *args, **kwargs)
    return _wrapped_view


def subscription_feature_required(feature_name):
    """
    Gates a view on a specific feature flag in plan.features, e.g.
    @subscription_feature_required('some_feature')
    Not currently used anywhere in ToxiPhar, but kept available in case a
    future plan-specific (non-quota) feature difference is needed.
    """
    def decorator(view_func):
        @wraps(view_func)
        def _wrapped_view(request, *args, **kwargs):
            sub, error_redirect = _get_active_subscription(request)
            if error_redirect:
                return error_redirect

            if not sub.has_feature(feature_name):
                messages.warning(request, f"The '{feature_name}' feature requires a higher-tier plan.")
                return redirect('subscription_plans')

            return view_func(request, *args, **kwargs)
        return _wrapped_view
    return decorator