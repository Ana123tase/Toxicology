from functools import wraps
from django.shortcuts import redirect
from django.contrib import messages
from django.utils import timezone

from reports.models import PredictionReport


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
        messages.warning(request, "You need an active subscription to access this feature.")
        return None, redirect('subscription_plans')

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
    Requires an active, non-expired subscription — covers every plan
    (Free, Starter, Growth, Unlimited; monthly/semi_annual/yearly alike) —
    AND enforces that plan's daily prediction limit, using PredictionReport
    as the source of truth for how many predictions the user has already
    made today.

    - On GET: only subscription validity is checked (form renders as
      normal; no quota check since no prediction is being made yet).
    - On POST: today's PredictionReport rows for this user are counted
      BEFORE the view runs. If the plan has a numeric daily_prediction_limit
      and the count already meets/exceeds it, the view is never called and
      the user is redirected with a warning instead. Unlimited plans
      (daily_prediction_limit=None, e.g. the Unlimited tier) skip the count
      entirely and always pass through.
    """
    @wraps(view_func)
    def _wrapped_view(request, *args, **kwargs):
        sub, error_redirect = _get_active_subscription(request)
        if error_redirect:
            return error_redirect

        if request.method == 'POST':
            limit = sub.get_daily_prediction_limit()
            if limit is not None:  # None == unlimited plan, skip the count entirely
                today = timezone.localdate()
                used_today = PredictionReport.objects.filter(
                    user=request.user,
                    created_at__date=today,
                ).count()

                if used_today >= limit:
                    messages.warning(
                        request,
                        f"You've reached your daily limit of {limit} prediction(s) "
                        f"for your current plan ({sub.plan.name}). "
                        f"Please upgrade or wait until tomorrow."
                    )
                    return redirect('subscription_plans')

        return view_func(request, *args, **kwargs)
    return _wrapped_view


def subscription_feature_required(feature_name):
    """
    Gates a view on a specific feature flag in plan.features, e.g.
    @subscription_feature_required('some_feature')
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