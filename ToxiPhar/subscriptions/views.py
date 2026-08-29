# subscriptions/views.py
from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.db import transaction, IntegrityError
from django.utils import timezone
from datetime import timedelta

from .models import SubscriptionPlan, Subscription


def get_user_subscription(profile):
    """Safe getter - OneToOne can raise DoesNotExist"""
    if profile is None:
        return None
    try:
        return profile.subscription
    except Subscription.DoesNotExist:
        return None
    except AttributeError:
        return None


# subscriptions/views.py — only subscription_plans_view changes

@login_required
def subscription_plans_view(request):
    plans_qs = SubscriptionPlan.objects.filter(is_active=True).order_by('tier', 'duration_days')

    # Serialized for the JS-driven tier/cycle tabs in the template — model
    # properties like price_per_month and get_billing_cycle_display() can't
    # be read directly from JSON, so they're resolved here in Python.
    plans_data = [
        {
            'name': p.name,
            'slug': p.slug,
            'tier': p.tier,
            'tier_display': p.get_tier_display(),
            'billing_cycle': p.billing_cycle,
            'billing_cycle_display': p.get_billing_cycle_display(),
            'price': str(p.price),
            'daily_prediction_limit': p.daily_prediction_limit,
            'price_per_month': str(p.price_per_month),
        }
        for p in plans_qs
    ]

    profile = getattr(request.user, 'profile', None)
    current_sub = get_user_subscription(profile) if profile else None

    return render(request, 'subscriptions/plans.html', {
        'plans_data': plans_data,
        'current_sub': current_sub,
    })


@login_required
def subscribe_view(request, slug):
    plan = get_object_or_404(SubscriptionPlan, slug=slug, is_active=True)

    # request.user is already the authenticated user object — no need to
    # re-fetch it from the DB just to reach .profile.
    profile = getattr(request.user, 'profile', None)
    if profile is None:
        messages.error(request, "Please complete your profile before subscribing.")
        return redirect('dashboard')  # adjust to your actual profile-setup URL name if different

    current_sub = get_user_subscription(profile)

    if current_sub:
        # SECURITY: switching plans must NOT grant immediate access to the
        # new plan's features. Deactivate until payment is verified, exactly
        # like a brand-new subscription would be. Do not touch expires_at
        # here — it's only recalculated once payment is confirmed
        # (presumably in a separate payment-webhook/verification view).
        current_sub.plan = plan
        current_sub.is_active = False
        current_sub.last_payment_ref = f"PENDING-{plan.slug}-{timezone.now().strftime('%Y%m%d')}"
        current_sub.save(update_fields=['plan', 'is_active', 'last_payment_ref'])
    else:
        try:
            with transaction.atomic():
                Subscription.objects.create(
                    researcher=profile,
                    plan=plan,
                    expires_at=timezone.now() + timedelta(days=plan.duration_days),
                    is_active=False,  # False until payment is verified
                    last_payment_ref=f"PENDING-{plan.slug}"
                )
        except IntegrityError:
            # Race: another concurrent request created the subscription first.
            # Re-fetch and apply the same pending-plan-switch update instead.
            current_sub = profile.subscription
            current_sub.plan = plan
            current_sub.is_active = False
            current_sub.last_payment_ref = f"PENDING-{plan.slug}-{timezone.now().strftime('%Y%m%d')}"
            current_sub.save(update_fields=['plan', 'is_active', 'last_payment_ref'])

    messages.info(request, f"You selected {plan.name}. Please complete payment to activate.")
    return redirect('subscription_checkout', slug=plan.slug)


@login_required
def subscription_checkout_view(request, slug):
    plan = get_object_or_404(SubscriptionPlan, slug=slug, is_active=True)

    profile = getattr(request.user, 'profile', None)
    current_sub = get_user_subscription(profile)

    # Built here (not in the template) so urlencode only needs to run once
    # on a finished string, rather than chaining |add filters in the
    # template — cleaner, and avoids missing an escape on one piece.
    whatsapp_message = (
        f"I paid for {plan.name}. "
        f"User: {request.user.username}. "
        f"Email: {request.user.email}"
    )
    email_subject = f"Payment Proof - {plan.name} - {request.user.username}"
    email_body = (
        "Hello ToxiPhar Team,\r\n\r\n"
        f"I paid for {plan.name} ({plan.price} {plan.currency}).\r\n"
        f"User: {request.user.username}\r\n"
        f"Email: {request.user.email}\r\n"
        f"Phone: {getattr(profile, 'phone_number', '')}\r\n"
        "Transaction ID: [paste here]\r\n\r\n"
        "Please activate my account.\r\nThanks."
    )

    return render(request, 'subscriptions/checkout.html', {
        'plan': plan,
        'current_sub': current_sub,
        'whatsapp_message': whatsapp_message,
        'email_subject': email_subject,
        'email_body': email_body,
    })