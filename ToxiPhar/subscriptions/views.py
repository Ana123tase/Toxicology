# subscriptions/views.py - FIXED
from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from .models import SubscriptionPlan, Subscription
from django.utils import timezone
from datetime import timedelta

def get_user_subscription(profile):
    """Safe getter - OneToOne can raise DoesNotExist"""
    try:
        return profile.subscription
    except Subscription.DoesNotExist:
        return None
    except AttributeError:
        return None

@login_required
def subscription_plans_view(request):
    plans = SubscriptionPlan.objects.filter(is_active=True).order_by('price')
    
    # Safe way to get profile + subscription
    profile = getattr(request.user, 'profile', None)
    current_sub = get_user_subscription(profile) if profile else None
    
    context = {
        'plans': plans,
        'current_sub': current_sub,
    }
    return render(request, 'subscriptions/plans.html', context)

@login_required
def subscribe_view(request, slug):
    plan = get_object_or_404(SubscriptionPlan, slug=slug, is_active=True)
    profile = get_object_or_404(request.user.__class__, id=request.user.id).profile
    
    # Try to get existing subscription
    current_sub = get_user_subscription(profile)
    
    if current_sub:
        current_sub.plan = plan
        current_sub.last_payment_ref = f"PENDING-{plan.slug}-{timezone.now().strftime('%Y%m%d')}"
        current_sub.save(update_fields=['plan', 'last_payment_ref'])
    else:
        # Create pending subscription
        Subscription.objects.create(
            pharmacy=profile,
            plan=plan,
            expires_at=timezone.now() + timedelta(days=plan.duration_days),
            is_active=False,  # False until you verify payment
            last_payment_ref=f"PENDING-{plan.slug}"
        )
    
    messages.info(request, f"You selected {plan.name}. Please complete payment to activate.")
    return redirect('subscription_checkout', slug=plan.slug)

@login_required
def subscription_checkout_view(request, slug):
    plan = get_object_or_404(SubscriptionPlan, slug=slug)
    profile = getattr(request.user, 'profile', None)
    current_sub = get_user_subscription(profile) if profile else None
    
    context = {
        'plan': plan,
        'current_sub': current_sub,
    }
    return render(request, 'subscriptions/checkout.html', context)