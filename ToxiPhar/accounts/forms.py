from django import forms
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.models import User
from .models import PharmacistProfile
import re

import logging
from datetime import timedelta
from django.utils import timezone

from subscriptions.models import SubscriptionPlan, Subscription

logger = logging.getLogger(__name__)


class RegisterForm(UserCreationForm):
    first_name = forms.CharField(max_length=30, required=True)
    last_name = forms.CharField(max_length=30, required=True)
    email = forms.EmailField(required=True)

    institution = forms.CharField(max_length=150, required=True)
    phone_number = forms.CharField(max_length=20, required=True)
    country = forms.CharField(max_length=100, required=True)
    district = forms.CharField(max_length=100, required=True)
    sector = forms.CharField(max_length=100, required=True)

    class Meta:
        model = User
        fields = [
            'first_name', 'last_name', 'email',
            'username', 'password1', 'password2',
        ]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Username is now a normal, required field the user fills in
        # themselves — no auto-generation, no hiding.
        self.fields['username'].required = True

    def clean_email(self):
        email = self.cleaned_data['email']
        if User.objects.filter(email__iexact=email).exists():
            raise forms.ValidationError(
                "An account with this email already exists."
            )
        return email

    def clean_username(self):
        username = self.cleaned_data['username']
        if User.objects.filter(username__iexact=username).exists():
            raise forms.ValidationError(
                "This username is already taken. Please choose another."
            )
        return username

    def save(self, commit=True):
        user = super().save(commit=False)
        user.first_name = self.cleaned_data['first_name']
        user.last_name = self.cleaned_data['last_name']
        user.email = self.cleaned_data['email']
        user.username = self.cleaned_data['username']

        if commit:
            user.save()
            profile = PharmacistProfile.objects.create(
                user=user,
                institution=self.cleaned_data['institution'],
                phone_number=self.cleaned_data['phone_number'],
                country=self.cleaned_data['country'],
                district=self.cleaned_data['district'],
                sector=self.cleaned_data['sector'],
            )

            free_plan = SubscriptionPlan.objects.filter(
                tier=SubscriptionPlan.Tier.FREE,
                billing_cycle=SubscriptionPlan.BillingCycle.FREE,
                is_active=True,
            ).first()

            if free_plan:
                Subscription.objects.create(
                    researcher=profile,
                    plan=free_plan,
                    is_active=True,
                    expires_at=timezone.now() + timedelta(days=free_plan.duration_days),
                )
            else:
                logger.error(
                    "No active Free SubscriptionPlan found — user %s was "
                    "created without a subscription.", user.username
                )
        return user

class EditProfileForm(forms.ModelForm):
    first_name = forms.CharField(max_length=30, required=True)
    last_name = forms.CharField(max_length=30, required=True)
    email = forms.EmailField(required=True)

    class Meta:
        model = PharmacistProfile
        fields = [
            'institution', 'phone_number',
            'country', 'district', 'sector',
        ]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.instance and self.instance.user_id:
            self.fields['first_name'].initial = self.instance.user.first_name
            self.fields['last_name'].initial = self.instance.user.last_name
            self.fields['email'].initial = self.instance.user.email

    def clean_email(self):
        email = self.cleaned_data['email']
        existing = User.objects.filter(email__iexact=email)
        if self.instance and self.instance.user_id:
            existing = existing.exclude(pk=self.instance.user_id)
        if existing.exists():
            raise forms.ValidationError(
                "Another account is already using this email."
            )
        return email

    def clean_institution(self):
        institution = self.cleaned_data['institution']
        new_username = re.sub(r'[^a-zA-Z0-9]+', '_', institution).strip('_').lower()

        if not new_username:
            raise forms.ValidationError(
                "Pharmacy name must contain at least one letter or number."
            )

        existing = User.objects.filter(username=new_username).exclude(pk=self.instance.user_id)
        if existing.exists():
            raise forms.ValidationError(
                "This pharmacy name would create a username that's already taken. Please choose a different name."
            )
        return institution

    def save(self, commit=True):
        profile = super().save(commit=False)
        if commit:
            profile.user.first_name = self.cleaned_data['first_name']
            profile.user.last_name = self.cleaned_data['last_name']
            profile.user.email = self.cleaned_data['email']

            new_username = re.sub(r'[^a-zA-Z0-9]+', '_', self.cleaned_data['institution']).strip('_').lower()
            profile.user.username = new_username

            profile.user.save()
            profile.save()
        return profile