from django import forms
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.models import User
from .models import PharmacistProfile
import re


class RegisterForm(UserCreationForm):
    first_name = forms.CharField(max_length=30, required=True)
    last_name = forms.CharField(max_length=30, required=True)
    email = forms.EmailField(required=True)

    pharmacy_name = forms.CharField(max_length=150, required=True)
    npc_number = forms.CharField(max_length=50, required=True, label="NPC Number (License)")
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
        self.fields['username'].required = False
        self.fields['username'].widget = forms.HiddenInput()

    def clean_email(self):
        # SECURITY: emails were previously not required to be unique,
        # which allows one person to register multiple pharmacy
        # accounts under the same email and complicates password
        # resets / support. Case-insensitive check since email
        # providers treat case as equivalent.
        email = self.cleaned_data['email']
        if User.objects.filter(email__iexact=email).exists():
            raise forms.ValidationError(
                "An account with this email already exists."
            )
        return email

    def clean_npc_number(self):
        npc_number = self.cleaned_data['npc_number']
        if PharmacistProfile.objects.filter(npc_number=npc_number).exists():
            raise forms.ValidationError(
                "This NPC number is already registered to another pharmacy."
            )
        return npc_number

    def clean(self):
        cleaned_data = super().clean()
        pharmacy_name = cleaned_data.get('pharmacy_name')
        if pharmacy_name:
            # Convert "City Pharmacy" → "city_pharmacy"
            clean_username = re.sub(r'[^a-zA-Z0-9]+', '_', pharmacy_name).strip('_').lower()

            # SECURITY/BUGFIX: previously this generated username was
            # never checked for uniqueness here, so two pharmacy names
            # normalizing to the same slug ("City Pharmacy" and
            # "CITY PHARMACY!!" both -> "city_pharmacy") would pass
            # validation and only fail later with an IntegrityError at
            # save() time (a 500 if uncaught). Catch it here instead so
            # it surfaces as a normal field error on pharmacy_name.
            if not clean_username:
                self.add_error(
                    'pharmacy_name',
                    "Pharmacy name must contain at least one letter or number."
                )
            elif User.objects.filter(username=clean_username).exists():
                self.add_error(
                    'pharmacy_name',
                    "This pharmacy name would create a username that's already "
                    "taken. Please choose a different name."
                )
            else:
                cleaned_data['username'] = clean_username
                self.cleaned_data['username'] = clean_username
        return cleaned_data

    def save(self, commit=True):
        user = super().save(commit=False)
        user.first_name = self.cleaned_data['first_name']
        user.last_name = self.cleaned_data['last_name']
        user.email = self.cleaned_data['email']
        user.username = self.cleaned_data['username']  # already sanitized + verified in clean()

        if commit:
            user.save()
            PharmacistProfile.objects.create(
                user=user,
                pharmacy_name=self.cleaned_data['pharmacy_name'],
                npc_number=self.cleaned_data['npc_number'],
                phone_number=self.cleaned_data['phone_number'],
                country=self.cleaned_data['country'],
                district=self.cleaned_data['district'],
                sector=self.cleaned_data['sector'],
            )
        return user


class EditProfileForm(forms.ModelForm):
    first_name = forms.CharField(max_length=30, required=True)
    last_name = forms.CharField(max_length=30, required=True)
    email = forms.EmailField(required=True)

    class Meta:
        model = PharmacistProfile
        fields = [
            'pharmacy_name', 'npc_number', 'phone_number',
            'country', 'district', 'sector',
        ]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.instance and self.instance.user_id:
            self.fields['first_name'].initial = self.instance.user.first_name
            self.fields['last_name'].initial = self.instance.user.last_name
            self.fields['email'].initial = self.instance.user.email

    def clean_email(self):
        # SECURITY: keep email unique across accounts here too,
        # excluding the current user so re-saving your own unchanged
        # email doesn't falsely trip the check.
        email = self.cleaned_data['email']
        existing = User.objects.filter(email__iexact=email)
        if self.instance and self.instance.user_id:
            existing = existing.exclude(pk=self.instance.user_id)
        if existing.exists():
            raise forms.ValidationError(
                "Another account is already using this email."
            )
        return email

    def clean_npc_number(self):
        # SECURITY/BUGFIX: npc_number is editable on this form but was
        # never re-checked for uniqueness on edit (only enforced by the
        # DB-level unique=True on the model, which would raise an
        # uncaught IntegrityError instead of a clean field error).
        npc_number = self.cleaned_data['npc_number']
        existing = PharmacistProfile.objects.filter(npc_number=npc_number)
        if self.instance and self.instance.pk:
            existing = existing.exclude(pk=self.instance.pk)
        if existing.exists():
            raise forms.ValidationError(
                "This NPC number is already registered to another pharmacy."
            )
        return npc_number

    def clean_pharmacy_name(self):
        pharmacy_name = self.cleaned_data['pharmacy_name']
        new_username = re.sub(r'[^a-zA-Z0-9]+', '_', pharmacy_name).strip('_').lower()

        if not new_username:
            raise forms.ValidationError(
                "Pharmacy name must contain at least one letter or number."
            )

        # Check if another user already has this username (excluding ourselves)
        existing = User.objects.filter(username=new_username).exclude(pk=self.instance.user_id)
        if existing.exists():
            raise forms.ValidationError(
                "This pharmacy name would create a username that's already taken. Please choose a different name."
            )
        return pharmacy_name

    def save(self, commit=True):
        profile = super().save(commit=False)
        if commit:
            profile.user.first_name = self.cleaned_data['first_name']
            profile.user.last_name = self.cleaned_data['last_name']
            profile.user.email = self.cleaned_data['email']

            # Regenerate username from (possibly updated) pharmacy name
            new_username = re.sub(r'[^a-zA-Z0-9]+', '_', self.cleaned_data['pharmacy_name']).strip('_').lower()
            profile.user.username = new_username

            profile.user.save()
            profile.save()
        return profile