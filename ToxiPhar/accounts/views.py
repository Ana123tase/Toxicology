from django.conf import settings
from django.shortcuts import render, redirect
from django.contrib.auth import login, logout
from django.contrib.auth.forms import AuthenticationForm
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.db.models import Sum, F
from django.db import transaction, IntegrityError
from django.utils import timezone
from datetime import timedelta
from decimal import Decimal
from django.core.cache import cache
import logging
from .forms import RegisterForm, EditProfileForm
from django.contrib.auth.forms import PasswordChangeForm
from django.contrib.auth import update_session_auth_hash
from .models import PharmacistProfile


from django.core.exceptions import ObjectDoesNotExist
from django.db.models.functions import Coalesce
from django.db.models import Value
from django.db.models import DecimalField

logger = logging.getLogger(__name__)

# --- SECURITY: lightweight brute-force protection -------------------------
MAX_ATTEMPTS = 5
LOCKOUT_SECONDS = 300  # 5 minutes

IP_MAX_ATTEMPTS = 50
IP_LOCKOUT_SECONDS = 3600  # 1 hour


def _get_client_ip(request):
    if getattr(settings, 'TRUST_X_FORWARDED_FOR', False):
        forwarded_for = request.META.get('HTTP_X_FORWARDED_FOR')
        if forwarded_for:
            return forwarded_for.split(',')[0].strip()
    return request.META.get('REMOTE_ADDR', 'unknown')


def _too_many_attempts(request, bucket, identifier):
    key = f'ratelimit:{bucket}:{_get_client_ip(request)}:{identifier}'
    return key, cache.get(key, 0)


def _register_failed_attempt(key, ttl=LOCKOUT_SECONDS):
    if cache.add(key, 1, ttl):
        return 1
    try:
        return cache.incr(key)
    except ValueError:
        cache.set(key, 1, ttl)
        return 1


def _clear_attempts(key):
    cache.delete(key)

# ---------------------------------------------------------------------------

def login_view(request):
    if request.user.is_authenticated:
        return redirect('dashboard')
    form = AuthenticationForm()
    if request.method == 'POST':
        submitted_username = request.POST.get('username', '')
        ip_key = f'ratelimit:login_ip:{_get_client_ip(request)}'
        ip_attempts = cache.get(ip_key, 0)
        rl_key, attempts = _too_many_attempts(request, 'login', submitted_username)
        if attempts >= MAX_ATTEMPTS or ip_attempts >= IP_MAX_ATTEMPTS:
            logger.warning(
                "Login blocked by rate limit: ip=%s username=%s",
                _get_client_ip(request), submitted_username,
            )
            messages.error(
                request,
                "Too many failed login attempts. Please wait a few minutes "
                "and try again."
            )
            return render(request, 'accounts/login.html', {'form': form})
        form = AuthenticationForm(data=request.POST)
        if form.is_valid():
            _clear_attempts(rl_key)
            _clear_attempts(ip_key)
            user = form.get_user()
            login(request, user)
            return redirect('dashboard')
        else:
            _register_failed_attempt(rl_key, LOCKOUT_SECONDS)
            _register_failed_attempt(ip_key, IP_LOCKOUT_SECONDS)
            logger.info(
                "Failed login attempt: ip=%s username=%s",
                _get_client_ip(request), submitted_username,
            )
    return render(request, 'accounts/login.html', {'form': form})


@login_required
def dashboard_view(request):
    return render(request, 'accounts/dashboard.html',)


def register_view(request):
    if request.method == 'POST':
        form = RegisterForm(request.POST)
        if form.is_valid():
            try:
                with transaction.atomic():
                    user = form.save()
            except IntegrityError:
                logger.warning("Registration IntegrityError (race condition) for submitted data")
                messages.error(
                    request,
                    "Registration failed. Please double-check your details "
                    "and try again."
                )
                return render(
                    request,
                    'accounts/register.html',
                    {'form': form}
                )
            login(request, user)
            messages.success(
                request,
                "Registration successful. Welcome to ToxiPhar!"
            )
            return redirect('dashboard')
    else:
        form = RegisterForm()
    return render(
        request,
        'accounts/register.html',
        {'form': form}
    )


@login_required
def edit_profile_view(request):
    profile = getattr(request.user, 'profile', None)
    if profile is None:
        messages.error(
            request,
            "No pharmacy profile found for your account. "
            "Please contact support to set one up."
        )
        return redirect('dashboard')
    old_username = request.user.username
    if request.method == 'POST':
        form = EditProfileForm(
            request.POST,
            instance=profile
        )
        if form.is_valid():
            try:
                with transaction.atomic():
                    form.save()
            except IntegrityError:
                logger.warning("Profile update IntegrityError (race condition) for user pk=%s", request.user.pk)
                messages.error(
                    request,
                    "Update failed. Please double-check your details and "
                    "try again."
                )
                return render(
                    request,
                    'accounts/edit_profile.html',
                    {'form': form}
                )
            request.user.refresh_from_db()
            new_username = request.user.username

            if new_username != old_username:
                messages.success(
                    request,
                    f"Profile updated. Your username has changed to '{new_username}'."
                )
            else:
                messages.success(
                    request,
                    "Profile updated successfully."
                )
            return redirect('dashboard')
    else:
        form = EditProfileForm(
            instance=profile
        )
    return render(
        request,
        'accounts/edit_profile.html',
        {'form': form}
    )


@login_required
def delete_account_view(request):

    error = None

    if request.method == 'POST':

        rl_key, attempts = _too_many_attempts(
            request, 'delete_account', request.user.pk
        )
        if attempts >= MAX_ATTEMPTS:
            logger.warning(
                "Account deletion blocked by rate limit: user pk=%s ip=%s",
                request.user.pk, _get_client_ip(request),
            )
            error = (
                "Too many failed attempts. Please wait a few minutes "
                "and try again."
            )
            return render(
                request,
                'accounts/delete_account.html',
                {'error': error}
            )

        username_entered = request.POST.get(
            'username_confirm',
            ''
        ).strip()

        password_entered = request.POST.get(
            'password',
            ''
        )

        profile = getattr(request.user, 'profile', None)
        user = request.user

        if profile is None:
            error = "No pharmacy profile found for your account."
        elif username_entered != user.username:
            _register_failed_attempt(rl_key)
            logger.info("Account deletion attempt failed username check: user pk=%s", user.pk)
            error = "Incorrect username."
        elif not user.check_password(password_entered):
            _register_failed_attempt(rl_key)
            logger.info("Account deletion attempt failed password check: user pk=%s", user.pk)
            error = "Incorrect password."
        else:
            _clear_attempts(rl_key)
            try:
                with transaction.atomic():
                    user.is_active = False
                    user.set_unusable_password()
                    user.save()

                logout(request)

                logger.info("Account deactivated: user pk=%s", user.pk)

                messages.success(
                    request,
                    "Your account has been deleted and you have been "
                    "logged out."
                )

                return redirect('login')

            except Exception:
                logger.exception("Account deactivation failed for user pk=%s", user.pk)
                error = (
                    "Account deletion failed. "
                    "Please contact developer."
                )

    return render(
        request,
        'accounts/delete_account.html',
        {'error': error}
    )


@login_required
def change_password_view(request):
    if request.method == 'POST':
        rl_key, attempts = _too_many_attempts(
            request, 'change_password', request.user.pk
        )
        if attempts >= MAX_ATTEMPTS:
            logger.warning(
                "Password change blocked by rate limit: user pk=%s ip=%s",
                request.user.pk, _get_client_ip(request),
            )
            form = PasswordChangeForm(user=request.user)
            return render(
                request,
                'accounts/password_change.html',
                {'form': form, 'rate_limit_error': 'Too many failed attempts. Please wait a few minutes and try again.'}
            )
        form = PasswordChangeForm(
                    user=request.user,
                    data=request.POST
                )
        if form.is_valid():
            _clear_attempts(rl_key)

            with transaction.atomic():
                user = form.save()
                update_session_auth_hash(
                    request,
                    user
                )

            logger.info("Password changed: user pk=%s", user.pk)

            messages.success(
                request,
                "Password changed successfully."
            )
            return redirect('password_change_done')
        else:
            _register_failed_attempt(rl_key)
            logger.info("Failed password change attempt: user pk=%s", request.user.pk)

    else:
        form = PasswordChangeForm(
            user=request.user
        )

    return render(
        request,
        'accounts/password_change.html',
        {
            'form': form
        }
    )


@login_required
def password_change_done_view(request):
    return render(
        request,
        'accounts/password_change_done.html'
    )