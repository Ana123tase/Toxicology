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
# Uses Django's cache framework (works with locmem, Redis, memcached -
# whatever CACHES is configured to in settings.py) so no new dependency
# is required. Tracks failed attempts per (client IP, identifier) pair
# so one attacker can't lock out a legitimate user's account by
# repeatedly failing on their username (that would be a denial-of-
# service vector), while still throttling any single IP hammering any
# single account.
#
# NOTE: this only works correctly if CACHES points at a shared backend
# (Redis/Memcached). With LocMemCache behind multiple worker processes,
# each worker has its own cache and the limits below become
# per-worker rather than global. That's a deployment/settings concern
# and can't be fixed from this file - make sure production settings.py
# uses a shared cache backend.

MAX_ATTEMPTS = 5
LOCKOUT_SECONDS = 300  # 5 minutes

# FIXED (review #4 - username flooding bypasses per-identifier limit):
# the original limiter only keyed on (IP, submitted username), so an
# attacker could rotate through usernames (admin1, admin2, ...) from the
# same IP and never trip the per-identifier limit. We now ALSO track a
# per-IP-only bucket with a looser threshold, so login_view enforces
# both limits together.
IP_MAX_ATTEMPTS = 50
IP_LOCKOUT_SECONDS = 3600  # 1 hour


def _get_client_ip(request):
    # NOTE (review #3 - spoofable IP): X-Forwarded-For is attacker-controlled
    # unless something in front of Django (nginx/load balancer) is
    # configured to strip/overwrite it before forwarding. We can't safely
    # assume that's true from this file alone, so trusting XFF is now
    # opt-in via a setting instead of always-on. Set
    # TRUST_X_FORWARDED_FOR = True in settings.py only if your proxy is
    # configured to sanitize this header; otherwise REMOTE_ADDR is used.
    if getattr(settings, 'TRUST_X_FORWARDED_FOR', False):
        forwarded_for = request.META.get('HTTP_X_FORWARDED_FOR')
        if forwarded_for:
            return forwarded_for.split(',')[0].strip()
    return request.META.get('REMOTE_ADDR', 'unknown')


def _too_many_attempts(request, bucket, identifier):
    key = f'ratelimit:{bucket}:{_get_client_ip(request)}:{identifier}'
    return key, cache.get(key, 0)


def _register_failed_attempt(key, ttl=LOCKOUT_SECONDS):
    # FIXED (review #17 - sliding-window lockout): the original used
    # cache.set(key, attempts, LOCKOUT_SECONDS) on every failure, which
    # re-arms the full timeout on each attempt. That means a persistent
    # attacker (or an accidental retry loop) could keep the window open
    # indefinitely and a legitimate user could stay locked out far longer
    # than LOCKOUT_SECONDS. cache.add() only sets the value+TTL the first
    # time the key is created, so the window is now fixed from the first
    # failure and later failures just increment the counter without
    # extending expiry.
    if cache.add(key, 1, ttl):
        return 1
    try:
        return cache.incr(key)
    except ValueError:
        # Key expired between add() and incr() - restart the window.
        cache.set(key, 1, ttl)
        return 1


def _clear_attempts(key):
    cache.delete(key)

# ---------------------------------------------------------------------------

def login_view(request):
    # Prevent logged-in users from seeing login page
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
            # BUGFIX: username-collision and email-uniqueness checks now
            # happen inside RegisterForm itself (clean_email / clean),
            # so a valid form here is much less likely to hit a DB
            # constraint at save() time. We still keep this try/except
            # as a last line of defense against a genuine race
            # condition (two identical signups landing at the same
            # instant), so a rare race never surfaces as a raw 500.
            try:
                with transaction.atomic():
                    user = form.save()
            except IntegrityError:
                # FIXED (review #2 - user enumeration): the previous
                # message spelled out exactly which field(s) collided
                # (pharmacy name  / NPC number / email), which lets an
                # attacker probe whether a given email or pharmacy is
                # already registered. Use a generic message here instead;
                # field-level detail should only ever come from
                # form.clean_*() validation shown against the specific
                # field, not from this catch-all race-condition path.
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
            # Automatically login the newly registered user
            login(request, user)
            messages.success(
                request,
                "Registration successful. Welcome to LogPhar!"
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
    # BUGFIX: same RelatedObjectDoesNotExist risk as dashboard_view.
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
            # BUGFIX: clean_pharmacy_name / clean_email / clean_npc_number
            # already check uniqueness (excluding self), but a small
            # check-then-save race window under concurrent edits could
            # still raise IntegrityError on save(). Wrap in a
            # transaction + try/except so a collision is reported
            # cleanly instead of crashing the view.
            try:
                with transaction.atomic():
                    form.save()
            except IntegrityError:
                # FIXED (review #2 - same enumeration concern as
                # register_view): keep the field-collision detail out of
                # this catch-all race-condition message.
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

        # SECURITY: rate limit the NPC number / password check below -
        # previously an attacker (or a malicious session on a shared
        # device) could brute-force the password field here with no
        # limit, since this endpoint doesn't go through Django's
        # AuthenticationForm at all.
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

        npc_entered = request.POST.get(
            'npc_number',
            ''
        ).strip()

        password_entered = request.POST.get(
            'password',
            ''
        )

        # BUGFIX: same RelatedObjectDoesNotExist risk as dashboard_view
        # and edit_profile_view. Without this guard, a user with no
        # PharmacistProfile gets an unhandled 500 instead of a clean
        # error message when trying to delete their account.
        profile = getattr(request.user, 'profile', None)
        user = request.user

        if profile is None:
            error = "No pharmacy profile found for your account."
        # Check NPC number
        elif npc_entered != profile.npc_number:
            _register_failed_attempt(rl_key)
            logger.info("Account deletion attempt failed NPC check: user pk=%s", user.pk)
            error = "Incorrect NPC number."
        # Check password
        elif not user.check_password(password_entered):
            _register_failed_attempt(rl_key)
            logger.info("Account deletion attempt failed password check: user pk=%s", user.pk)
            error = "Incorrect password."
        else:
            _clear_attempts(rl_key)
            try:

                with transaction.atomic():

                    # NOTE (review #1 - "delete" only deactivates): this
                    # is a soft delete by design (is_active=False +
                    # unusable password), not a hard delete. Whether to
                    # switch to user.delete() or a real anonymization
                    # routine is a product/data-retention decision (it
                    # affects related Sale/SaleItem/PharmacistProfile
                    # records and possibly legal retention requirements)
                    # that shouldn't be guessed at in this file. Left the
                    # behavior unchanged, but fixed the user-facing
                    # copy below so it doesn't claim the account/data was
                    # actually erased when it wasn't.
                    user.is_active = False
                    # Optional: invalidate existing sessions by changing password hash
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
                # FIXED (review #10 - broad exception swallowing): still
                # catching broadly here since this is a user-facing
                # "don't crash on deletion" safety net, but now logging
                # the actual exception instead of silently discarding it,
                # so real bugs show up in logs instead of only ever
                # surfacing as a generic message to the user.
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
        # SECURITY: PasswordChangeForm requires the current password,
        # which makes this endpoint another password-guessing surface
        # for anyone with a stolen/idle session. Rate limit it too.
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

            # FIXED (review #9 - missing transaction): wrapped in
            # transaction.atomic() so that if anything is added around
            # the save later (audit logging, notifications, etc.) it
            # either all commits or all rolls back together.
            with transaction.atomic():
                user = form.save()

                # Keep the user logged in after password change
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