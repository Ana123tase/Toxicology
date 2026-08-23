from django.urls import path, reverse_lazy
from django.contrib.auth.mixins import LoginRequiredMixin
from django.contrib.auth.views import (
    LogoutView,
    PasswordResetView,
    PasswordResetDoneView,
    PasswordResetConfirmView,
    PasswordResetCompleteView,
)

from . import views


urlpatterns = [

    # Authentication
    path(
        'login/',
        views.login_view,
        name='login'
    ),

    path(
        'register/',
        views.register_view,
        name='register'
    ),

    path(
        'logout/',
        LogoutView.as_view(next_page='login'),
        name='logout'
    ),


    # Dashboard
    path(
        'dashboard/',
        views.dashboard_view,
        name='dashboard'
    ),


    # Profile
    path(
        'profile/edit/',
        views.edit_profile_view,
        name='edit_profile'
    ),


    # Account deletion
    path(
        'delete-account/',
        views.delete_account_view,
        name='delete_account'
    ),


    # ==========================
    # Password Reset Flow
    # ==========================

    path(
        'password-reset/',
        PasswordResetView.as_view(
            template_name='accounts/password_reset.html'
        ),
        name='password_reset'
    ),


    path(
        'password-reset/done/',
        PasswordResetDoneView.as_view(
            template_name='accounts/password_reset_done.html'
        ),
        name='password_reset_done'
    ),


    path(
        'reset/<uidb64>/<token>/',
        PasswordResetConfirmView.as_view(
            template_name='accounts/password_reset_confirm.html'
        ),
        name='password_reset_confirm'
    ),


    path(
        'reset/done/',
        PasswordResetCompleteView.as_view(
            template_name='accounts/password_reset_complete.html'
        ),
        name='password_reset_complete'
    ),

    # Password Change Flow
    # FIXED: this previously pointed at CustomPasswordChangeView, a thin
    # subclass of Django's built-in class-based PasswordChangeView. That
    # class has no knowledge of the rate-limiting logic in
    # views.change_password_view (_too_many_attempts, MAX_ATTEMPTS,
    # rate_limit_error, etc.), so the lockout feature was completely
    # unreachable from this URL no matter what was changed in views.py
    # or the template. Routed to the actual function-based view instead.
    path(
        'password-change/',
        views.change_password_view,
        name='password_change'
    ),
    path(
        'password-change/done/',
        views.password_change_done_view,
        name='password_change_done'
    ),
]