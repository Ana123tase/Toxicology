from django.shortcuts import redirect
from django.http import HttpResponseForbidden


class ProfileRequiredMiddleware:
    """
    1. Superusers can only access /admin/ from localhost (127.0.0.1).
       Anyone connecting from another machine cannot use superuser.
    2. Authenticated users without a PharmacistProfile visiting
       app pages are redirected to /admin/.
    """

    EXEMPT_PREFIXES = (
        '/accounts/login/',
        '/accounts/logout/',
        '/accounts/register/',
        '/accounts/password-reset/',
        '/accounts/reset/',
        '/static/',
    )

    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        user = request.user
        ip = request.META.get('REMOTE_ADDR', '')
        is_local = ip in ('127.0.0.1', '::1')

        # Rule 0: Anyone trying /admin/ from a remote machine → login page
        if request.path.startswith('/admin/') and not is_local:
            return redirect('/accounts/login/')

        # Rule 1: Superuser on local machine trying app pages → /admin/
        if (
            user.is_authenticated
            and user.is_superuser
            and not request.path.startswith('/admin/')
            and not any(request.path.startswith(p) for p in self.EXEMPT_PREFIXES)
        ):
            return redirect('/admin/')

        # Rule 2: Logged-in non-superuser with no profile
        if (
            user.is_authenticated
            and not user.is_superuser
            and not hasattr(user, 'profile')
            and not any(request.path.startswith(p) for p in self.EXEMPT_PREFIXES)
        ):
            return redirect('/accounts/login/')

        return self.get_response(request)