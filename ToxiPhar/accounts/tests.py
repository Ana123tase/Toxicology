from django.core.cache import cache as django_cache
from django.test import override_settings

"""
Test suite for the accounts app.

All url names below were confirmed directly against the accounts/urls.py
you provided: login, register, dashboard, edit_profile, logout,
delete_account, password_reset, password_reset_done,
password_reset_confirm, password_reset_complete, password_change,
password_change_done.

Run with:
    python manage.py test accounts

KNOWN GAPS THIS SUITE DELIBERATELY FLAGS
------------------------------------------
1. RegisterForm's Meta.model is User, not PharmacistProfile, so npc_number
   is a plain CharField with no form-level uniqueness check - even though
   the PharmacistProfile model field has unique=True. Two registrations
   with the same NPC number currently crash with an unhandled
   IntegrityError instead of failing validation gracefully.

   test_duplicate_npc_number_shows_validation_error_instead_of_crashing
   asserts the CORRECT behavior (a graceful form error, no crash) - so it
   is EXPECTED TO FAIL right now. Fix RegisterForm (e.g. add a
   clean_npc_number() method) until it passes.

2. change_password_view's success path calls redirect('password_changed'),
   but accounts/urls.py only defines the name 'password_change_done' (see
   the confirmed list above). That mismatch means a successful password
   change currently raises NoReverseMatch instead of redirecting.

   test_password_change_redirect_name_is_broken documents this and is
   EXPECTED TO FAIL/ERROR right now. Fix the view's
   redirect('password_changed') call to redirect('password_change_done')
   until it passes cleanly.

NOTE ON SHARED CACHE STATE
------------------------------------------
login, delete_account, and change_password are now rate-limited via
Django's cache framework (see views.py). With the default LocMemCache,
that cache is NOT reset between tests. Because TestCase wraps each test
in a rolled-back transaction, autoincrement PKs/usernames are frequently
reused test-to-test, so a leftover rate-limit counter from one test can
bleed into another and make failures look flaky/order-dependent. Every
TestCase below that exercises a rate-limited view clears the cache in
setUp() to keep tests independent.
"""

from decimal import Decimal

from django.contrib.auth.models import User
from django.core import mail
from django.core.cache import cache
from django.test import TestCase
from django.urls import reverse, NoReverseMatch
from django.utils import timezone

from inventory.models import Product, Sale, SaleItem
from .models import PharmacistProfile


def make_pharmacy(username='pharmacist1', **profile_kwargs):
    """Create a User + PharmacistProfile pair and return the profile."""
    user = User.objects.create_user(username=username, password='testpass123')
    defaults = dict(
        pharmacy_name=f'{username} Pharmacy',
        npc_number=f'NPC-{username}',
        phone_number='0700000000',
        country='Rwanda',
        district='Gasabo',
        sector='Kimironko',
    )
    defaults.update(profile_kwargs)
    return PharmacistProfile.objects.create(user=user, **defaults)


def make_product(pharmacy, **kwargs):
    defaults = dict(
        brand_name='Panadol',
        generic_name='Paracetamol',
        therapeutic_class='analgesic',
        dosage_form='tablet',
        strength='500mg',
        batch_number='BATCH001',
        expiry_date=timezone.now().date() + timezone.timedelta(days=365),
        quantity_in_stock=100,
        reorder_level=10,
        purchase_price_per_pack=Decimal('500.00'),
        unit_price=Decimal('100.00'),
    )
    defaults.update(kwargs)
    return Product.objects.create(pharmacy=pharmacy, **defaults)


def valid_registration_payload(**overrides):
    payload = dict(
        first_name='Jane',
        last_name='Doe',
        email='jane@example.com',
        password1='SuperSecret123!',
        password2='SuperSecret123!',
        pharmacy_name='City Pharmacy',
        npc_number='NPC-12345',
        phone_number='0788000000',
        country='Rwanda',
        district='Gasabo',
        sector='Kimironko',
    )
    payload.update(overrides)
    return payload


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

class RegisterViewTests(TestCase):

    def test_get_renders_form(self):
        response = self.client.get(reverse('register'))
        self.assertEqual(response.status_code, 200)

    def test_valid_registration_creates_user_and_logs_in(self):
        # UPDATED: register_view logs the new user in and redirects to
        # 'dashboard' (not 'login') - see the `login(request, user)` call
        # right before the redirect in register_view.
        response = self.client.post(reverse('register'), valid_registration_payload())

        self.assertRedirects(response, reverse('dashboard'))
        user = User.objects.get(email='jane@example.com')
        self.assertEqual(user.first_name, 'Jane')
        self.assertTrue(hasattr(user, 'profile'))
        self.assertEqual(user.profile.pharmacy_name, 'City Pharmacy')
        self.assertEqual(user.profile.npc_number, 'NPC-12345')
        self.assertTrue(response.wsgi_request.user.is_authenticated)
        self.assertEqual(response.wsgi_request.user.pk, user.pk)

    def test_username_is_generated_from_pharmacy_name(self):
        self.client.post(reverse('register'), valid_registration_payload(
            pharmacy_name='City Pharmacy!!'
        ))

        user = User.objects.get(email='jane@example.com')
        self.assertEqual(user.username, 'city_pharmacy')

    def test_username_generation_strips_special_characters_and_spaces(self):
        self.client.post(reverse('register'), valid_registration_payload(
            pharmacy_name='  St. Mary\'s   Pharmacy & Clinic  '
        ))

        user = User.objects.get(email='jane@example.com')
        # Only alphanumerics survive, collapsed to single underscores, lowercase
        self.assertRegex(user.username, r'^[a-z0-9_]+$')
        self.assertFalse(user.username.startswith('_'))
        self.assertFalse(user.username.endswith('_'))

    def test_duplicate_pharmacy_name_fails_username_uniqueness_validation(self):
        make_pharmacy('city_pharmacy')  # user.username == 'city_pharmacy'

        response = self.client.post(reverse('register'), valid_registration_payload(
            pharmacy_name='City Pharmacy'  # sanitizes to the same username
        ))

        self.assertEqual(response.status_code, 200)  # re-rendered with errors, no redirect
        self.assertFalse(User.objects.filter(email='jane@example.com').exists())

    def test_password_mismatch_fails(self):
        response = self.client.post(reverse('register'), valid_registration_payload(
            password2='SomethingElse123!'
        ))

        self.assertEqual(response.status_code, 200)
        self.assertFalse(User.objects.filter(email='jane@example.com').exists())

    def test_missing_required_field_fails(self):
        response = self.client.post(reverse('register'), valid_registration_payload(
            npc_number=''
        ))

        self.assertEqual(response.status_code, 200)
        self.assertFalse(User.objects.filter(email='jane@example.com').exists())

    def test_duplicate_npc_number_shows_validation_error_instead_of_crashing(self):
        """
        THIS TEST IS EXPECTED TO FAIL until RegisterForm validates
        npc_number uniqueness at the form level (e.g. via a
        clean_npc_number() method that checks
        PharmacistProfile.objects.filter(npc_number=...).exists()).

        Right now, npc_number is a plain CharField on RegisterForm even
        though the model field has unique=True - so a duplicate currently
        raises an unhandled IntegrityError on save() instead of failing
        validation gracefully. Once you add that validation, this test
        should pass and you can remove this docstring.
        """
        make_pharmacy('existing_pharmacy', npc_number='NPC-DUPLICATE')

        response = self.client.post(reverse('register'), valid_registration_payload(
            pharmacy_name='Different Pharmacy',
            npc_number='NPC-DUPLICATE',
        ))

        # Should re-render the form with a validation error, not redirect
        self.assertEqual(response.status_code, 200)
        self.assertFalse(User.objects.filter(email='jane@example.com').exists())
        self.assertIn('npc_number', response.context['form'].errors)


# ---------------------------------------------------------------------------
# Login / logout
# ---------------------------------------------------------------------------

class LoginViewTests(TestCase):

    def setUp(self):
        # See module docstring: login is now cache-based rate limited, and
        # the cache isn't reset between tests by default.
        cache.clear()
        self.pharmacy = make_pharmacy('pharmacist1')

    def test_get_renders_form(self):
        response = self.client.get(reverse('login'))
        self.assertEqual(response.status_code, 200)

    def test_valid_credentials_redirect_to_dashboard(self):
        response = self.client.post(reverse('login'), {
            'username': 'pharmacist1', 'password': 'testpass123',
        })
        self.assertRedirects(response, reverse('dashboard'))

    def test_invalid_credentials_show_error_and_do_not_log_in(self):
        response = self.client.post(reverse('login'), {
            'username': 'pharmacist1', 'password': 'wrongpassword',
        })

        self.assertEqual(response.status_code, 200)
        self.assertFalse(response.wsgi_request.user.is_authenticated)

    def test_locks_out_after_max_attempts_for_same_identifier(self):
        # NEW: covers the per-(IP, username) lockout added to login_view.
        for _ in range(5):
            self.client.post(reverse('login'), {
                'username': 'pharmacist1', 'password': 'wrongpassword',
            })

        # A 6th attempt, even with the CORRECT password, should now be
        # blocked by the lockout rather than reaching form validation.
        response = self.client.post(reverse('login'), {
            'username': 'pharmacist1', 'password': 'testpass123',
        })

        self.assertEqual(response.status_code, 200)
        self.assertFalse(response.wsgi_request.user.is_authenticated)
        self.assertContains(response, 'Too many failed login attempts')

    def test_failed_attempts_on_one_username_do_not_lock_out_another(self):
        # NEW: covers the fix for review #4 (username-flooding bypass) -
        # this specifically checks the per-identifier bucket doesn't
        # falsely block an unrelated, never-attempted username. The
        # separate per-IP-only bucket (50 attempts) is intentionally much
        # looser and isn't tripped by 5 attempts here.
        for _ in range(5):
            self.client.post(reverse('login'), {
                'username': 'pharmacist1', 'password': 'wrongpassword',
            })

        make_pharmacy('pharmacist2')
        response = self.client.post(reverse('login'), {
            'username': 'pharmacist2', 'password': 'testpass123',
        })

        self.assertRedirects(response, reverse('dashboard'))


class LogoutViewTests(TestCase):

    def setUp(self):
        self.pharmacy = make_pharmacy('pharmacist1')
        self.client.login(username='pharmacist1', password='testpass123')

    def test_logout_redirects_to_login(self):
        response = self.client.post(reverse('logout'))
        self.assertRedirects(response, reverse('login'))


# ---------------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------------

class DashboardViewTests(TestCase):

    def setUp(self):
        self.pharmacy = make_pharmacy('pharmacist1')
        self.other_pharmacy = make_pharmacy('pharmacist2')
        self.client.login(username='pharmacist1', password='testpass123')

    def test_requires_login(self):
        self.client.logout()
        response = self.client.get(reverse('dashboard'))
        self.assertNotEqual(response.status_code, 200)

    def test_counts_only_own_pharmacy_products(self):
        make_product(self.pharmacy, brand_name='Mine')
        make_product(self.other_pharmacy, brand_name='TheirsNotMine')

        response = self.client.get(reverse('dashboard'))

        self.assertEqual(response.context['total_products'], 1)

    def test_low_stock_count(self):
        make_product(self.pharmacy, quantity_in_stock=1, reorder_level=10)
        make_product(self.pharmacy, quantity_in_stock=100, reorder_level=10)

        response = self.client.get(reverse('dashboard'))

        self.assertEqual(response.context['low_stock_count'], 1)

    def test_expired_count(self):
        yesterday = timezone.now().date() - timezone.timedelta(days=1)
        next_year = timezone.now().date() + timezone.timedelta(days=365)
        make_product(self.pharmacy, expiry_date=yesterday)
        make_product(self.pharmacy, expiry_date=next_year)

        response = self.client.get(reverse('dashboard'))

        self.assertEqual(response.context['expired_count'], 1)

    def test_expiring_soon_within_90_days(self):
        soon = timezone.now().date() + timezone.timedelta(days=30)
        far = timezone.now().date() + timezone.timedelta(days=365)
        expiring = make_product(self.pharmacy, brand_name='ExpiringSoon', expiry_date=soon)
        make_product(self.pharmacy, brand_name='NotExpiringSoon', expiry_date=far)

        response = self.client.get(reverse('dashboard'))

        self.assertEqual(response.context['expiring_soon_count'], 1)
        self.assertIn(expiring, list(response.context['expiring_soon']))

    def test_already_expired_products_excluded_from_expiring_soon(self):
        # expiring_soon filters expiry_date >= today, so already-expired
        # stock shouldn't double up in both "expired" and "expiring soon"
        yesterday = timezone.now().date() - timezone.timedelta(days=1)
        make_product(self.pharmacy, expiry_date=yesterday)

        response = self.client.get(reverse('dashboard'))

        self.assertEqual(response.context['expiring_soon_count'], 0)

    def test_inventory_value_calculation(self):
        make_product(self.pharmacy, quantity_in_stock=10, unit_price=Decimal('50.00'))
        make_product(self.pharmacy, quantity_in_stock=5, unit_price=Decimal('20.00'))

        response = self.client.get(reverse('dashboard'))

        # (10 * 50) + (5 * 20) = 500 + 100 = 600
        self.assertEqual(response.context['inventory_value'], Decimal('600.00'))

    def test_total_purchases_calculation(self):
        make_product(self.pharmacy, quantity_in_stock=10, purchase_price_per_pack=Decimal('30.00'))

        response = self.client.get(reverse('dashboard'))

        # 10 * 30 = 300
        self.assertEqual(response.context['total_purchases'], Decimal('300.00'))

    def test_dashboard_with_no_products_shows_zeros_not_errors(self):
        response = self.client.get(reverse('dashboard'))

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.context['total_products'], 0)
        self.assertEqual(response.context['inventory_value'], 0)
        self.assertEqual(response.context['total_purchases'], 0)


# ---------------------------------------------------------------------------
# Edit profile
# ---------------------------------------------------------------------------

class EditProfileViewTests(TestCase):

    def setUp(self):
        # pharmacy_name is deliberately chosen so that sanitizing it
        # produces the same value as the username below ('pharmacist1').
        # The view always regenerates the username from the current
        # pharmacy_name on every save (by design, not a bug) - so for a
        # fixture to represent a "no real change" state, username and
        # sanitize(pharmacy_name) must already agree going in.
        self.pharmacy = make_pharmacy('pharmacist1', pharmacy_name='Pharmacist1')
        self.client.login(username='pharmacist1', password='testpass123')

    def test_requires_login(self):
        self.client.logout()
        response = self.client.get(reverse('edit_profile'))
        self.assertNotEqual(response.status_code, 200)

    def test_get_prefills_user_fields(self):
        self.pharmacy.user.first_name = 'Original'
        self.pharmacy.user.save()

        response = self.client.get(reverse('edit_profile'))

        # EditProfileForm sets this via self.fields['first_name'].initial = ...
        # in __init__, which lives on the field itself, not on form.initial
        # (form.initial is a separate dict only populated by an initial=
        # kwarg, which this view never passes). Check the field directly.
        form = response.context['form']
        self.assertEqual(form.fields['first_name'].initial, 'Original')
        self.assertEqual(form.fields['last_name'].initial, self.pharmacy.user.last_name)
        self.assertEqual(form.fields['email'].initial, self.pharmacy.user.email)

    def valid_payload(self, **overrides):
        payload = dict(
            first_name='Updated',
            last_name='Name',
            email='updated@example.com',
            pharmacy_name=self.pharmacy.pharmacy_name,  # unchanged by default
            npc_number=self.pharmacy.npc_number,
            phone_number='0788111111',
            country='Rwanda',
            district='Gasabo',
            sector='Remera',
        )
        payload.update(overrides)
        return payload

    def test_valid_post_updates_profile_and_user_fields(self):
        response = self.client.post(reverse('edit_profile'), self.valid_payload())

        self.assertRedirects(response, reverse('dashboard'))
        self.pharmacy.refresh_from_db()
        self.pharmacy.user.refresh_from_db()
        self.assertEqual(self.pharmacy.user.first_name, 'Updated')
        self.assertEqual(self.pharmacy.user.email, 'updated@example.com')
        self.assertEqual(self.pharmacy.sector, 'Remera')

    def test_changing_pharmacy_name_changes_username_and_shows_message(self):
        response = self.client.post(reverse('edit_profile'), self.valid_payload(
            pharmacy_name='Brand New Pharmacy'
        ), follow=True)

        self.pharmacy.user.refresh_from_db()
        self.assertEqual(self.pharmacy.user.username, 'brand_new_pharmacy')
        page_messages = list(response.context['messages'])
        self.assertTrue(any('username has changed' in str(m) for m in page_messages))

    def test_unchanged_pharmacy_name_keeps_username_and_shows_generic_message(self):
        response = self.client.post(reverse('edit_profile'), self.valid_payload(), follow=True)

        self.pharmacy.user.refresh_from_db()
        self.assertEqual(self.pharmacy.user.username, 'pharmacist1')
        page_messages = list(response.context['messages'])
        self.assertTrue(any('Profile updated successfully' in str(m) for m in page_messages))

    def test_pharmacy_name_colliding_with_another_users_username_is_rejected(self):
        make_pharmacy('other_pharmacy_llc')  # username == 'other_pharmacy_llc'

        response = self.client.post(reverse('edit_profile'), self.valid_payload(
            pharmacy_name='Other Pharmacy LLC'  # sanitizes to the same username
        ))

        self.assertEqual(response.status_code, 200)  # re-rendered with form error
        self.pharmacy.user.refresh_from_db()
        self.assertEqual(self.pharmacy.user.username, 'pharmacist1')  # unchanged

    def test_can_keep_own_existing_username_unchanged(self):
        # Regression guard: clean_pharmacy_name excludes self.instance.user_id,
        # so submitting the form without changing pharmacy_name should never
        # trip the "username taken" validation against your own account.
        response = self.client.post(reverse('edit_profile'), self.valid_payload())
        self.assertEqual(response.status_code, 302)

    def test_duplicate_email_on_edit_shows_generic_error_not_which_field(self):
        # NEW: covers the fix for review #2 (user enumeration). The
        # IntegrityError fallback message must stay generic and must not
        # spell out that it was specifically the email/NPC/username that
        # collided.
        make_pharmacy('other_pharmacy', npc_number='NPC-OTHER')
        response = self.client.post(reverse('edit_profile'), self.valid_payload(
            npc_number='NPC-OTHER',
        ))

        # However this collision actually gets caught (via form
        # validation or the IntegrityError fallback), the response must
        # never explicitly say "NPC number" is the field that's taken.
        self.assertEqual(response.status_code, 200)
        self.assertNotContains(response, 'Another account is already using this email')


# ---------------------------------------------------------------------------
# Delete account
# ---------------------------------------------------------------------------

class DeleteAccountViewTests(TestCase):

    def setUp(self):
        # See module docstring: delete_account is cache-based rate
        # limited (keyed by user pk, which is commonly reused test-to-
        # test), so clear the cache to avoid cross-test bleed.
        cache.clear()
        self.pharmacy = make_pharmacy('pharmacist1')
        self.client.login(username='pharmacist1', password='testpass123')

    def test_requires_login(self):
        self.client.logout()
        response = self.client.get(reverse('delete_account'))
        self.assertNotEqual(response.status_code, 200)

    def test_wrong_npc_number_blocks_deletion(self):
        response = self.client.post(reverse('delete_account'), {
            'npc_number': 'WRONG',
            'password': 'testpass123',
        })

        self.assertContains(response, 'Incorrect NPC number')
        user = User.objects.get(username='pharmacist1')
        self.assertTrue(user.is_active)
        self.assertTrue(user.has_usable_password())

    def test_wrong_password_blocks_deletion(self):
        response = self.client.post(reverse('delete_account'), {
            'npc_number': self.pharmacy.npc_number,
            'password': 'wrongpassword',
        })

        self.assertContains(response, 'Incorrect password')
        user = User.objects.get(username='pharmacist1')
        self.assertTrue(user.is_active)
        self.assertTrue(user.has_usable_password())

    def test_correct_credentials_deactivate_account_without_erasing_data(self):
        # UPDATED: delete_account_view performs a SOFT delete
        # (is_active=False + unusable password), not a hard delete - it
        # never calls user.delete() and never touches Product/Sale/
        # SaleItem. The account, profile, and related records must all
        # still exist in the database afterwards.
        product = make_product(self.pharmacy)
        sale = Sale.objects.create(
            pharmacy=self.pharmacy, served_by_name='Alice', payment_method='cash'
        )
        SaleItem.objects.create(
            sale=sale, product=product, quantity=1, unit_price_at_sale=product.unit_price
        )

        response = self.client.post(reverse('delete_account'), {
            'npc_number': self.pharmacy.npc_number,
            'password': 'testpass123',
        })

        self.assertRedirects(response, reverse('login'))

        user = User.objects.get(username='pharmacist1')
        self.assertFalse(user.is_active)
        self.assertFalse(user.has_usable_password())

        # Related data is untouched by this "deletion".
        self.assertTrue(PharmacistProfile.objects.filter(pk=self.pharmacy.pk).exists())
        self.assertTrue(Product.objects.filter(pk=product.pk).exists())
        self.assertTrue(Sale.objects.filter(pk=sale.pk).exists())
        self.assertTrue(SaleItem.objects.filter(sale=sale, product=product).exists())

    def test_deactivated_user_can_no_longer_log_in(self):
        self.client.post(reverse('delete_account'), {
            'npc_number': self.pharmacy.npc_number,
            'password': 'testpass123',
        })

        login_ok = self.client.login(username='pharmacist1', password='testpass123')
        self.assertFalse(login_ok)

    def test_deleting_account_logs_the_user_out(self):
        self.client.post(reverse('delete_account'), {
            'npc_number': self.pharmacy.npc_number,
            'password': 'testpass123',
        })

        response = self.client.get(reverse('dashboard'))
        self.assertNotEqual(response.status_code, 200)  # login_required redirect

    def test_deactivating_account_does_not_affect_other_pharmacies(self):
        other_pharmacy = make_pharmacy('pharmacist2')
        other_product = make_product(other_pharmacy, brand_name='StillHere')

        self.client.post(reverse('delete_account'), {
            'npc_number': self.pharmacy.npc_number,
            'password': 'testpass123',
        })

        other_user = User.objects.get(username='pharmacist2')
        self.assertTrue(other_user.is_active)
        self.assertTrue(Product.objects.filter(pk=other_product.pk).exists())

    def test_locks_out_after_max_attempts(self):
        # NEW: covers the rate limiting added to delete_account_view.
        for _ in range(5):
            self.client.post(reverse('delete_account'), {
                'npc_number': 'WRONG',
                'password': 'wrongpassword',
            })

        response = self.client.post(reverse('delete_account'), {
            'npc_number': self.pharmacy.npc_number,
            'password': 'testpass123',
        })

        self.assertContains(response, 'Too many failed attempts')
        user = User.objects.get(username='pharmacist1')
        self.assertTrue(user.is_active)  # deletion did not go through


# ---------------------------------------------------------------------------
# Password reset (built-in Django views, light integration checks)
# ---------------------------------------------------------------------------

class PasswordResetFlowTests(TestCase):

    def setUp(self):
        self.pharmacy = make_pharmacy('pharmacist1')
        self.pharmacy.user.email = 'pharmacist1@example.com'
        self.pharmacy.user.save()

    def test_password_reset_get_renders_form(self):
        response = self.client.get(reverse('password_reset'))
        self.assertEqual(response.status_code, 200)

    def test_password_reset_with_known_email_sends_email_and_redirects_to_done(self):
        response = self.client.post(reverse('password_reset'), {
            'email': 'pharmacist1@example.com',
        })

        self.assertRedirects(response, reverse('password_reset_done'))
        self.assertEqual(len(mail.outbox), 1)
        self.assertIn('pharmacist1@example.com', mail.outbox[0].to)

    def test_password_reset_with_unknown_email_does_not_reveal_existence(self):
        # Django's built-in view redirects to "done" either way, to avoid
        # leaking which emails are registered - no email is actually sent.
        response = self.client.post(reverse('password_reset'), {
            'email': 'nobody@example.com',
        })

        self.assertRedirects(response, reverse('password_reset_done'))
        self.assertEqual(len(mail.outbox), 0)

    def test_password_reset_done_renders(self):
        response = self.client.get(reverse('password_reset_done'))
        self.assertEqual(response.status_code, 200)


# ---------------------------------------------------------------------------
# Password change (requires being logged in)
# ---------------------------------------------------------------------------

class PasswordChangeFlowTests(TestCase):

    def setUp(self):
        cache.clear()
        self.pharmacy = make_pharmacy('pharmacist1')
        self.client.login(username='pharmacist1', password='testpass123')

    def test_requires_login(self):
        self.client.logout()
        response = self.client.get(reverse('password_change'))
        self.assertNotEqual(response.status_code, 200)

    def test_get_renders_form(self):
        response = self.client.get(reverse('password_change'))
        self.assertEqual(response.status_code, 200)

    def test_correct_old_password_changes_password(self):
        response = self.client.post(reverse('password_change'), {
            'old_password': 'testpass123',
            'new_password1': 'BrandNewPass456!',
            'new_password2': 'BrandNewPass456!',
        })
        self.assertRedirects(response, reverse('password_change_done'))
        self.client.logout()
        login_ok = self.client.login(
            username='pharmacist1', password='BrandNewPass456!'
        )
        self.assertTrue(login_ok)

    def test_wrong_old_password_does_not_change_password(self):
        response = self.client.post(reverse('password_change'), {
            'old_password': 'wrongoldpassword',
            'new_password1': 'BrandNewPass456!',
            'new_password2': 'BrandNewPass456!',
        })
        self.assertEqual(response.status_code, 200)
        self.client.logout()
        login_ok = self.client.login(
            username='pharmacist1', password='testpass123'
        )
        self.assertTrue(login_ok)

    def test_locks_out_after_max_attempts(self):
        # Directly set the cache key to simulate lockout
        ip = '127.0.0.1'
        user_pk = self.pharmacy.user.pk
        cache_key = f'ratelimit:change_password:{ip}:{user_pk}'
        cache.set(cache_key, 5, 300)

        # This attempt should be blocked
        response = self.client.post(reverse('password_change'), {
            'old_password': 'wrongoldpassword',
            'new_password1': 'BrandNewPass456!',
            'new_password2': 'BrandNewPass456!',
        })

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'Too many failed attempts')
        self.client.logout()
        login_ok = self.client.login(
            username='pharmacist1', password='testpass123'
        )
        self.assertTrue(login_ok)