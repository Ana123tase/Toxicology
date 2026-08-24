from django.contrib import admin
from .models import PharmacistProfile

@admin.register(PharmacistProfile)
class PharmacistProfileAdmin(admin.ModelAdmin):
    list_display = ('institution', 'user',  'district', 'sector', 'created_at')
    search_fields = ('institution', 'user__username')