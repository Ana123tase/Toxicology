from django.contrib import admin
from .models import PharmacistProfile

@admin.register(PharmacistProfile)
class PharmacistProfileAdmin(admin.ModelAdmin):
    list_display = ('pharmacy_name', 'user', 'npc_number', 'district', 'sector', 'created_at')
    search_fields = ('pharmacy_name', 'npc_number', 'user__username')