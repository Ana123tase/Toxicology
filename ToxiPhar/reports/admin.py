from django.contrib import admin

from .models import PredictionReport


@admin.register(PredictionReport)
class PredictionReportAdmin(admin.ModelAdmin):
    list_display = ("brand_name", "generic_name", "user", "created_at")
    list_filter = ("created_at",)
    search_fields = ("brand_name", "generic_name", "chemical_name", "smiles", "user__username")
    readonly_fields = ("created_at",)
    date_hierarchy = "created_at"