from django.conf import settings
from django.db import models


class PredictionReport(models.Model):
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="prediction_reports",
        null=True,
        blank=True,
    )
    brand_name = models.CharField(max_length=500)
    generic_name = models.CharField(max_length=500, blank=True)
    chemical_name = models.CharField(max_length=3000, blank=True)
    smiles = models.TextField()
    result = models.JSONField()
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self):
        return f"{self.brand_name} ({self.created_at:%Y-%m-%d %H:%M})"