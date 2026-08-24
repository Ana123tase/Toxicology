from django.db import models
from django.contrib.auth.models import User

class PharmacistProfile(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE, related_name='profile')
    
    # Pharmacy details
    institution = models.CharField(max_length=150)
    phone_number = models.CharField(max_length=20)
    
    # Location
    country = models.CharField(max_length=100)
    district = models.CharField(max_length=100)
    sector = models.CharField(max_length=100)
    
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.institution} ({self.user.username})"