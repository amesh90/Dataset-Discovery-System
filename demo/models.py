from django.db import models

# Create your models here.

class Search(models.Model):
	keywords = models.CharField(max_length=200)
