from django.db import models

class Photo(models.Model):
    name = models.CharField(max_length=250)
    img = models.ImageField(upload_to='media/')