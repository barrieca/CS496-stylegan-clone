from django.http import HttpResponse
from django.shortcuts import render
import time
import os
import subprocess

def index(request):
    if request.method == "POST":
        os.system('rm ../raw_images/*')
        file = request.FILES["image_file"]
        with open('../raw_images/' + file.name, 'wb+') as destination:
            for chunk in file.chunks():
                destination.write(chunk)
        os.chdir('../')
        temp = os.getcwd()
        #if not os.listdir('aligned_images'):
        os.system('rm aligned_images/*.png')
        os.system('/bin/bash -c "python align_images.py raw_images aligned_images"')
        #if not os.listdir('latent_codes'):
        os.system('rm latent_codes/*.npy')
        os.system('python encode_images.py aligned_images generated_images latent_codes')
        base_file_name = file.name.split('.')[0]
        style_mix_file_name = base_file_name + '_style_mix.png'
        os.system('python draw_images.py ' + base_file_name + '_01.npy ' + style_mix_file_name)
        os.system('mv results/' + style_mix_file_name + ' website/media/')
        os.chdir('website')
    else:
        style_mix_file_name = None

    return render(request, 'index.html', {'file_name': style_mix_file_name})