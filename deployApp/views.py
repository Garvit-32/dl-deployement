import os
from io import BytesIO
from django.shortcuts import render
from django.core.files.base import ContentFile
from deployApp.numplatedetect.main import process
from django.core.files.storage import FileSystemStorage
from django.utils.datastructures import MultiValueDictKeyError
from django.core.files.uploadedfile import InMemoryUploadedFile
# Create your views here.


def remove():
    if os.path.exists('result/input.png'):
        os.remove("result/input.png")

    if os.path.exists('result/output.png'):
        os.remove("result/output.png")


def index(request):
    context = {'a': 1}
    return render(request, 'deployApp/index.html', context)


def predictImage(request):

    remove()
    context = {}

    try:
        fileObj = request.FILES['filePath']
    except MultiValueDictKeyError:
        context['error_message'] = "Please select a image"
        return render(request, 'deployApp/index.html', context)

    fs = FileSystemStorage()
    filePathName = fs.save('result/input.png', fileObj)
    filePathName = fs.url(filePathName)

    data_dictionary, final_image = process(img='result/input.png')

    in_mem_file = BytesIO()
    final_image.save(in_mem_file, format='PNG')
    val = ContentFile(in_mem_file.getvalue())

    image_file = InMemoryUploadedFile(
        val, None, 'foo.jpeg', 'image/jpeg', val.tell, None)

    outputName = fs.save('result/output.png', image_file)
    output_url = fs.url(outputName)

    context['filePathName'] = output_url
    context['imgArray'] = data_dictionary
    return render(request, 'deployApp/index.html', context)
