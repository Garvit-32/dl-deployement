import os
import torch
import numpy as np
from PIL import Image
from .utility import runner
from .preload import preloader


def process(img="download.jpg"):
    current_path = os.path.dirname(os.path.abspath(__file__))

    with torch.no_grad():
        # for django integration
        # im = Image.open(
        #     img
        # )
        # for separate use
        model = preloader(
            'deployApp/numplatedetect/weights/hrnetv2_hrnet_plate_199.pth')
        im = Image.open(
            img
        )
        frame = np.array(im)[:, :, :3]
        array_image, data_dictionary = runner(
            frame, model, 'deployApp/numplatedetect/weights/iter2.pth')
        final_image = Image.fromarray(array_image, "RGB")
        # print(data_dictionary)
        # print(json_object)

        # filePathName = None
        # fs = FileSystemStorage()
        # filePathName = fs.save('result.jpg', final_image)
        # filePathName = fs.url(filePathName)
        # print('='*40)
        # print(type(final_image))
        # print('='*40)

        # print('='*40)
        # print(in_mem_file)
        # print('='*40)
        # file = default_storage.open(f'Output/{img.name}', 'w')
        # file.write(in_mem_file)
        # file.close()

        # image_file = InMemoryUploadedFile(
        #     val, None, 'foo.jp  g', 'image/jpeg', val.tell, None)

        # fs = default_storage.save(
        #     f'output_img/{randomString+pathlib.Path(img.name).suffix }', in_mem_file)

        # in_mem_file = ""
        # print(fs,"path of output")
        # imgUrl = settings.MEDIA_ROOT + fs
        # settings.MEDIA_ROOT
        # final_image.save(f"media/OUT/{img.name}") #for django integration
        # final_image.save(f"{current_path}/{img}")  # for Separate Use
    # return data_dictionary , img.name

        # print(imgUrl)
    return data_dictionary, final_image
