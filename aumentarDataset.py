import cv2
import os
import shutil

vid = "simpson2"

def ob_img(video_dir, name, dim):
    print(video_dir)
    vidcap = cv2.VideoCapture(video_dir)
    success,image = vidcap.read()
    count = 100000
    if os.path.exists(f"./dataset/{name}"):
        shutil.rmtree(f"./dataset/{name}")
    os.mkdir(f"./dataset/{name}")
    os.chdir(f"./dataset/{name}")
    while success:
        if dim: 
            image = cv2.resize(image,dim, interpolation = cv2.INTER_AREA)
        imgname = "%d.jpg" % count
        cv2.imwrite(imgname, image)
        success,image = vidcap.read()
        count += 1

ob_img(f"{vid}.mp4",vid, (400, 300))