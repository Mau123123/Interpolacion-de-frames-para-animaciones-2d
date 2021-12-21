import cv2
import os
import shutil

vid = "simpson2"

def ob_img(video_dir, dim):
    print(video_dir)
    vidcap = cv2.VideoCapture(video_dir)
    success,image = vidcap.read()
    count = 100000
    os.chdir(f"./dataset/{vid}")
    while success:
        if dim: 
            image = cv2.resize(image,dim, interpolation = cv2.INTER_AREA)
        imgname = "%d.jpg" % count
        cv2.imwrite(imgname, image)
        success,image = vidcap.read()
        count += 1
        
if os.path.exists(f"./dataset/{vid}"):
    shutil.rmtree(f"./dataset/{vid}")
os.mkdir(f"./dataset/{vid}")

ob_img(f"{vid}.mp4", (400, 300))