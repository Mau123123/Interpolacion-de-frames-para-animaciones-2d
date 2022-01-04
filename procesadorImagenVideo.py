import cv2
import os
import shutil

class procesadorImagenVideo():
    def __init__(self):
        super().__init__()
    
    def ob_img(self,video_dir):
        imagenes = []
        _,archivo = os.path.split(video_dir)
        vidcap = cv2.VideoCapture(video_dir)
        fps = vidcap.get(cv2.CAP_PROP_FPS)
        print(fps)
        success,image = vidcap.read()
        dim = (400, 300)
        count = 0
        if os.path.exists("imagenes"):
            shutil.rmtree("./imagenes")
        os.mkdir("imagenes")
        os.chdir("./imagenes")
        while success:
            imageRed = cv2.resize(image,dim, interpolation = cv2.INTER_AREA)
            imgname = "frame%d.jpg" % count
            cv2.imwrite(imgname, imageRed)
            imagenes.append(imgname)
            success,image = vidcap.read()
            count += 1
        path_parent = os.path.dirname(os.getcwd())
        os.chdir(path_parent)
        return imagenes, fps