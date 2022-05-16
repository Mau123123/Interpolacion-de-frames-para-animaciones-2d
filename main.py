import sys, os
import cv2
import torchvision.transforms as transforms
import numpy as np
import procesadorImagenVideo as PIV
from PyQt5.QtWidgets import QApplication, QMainWindow, QListWidget, QLabel, QPushButton
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt
from PIL import Image
from PIL.Image import open
from model.interpolacion import interpolacion

class ListboxWidget(QListWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        
    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls:
            event.accept()
        else:
            event.ignore()
    
    def dragMoveEvent(self, event):
        if event.mimeData().hasUrls:
            event.setDropAction(Qt.CopyAction)
            event.accept()
        else:
            event.ignore()
            
    def dropEvent(self, event):
        if event.mimeData().hasUrls:
            event.setDropAction(Qt.CopyAction)
            event.accept()
            links = []
            for url in event.mimeData().urls():
                if url.isLocalFile():
                    links.append(str(url.toLocalFile()))
                else:
                    links.append(str(url.toString()))
            if(len(links)>1):
                return
            self.clear()
            self.addItems(links)
            self.link = links[0]
        else:
            event.ignore()
            


class Interfaz(QMainWindow):
    def __init__(self):
        super().__init__()
        self.device = "cuda"
        self.interpolacion = interpolacion()
        self.transform =  transforms.ToTensor()
        self.contador = 0
        self.fps = 0
        self.imagenes = []
        self.resize(1500, 600)
        self.lst = ListboxWidget(self)
        self.lst.setGeometry(50,50,200,100)
        self.btn1 = QPushButton('procesar video', self)
        self.btn1.setGeometry(50,200,200,25)
        self.btn1.clicked.connect(lambda : self.procesarVideo(self.lst.link))
        self.btn1 = QPushButton('generar video', self)
        self.btn1.setGeometry(50,250,200,25)
        self.btn1.clicked.connect(lambda : self.generarVideo())
        self.btn2 = QPushButton('siguiente', self)
        self.btn2.setGeometry(50,300,200,25)
        self.btn2.clicked.connect(lambda : self.siguiente())
        self.btn4 = QPushButton('anterior', self)
        self.btn4.setGeometry(50,350,200,25)
        self.btn4.clicked.connect(lambda : self.anterior())
        self.btn5 = QPushButton('reducir fps', self)
        self.btn5.setGeometry(50,350,200,25)
        self.btn5.clicked.connect(lambda : self.reducir())   
        self.img1 = QLabel(self)
        self.img1.setGeometry(300,10,600,580)
        self.img2 = QLabel(self)
        self.img2.setGeometry(950,10,600,580)
        self.imgProcesada = QLabel(self)
        self.labelContador = QLabel(str(self.contador),self)
        self.labelContador.setGeometry(1000,665,25,25)
    
    
    def siguiente(self):
        if(self.contador<len(self.imagenes)-5):
            self.contador= self.contador+ 4
            self.pixmap = QPixmap(f'./imagenes/{self.imagenes[self.contador]}')
            self.img1.setPixmap(self.pixmap)
            self.pixmap2 = QPixmap(f'./imagenes/{self.imagenes[self.contador+1]}')
            self.img2.setPixmap(self.pixmap2)
            self.labelContador.setText(str(self.contador))
            self.procesarImagen()
            
       
    def anterior(self):
        if(self.contador>0):
            self.contador=self.contador - 1
            self.pixmap = QPixmap(f'./imagenes/{self.imagenes[self.contador]}')
            self.img1.setPixmap(self.pixmap)
            self.pixmap2 = QPixmap(f'./imagenes/{self.imagenes[self.contador+1]}')
            self.img2.setPixmap(self.pixmap2)
            self.labelContador.setText(str(self.contador))
            self.procesarImagen()
         
    def procesarVideo(self, dir):
        piv = PIV.procesadorImagenVideo()
        self.imagenes, self.fps = piv.ob_img(dir)
        self.pixmap = QPixmap(f'./imagenes/{self.imagenes[0]}')
        self.img1.setPixmap(self.pixmap)
        self.pixmap2 = QPixmap(f'./imagenes/{self.imagenes[1]}')
        self.img2.setPixmap(self.pixmap2)
        self.procesarImagen()
        
    def generarVideo(self):
        print ('start')
        i = 0
        newVideo = []
        height = 0
        width = 0
        os.chdir('imagenes')
        while i < len(self.imagenes)-1:
            image1 = cv2.imread(f'./{self.imagenes[i]}', cv2.IMREAD_UNCHANGED)
            image2 = cv2.imread(f'./{self.imagenes[i + 1]}', cv2.IMREAD_UNCHANGED)
            img1 = self.load_image(f'./{self.imagenes[i]}')
            img2 = self.load_image(f'./{self.imagenes[i + 1]}')
            newFrame = self.interpolacion.forward(img1, img2)*255
            height = newFrame.shape[0]
            width = newFrame.shape[1]
            imgname = "Newframe%d.jpg" % i
            cv2.imwrite(imgname, newFrame)
            newFrame = cv2.imread(imgname, cv2.IMREAD_UNCHANGED)
            newVideo.append(image1)
            newVideo.append(newFrame)
            print(i)
            i += 1
        newVideo.append(cv2.imread(f'./{self.imagenes[-1]}', cv2.IMREAD_UNCHANGED))
        os.chdir('..')
        out = cv2.VideoWriter('generated.mp4v',cv2.VideoWriter_fourcc(*'DIVX'), self.fps*2, (width,height))
        for i in range(len(newVideo)):
            out.write(newVideo[i])
        out.release()
        print('finish')
        
    def reducir(self):
        print ('start')
        i = 0
        newVideo = []
        os.chdir('imagenes')
        while i < len(self.imagenes)-1:
            if i%2 == 0:
                image1 = cv2.imread(f'./{self.imagenes[i]}', cv2.IMREAD_UNCHANGED)
                height, width, layers = image1.shape
                size = (width,height)
                newVideo.append(image1)
            i += 1
        newVideo.append(cv2.imread(f'./{self.imagenes[-1]}', cv2.IMREAD_UNCHANGED))
        os.chdir('..')
        out = cv2.VideoWriter('project.mp4v',cv2.VideoWriter_fourcc(*'DIVX'), self.fps/2, size)
        for i in range(len(newVideo)):
            out.write(newVideo[i])
            print(i)
        out.release()
        print('finish')
          
    def load_image(self,imfile):
        return self.interpolacion.loadImage(imfile)
         
    def procesarImagen(self):
        self.image = cv2.imread(f'./imagenes/{self.imagenes[self.contador]}', cv2.IMREAD_UNCHANGED)
        self.nextImage = cv2.imread(f'./imagenes/{self.imagenes[self.contador + 1]}', cv2.IMREAD_UNCHANGED)
        image1 = self.load_image(f'./imagenes/{self.imagenes[self.contador]}')
        image2 = self.load_image(f'./imagenes/{self.imagenes[self.contador + 1]}')
        self.mostrarImagen()
        newFrame = self.interpolacion.forward(image1, image2)
        cv2.imshow("newFrame", newFrame)
        
        
    def mostrarImagen(self):
        size = self.image.shape
        step = int(self.image.size / size[0])
        qformat = QImage.Format_Indexed8
        if len(size) == 3:
            if size[2] == 4:
                qformat = QImage.Format_RGBA8888
            else:
                qformat = QImage.Format_RGB888
        img = QImage(self.image, size[1], size[0], step, qformat)
        img = img.rgbSwapped()
        self.imgProcesada.setPixmap(QPixmap.fromImage(img))
        
        
app = QApplication(sys.argv)

interfaz = Interfaz()
interfaz.show()

sys.exit(app.exec_())