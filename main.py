import sys, os
import cv2
import torch
import matplotlib.pyplot as plt
from torchvision import transforms
import numpy as np
import procesadorImagenVideo as PIV
from PyQt5.QtWidgets import QApplication, QMainWindow, QListWidget, QLabel, QPushButton
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt, QUrl
from model.UNET import UNET 

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
        self.model = UNET(in_channels=6, out_channels=3)
        self.contador = 0
        self.imagenes = []
        self.resize(1725, 700)
        self.lst = ListboxWidget(self)
        self.lst.setGeometry(50,50,200,100)
        
        self.btn1 = QPushButton('procesar video', self)
        self.btn1.setGeometry(50,200,200,25)
        self.btn1.clicked.connect(lambda : self.procesarVideo(self.lst.link))
        
        self.btn1 = QPushButton('procesar imagen', self)
        self.btn1.setGeometry(50,250,200,25)
        self.btn1.clicked.connect(lambda : self.procesarImagen())
              
        self.btn2 = QPushButton('siguiente', self)
        self.btn2.setGeometry(50,300,200,25)
        self.btn2.clicked.connect(lambda : self.siguiente())
        
        self.btn4 = QPushButton('anterior', self)
        self.btn4.setGeometry(50,350,200,25)
        self.btn4.clicked.connect(lambda : self.anterior())
                
        self.img1 = QLabel(self)
        self.img1.setGeometry(300,25,700,600)
        self.img2 = QLabel(self)
        self.img2.setGeometry(1000,25,700,600)
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
        self.imagenes = piv.ob_img(dir)
        self.pixmap = QPixmap(f'./imagenes/{self.imagenes[0]}')
        self.img1.setPixmap(self.pixmap)
        self.pixmap2 = QPixmap(f'./imagenes/{self.imagenes[1]}')
        self.img2.setPixmap(self.pixmap2)
        self.procesarImagen()
        
    def procesarImagen(self):
        self.image = cv2.imread(f'./imagenes/{self.imagenes[self.contador]}', cv2.IMREAD_UNCHANGED)
        self.nextImage = cv2.imread(f'./imagenes/{self.imagenes[self.contador + 1]}', cv2.IMREAD_UNCHANGED)
        
        #calculo en u-net
        convert_tensor = transforms.ToTensor()
        T1 = convert_tensor(self.image)
        T2 = convert_tensor(self.nextImage)
        tensor = torch.unsqueeze(torch.cat((T1,T2),0),0)
        tensor_image = self.model(tensor).detach().numpy()[0]
        print(tensor_image.shape)
        tensor_image = np.transpose(tensor_image, (1,2,0))*255
        cv2.imshow("tensor_image", tensor_image)

        
        self.mostrarImagen()
        img1 = cv2.GaussianBlur(self.image, (3, 3), 0)
        img2 = cv2.GaussianBlur(self.nextImage, (3, 3), 0)
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        img1 = cv2.Laplacian(img1, cv2.CV_16S, ksize=1)
        img1 = cv2.convertScaleAbs(img1)
        img2 = cv2.Laplacian(img2, cv2.CV_16S, ksize=1)
        img2 = cv2.convertScaleAbs(img2)
        cv2.imshow("bordes img1", img1)
        cv2.imshow("bordes img2", img2)
        mask12 = np.zeros_like(self.image)
        mask21 = np.zeros_like(self.image)
        flow12 = cv2.calcOpticalFlowFarneback(img1, img2,cv2.CV_16S, 0.2, 5, 2, 3, 5, 1.1, 0)
        flow21 = cv2.calcOpticalFlowFarneback(img2, img1,cv2.CV_16S, 0.2, 5, 2, 3, 5, 1.1, 0)
        magnitude12, angle12 = cv2.cartToPolar(flow12[..., 0], flow12[..., 1])
        magnitude21, angle21 = cv2.cartToPolar(flow21[..., 0], flow21[..., 1])
        mask12[..., 0] = angle12 * 180 / np.pi / 2
        mask12[..., 2] = cv2.normalize(magnitude12, None, 0, 255, cv2.NORM_MINMAX)
        rgb12 = cv2.cvtColor(mask12, cv2.COLOR_HSV2BGR)
        mask21[..., 0] = angle21 * 180 / np.pi / 2
        mask21[..., 2] = cv2.normalize(magnitude21, None, 0, 255, cv2.NORM_MINMAX)
        rgb21 = cv2.cvtColor(mask21, cv2.COLOR_HSV2BGR)
        cv2.imshow("B0-B1", rgb12)
        cv2.imshow("B1-B0", rgb21)
        
    def mostrarImagen(self):
        size = self.image.shape
        step = self.image.size / size[0]
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