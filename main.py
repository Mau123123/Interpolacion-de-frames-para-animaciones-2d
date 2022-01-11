import sys, os
import cv2
import torch
import torch.cuda as cuda
from torchvision import transforms
import numpy as np
import procesadorImagenVideo as PIV
from PyQt5.QtWidgets import QApplication, QMainWindow, QListWidget, QLabel, QPushButton
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt
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
        self.device = "cuda" if cuda.is_available() else "cpu"
        self.model = UNET(in_channels=6, out_channels=3).to(self.device)
        self.modelEdges = UNET(in_channels=2, out_channels=1).to(self.device)
        self.model.load_state_dict(torch.load('model/weights.pth'))
        self.contador = 0
        self.fps = 0
        self.imagenes = []
        self.resize(1725, 700)
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
        os.chdir('imagenes')
        while i < len(self.imagenes)-1:
            img1 = cv2.imread(f'./{self.imagenes[i]}', cv2.IMREAD_UNCHANGED)
            img2 = cv2.imread(f'./{self.imagenes[i + 1]}', cv2.IMREAD_UNCHANGED)
            height, width, layers = img1.shape
            size = (width,height)
            f1 = self.warpImage(img1, img2, layers)
            f2 = self.warpImage(img2, img1, layers)
            newFrame = self.calcUnet(f1,f2)
            imgname = "Newframe%d.jpg" % i
            cv2.imwrite(imgname, newFrame)
            newFrame = cv2.imread(imgname, cv2.IMREAD_UNCHANGED)
            newVideo.append(img1)
            newVideo.append(newFrame)
            print(i)
            i += 1
        newVideo.append(cv2.imread(f'./{self.imagenes[-1]}', cv2.IMREAD_UNCHANGED))
        os.chdir('..')
        print(len(newVideo))
        out = cv2.VideoWriter('project.avi',cv2.VideoWriter_fourcc(*'DIVX'), self.fps*2, size)
        for i in range(len(newVideo)):
            out.write(newVideo[i])
        out.release()
        print('finish')
        
        
    def calcUnet(self, image, nextImage):
        convert_tensor = transforms.ToTensor()
        T1 = convert_tensor(image).to(self.device)
        T2 = convert_tensor(nextImage).to(self.device)
        tensor = torch.unsqueeze(torch.cat((T1,T2),0),0)
        tensor_image = self.model(tensor).cpu().detach().numpy()[0]
        tensor_image = np.transpose(tensor_image, (1,2,0))    
        return tensor_image*255
    
    def calcEdges(self, image, nextImage):
        convert_tensor = transforms.ToTensor()
        T1 = convert_tensor(image).type(torch.FloatTensor).to(self.device)
        T2 = convert_tensor(nextImage).type(torch.FloatTensor).to(self.device)
        tensor = torch.unsqueeze(torch.cat((T1,T2),0),0)
        tensor_image = self.modelEdges(tensor).cpu().detach().numpy()[0]
        tensor_image = np.transpose(tensor_image, (1,2,0))    
        return tensor_image*255
     
    def warpImage(self, img1, img2, channels):
        if channels > 1:
            img1gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
            img2gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
            flow = cv2.calcOpticalFlowFarneback(img1gray, img2gray,cv2.CV_16S, 0.2, 5, 2, 3, 5, 1.1, 0)
        else:
            flow = cv2.calcOpticalFlowFarneback(img1, img2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            img1 = img1
            img2 = img2
        h, w = flow.shape[:2]
        flow = -flow*0.5
        flow[:,:,0] += np.arange(w)
        flow[:,:,1] += np.arange(h)[:,np.newaxis]
        warped = cv2.remap(img1, flow, None, cv2.INTER_LINEAR)
        return warped
         
    def procesarImagen(self):
        self.image = cv2.imread(f'./imagenes/{self.imagenes[self.contador]}', cv2.IMREAD_UNCHANGED)
        self.nextImage = cv2.imread(f'./imagenes/{self.imagenes[self.contador + 1]}', cv2.IMREAD_UNCHANGED)
        self.mostrarImagen()
        img_gray1 = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        img_gray2 = cv2.cvtColor(self.nextImage, cv2.COLOR_BGR2GRAY)
        img_gray1 = cv2.adaptiveThreshold(img_gray1,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
        img_gray2 = cv2.adaptiveThreshold(img_gray2,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
        img_gray1 = np.amax(img_gray1) - img_gray1 
        img_gray2 = np.amax(img_gray2) - img_gray2 
        b1 = self.warpImage(img_gray1, img_gray2, 1)
        b2 = self.warpImage(img_gray2, img_gray1, 1)
        cv2.imshow("edge", b1)
        edge_image = self.calcEdges(b1, b2)/255
        f1 = self.warpImage(self.image, self.nextImage, 3)
        f2 = self.warpImage(self.nextImage, self.image, 3)
        tensor_image = self.calcUnet(f1, f2)/255
        cv2.imshow("tensor_image", tensor_image)
        cv2.imshow("edge_image", edge_image)
        
        
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