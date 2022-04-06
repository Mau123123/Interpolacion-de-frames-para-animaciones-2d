import sys, os
import cv2
import torch
import torch.cuda as cuda
from torchvision import transforms
import numpy as np
import argparse
import procesadorImagenVideo as PIV
from PyQt5.QtWidgets import QApplication, QMainWindow, QListWidget, QLabel, QPushButton
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt
from PIL import Image
from GMA.core.network import RAFTGMA
from GMA.core.utils.utils import InputPadder
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
        self.args = self.loadArgs()
        self.device = "cuda" if cuda.is_available() else "cpu"
        self.model = UNET(in_channels=6, out_channels=3).to(self.device)
        self.model.load_state_dict(torch.load('model/weights.pth'))
        
        self.flow = torch.nn.DataParallel(RAFTGMA(self.args))
        self.flow.load_state_dict(torch.load(self.args.model))
        self.flow = self.flow.module
        self.flow.to(self.device)
        self.flow.eval()

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
            
    def loadArgs(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--model', help="restore checkpoint", default="GMA/checkpoints/gma-sintel.pth")
        parser.add_argument('--model_name', help="define model name", default="GMA")
        parser.add_argument('--path', help="dataset for evaluation")
        parser.add_argument('--num_heads', default=1, type=int,
                            help='number of heads in attention and aggregation')
        parser.add_argument('--position_only', default=False, action='store_true',
                            help='only use position-wise attention')
        parser.add_argument('--position_and_content', default=False, action='store_true',
                            help='use position and content-wise attention')
        parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
        return parser.parse_args()
        
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
            image1 = cv2.imread(f'./{self.imagenes[i]}', cv2.IMREAD_UNCHANGED)
            image2 = cv2.imread(f'./{self.imagenes[i + 1]}', cv2.IMREAD_UNCHANGED)
            img1 = self.load_image(f'./{self.imagenes[i]}')
            img2 = self.load_image(f'./{self.imagenes[i + 1]}')
            height, width, layers = img1[0].shape
            size = (width,height)
            f1 = self.warpImage(image1, img1, img2)
            f2 = self.warpImage(image2, img2, img1)
            newFrame = self.calcUnet(f1,f2)*255
            newFrame = cv2.resize(newFrame, (400,300), interpolation = cv2.INTER_AREA)
            imgname = "Newframe%d.jpg" % i
            cv2.imwrite(imgname, newFrame)
            newFrame = cv2.imread(imgname, cv2.IMREAD_UNCHANGED)
            newVideo.append(image1)
            newVideo.append(newFrame)
            print(newFrame.shape)
            print(image1.shape)
            print(i)
            i += 1
        newVideo.append(cv2.imread(f'./{self.imagenes[-1]}', cv2.IMREAD_UNCHANGED))
        os.chdir('..')
        print(len(newVideo))
        out = cv2.VideoWriter('generated.avi',cv2.VideoWriter_fourcc(*'DIVX'), self.fps*2, (400,300))
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
        print(len(newVideo))
        out = cv2.VideoWriter('project.mp4v',cv2.VideoWriter_fourcc(*'DIVX'), self.fps/2, size)
        for i in range(len(newVideo)):
            out.write(newVideo[i])
            print(i)
        out.release()
        print('finish')
        
        
    def calcUnet(self, image, nextImage):
        convert_tensor = transforms.ToTensor()
        T1 = torch.unsqueeze(convert_tensor(image).to(self.device),0)
        T2 = torch.unsqueeze(convert_tensor(nextImage).to(self.device),0)
        tensor = torch.unsqueeze(torch.cat((T1,T2),0),0)
        tensor_image = self.model(T1,T2).cpu().detach().numpy()[0]
        tensor_image = np.transpose(tensor_image, (1,2,0))    
        return tensor_image
    
    def calcEdges(self, image, nextImage):
        convert_tensor = transforms.ToTensor()
        T1 = torch.unsqueeze(convert_tensor(image).type(torch.FloatTensor).to(self.device),0)
        T2 = torch.unsqueeze(convert_tensor(nextImage).type(torch.FloatTensor).to(self.device),0)
        tensor = torch.unsqueeze(torch.cat((T1,T2),0),0)
        tensor_image = self.modelEdges(T1,T2).cpu().detach().numpy()[0]
        tensor_image = np.transpose(tensor_image, (1,2,0))    
        return tensor_image*255
     
    def warpImage(self, image, img1, img2):
        print(img1.dim)
        padder = InputPadder(img1.shape)
        img1, img2 = padder.pad(img1, img2)
        flow_low, flow_up = self.flow(img1, img2, iters=16, test_mode=True)
        flow_up = flow_up[0].permute(1, 2, 0).cpu().detach().numpy()
        h, w = flow_up.shape[:2]
        flow = -flow_up*0.5
        flow[:,:,0] += np.arange(w)
        flow[:,:,1] += np.arange(h)[:,np.newaxis]
        warped = cv2.remap(image, flow, None, cv2.INTER_LINEAR)
        return warped
    
    def load_image(self,imfile):
        img = np.array(Image.open(imfile)).astype(np.uint8)
        img = torch.from_numpy(img).permute(2, 0, 1).float()
        return img[None].to(self.device)
         
    def procesarImagen(self):
        self.image = cv2.imread(f'./imagenes/{self.imagenes[self.contador]}', cv2.IMREAD_UNCHANGED)
        self.nextImage = cv2.imread(f'./imagenes/{self.imagenes[self.contador + 1]}', cv2.IMREAD_UNCHANGED)
        image1 = self.load_image(f'./imagenes/{self.imagenes[self.contador]}')
        image2 = self.load_image(f'./imagenes/{self.imagenes[self.contador + 1]}')
        self.mostrarImagen()
        f1 = self.warpImage(self.image,image1, image2)
        f2 = self.warpImage(self.nextImage,image2, image1)
        tensor_image = self.calcUnet(f1, f2)
        cv2.imshow("tensor_image", tensor_image)
        cv2.imshow("f1", f1)
        cv2.imshow("f2", f2)
        
        
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