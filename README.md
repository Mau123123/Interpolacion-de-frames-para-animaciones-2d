# Interpolacion-de-frames-para-animaciones-2d

requisitos:
    python3
    cv2
    PyQt5
    shutil
    torch 1.10.1
    numpy
    cuda 11.5

conda activate C:\Users\Mau\Desktop\proyectos\Proyecto\projectDependecies

 instalacion:
    descargar las librerias:
        cv2: pip install cv2
        PyQt5: pip install PyQt5
        shutil: pip install shutil
        torch: pip3 install torch==1.10.1+cu113 torchvision==0.11.2+cu113 torchaudio===0.10.1+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
        numpy: pip install numpy
    descargar los pesos ya entrenados del drive y copiarlos en la carpeta model
    
    la instalacion de cuda es opcional ya que se puede ejecutar el programa solamente en el cpu

 ejecucion:
 python3 main.py para el programa principal que genera el video
 para el entrenamiento se puede ejecutar el train.ipynb en jupiter notebook para ver su ejecucuion o 
 train.py para hacer el entrenamiento automaticamente