import warnings, os
# os.environ["CUDA_VISIBLE_DEVICES"]="-1"   
# os.environ["CUDA_VISIBLE_DEVICES"]="0"     

warnings.filterwarnings('ignore')
from ultralytics import RTDETR

if __name__ == '__main__':
    model = RTDETR('ultralytics/cfg/models/rt-detr/rtdetr-SPE+SFConv.yaml')
    # model.load('') # loading pretrain weights
    model.train(data='D:/RTDETR-main/ultralytics/cfg/SeaDronesSee.yaml',
                cache=False,
                imgsz=640,
                epochs=300,
                batch=16,
                workers=4, 
                # device='0,1', 
                # resume='', # last.pt path
                project='runs/train',
                name='r18'
    )
