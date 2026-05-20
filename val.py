import warnings
warnings.filterwarnings('ignore')
from ultralytics import RTDETR



if __name__ == '__main__':
    model = RTDETR('D:/RTDETR-main/runs/train/rtdetr-r18/weights/best.pt')
    model.val(data='D:/RTDETR-main/ultralytics/cfg/SeaDronesSee.yaml',
              split='train', 
              imgsz=640,
              batch=4,
            #   save_json=True, # if you need to cal coco metrice
              project='runs/val',
              name='final',
              )
