import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('ultralytics/cfg/models/v10/yolov10n.yaml')
    # model.load('yolo11n.pt') # loading pretrain weights
    model.train(data='D:/Desktop/ultralyticsPro0214-YOLO11/data.yaml',
                cache=False,
                imgsz=640,
                epochs=300,
                batch=16,
                close_mosaic=10,
                workers=2, # Windows下出现莫名其妙卡主的情况可以尝试把workers设置为0
                # device='0',
                optimizer='SGD', # using SGD
                # patience=0, # set 0 to close earlystop.
                # resume=True, # 断点续训,YOLO初始化时选择last.pt
                # amp=False, # close amp
                # fraction=0.2,
                project='runs/train',
                name='exp',
                )