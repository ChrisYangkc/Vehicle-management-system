

# 1. 文件夹结构介绍
- configs：配置文件
- Font：字体文件
- icon：图标文件
- img：测试数据存放文件夹
- LicensePlate_detect：车牌检测模块
- LicensePlate_OCR：车牌识别模块
- output：输出文件夹
- tools：脚本文件夹
- weights：权重文件
- main.py：主程序
- requirements.txt：环境版本
- UI.py：界面文件
- UI.ui：界面文件


# 2. 环境安装（已安装好的忽略）

博客中的讲解，只是示例，具体的安装版本以下面提供的为准（安装流程不管哪个版本都是一样的）：
    
    1. python版本： 3.8.10
    
    2. cuda版本：安装哪个版本同自己的电脑显卡有关
        CUDA10.2
        CUDA11.1（建议）
        CUDA11.3

    3. torch版本：需要同安装的cuda进行匹配
        CUDA10.2 安装：torch1.9.0==cuda10.2
        CUDA11.1 安装：torch1.9.0==cuda11.1 （建议）
        CUDA11.3 安装：torch1.10.0==cuda11.3

    4. 其他的第三方库版本见：requirements.txt

# 3. LicensePlate_detect 车牌检测训练
## 训练步骤
CCPD数据集转化见：主目录下tools文件夹下ccpd2VOC.py
注：以下的涉及到的代码都是在LicensePlate_detect/yolox_pytorch 目录下
1. 数据集的准备
**本文使用VOC格式进行训练，训练前需要自己制作好数据集，**    
训练前将标签文件放在Annotation中；将图片文件放在JPEGImages中。

2. 数据集的处理  
在完成数据集的摆放之后，我们需要利用voc_annotation.py获得训练用的train.txt和val.txt。
根据自身情况修改voc_annotation.py里面的以下内容
```python
    # 训练集图片和ann路径
    img_path_train = r'E:\lg\BaiduSyncdisk\project\person_code\chepai_OCR\data\traindata\VOC_LP_det\JPEGImages'
    ann_path_train = r'E:\lg\BaiduSyncdisk\project\person_code\chepai_OCR\data\traindata\VOC_LP_det\Annotations'
    # txt保存路径    
    train_txt_path = r'./VOCdevkit/train.txt'
    # 测试集图片和ann路径
    img_path_val = r'E:\lg\BaiduSyncdisk\project\person_code\chepai_OCR\data\traindata\VOC_LP_det\JPEGImages'
    ann_path_val = r'E:\lg\BaiduSyncdisk\project\person_code\chepai_OCR\data\traindata\VOC_LP_det\Annotations'
    # txt保存路径  
    val_txt_path = r'./VOCdevkit/val.txt'
    # 类别文件，里面填写自己的类别
    classes_path = 'model_data/self_classes.txt'
```

model_data/self_classes.txt文件内容为：      
```python
green
blue
...
```
修改完后，运行voc_annotation.py，会在对应的路径下，生成train.txt 和 val.txt，格式如下：
```python
E:\lg\BaiduSyncdisk\project\person_code\chepai_OCR\data\traindata\VOC_LP_det\JPEGImages\00205459770115-90_85-352&516_448&547-444&547_368&549_364&517_440&515-0_0_22_10_26_29_24-128-7.jpg 352,516,448,547,0
...
```

3. 开始网络训练  
**训练的参数较多，均在train.py中，大家可以在下载库后仔细看注释，其中最重要的部分依然是train.py里的classes_path。**  
**classes_path用于指向检测类别所对应的txt，这个txt和voc_annotation.py里面的txt一样！训练自己的数据集必须要修改！**  
修改完classes_path后就可以运行train.py开始训练了，在训练多个epoch后，权值会生成在logs文件夹中。  

4. 模型预测  
在yolo.py里面修改model_path以及classes_path。  
**model_path指向训练好的权值文件，在logs文件夹里。  
classes_path指向检测类别所对应的txt。**  
完成修改后就可以运行predict_dir_img.py 或者 predict_video.py进行检测了。

## 预测步骤

1. 按照训练步骤训练。  
2. 预测文件夹下图片使用predict_dir_img.py，预测视频使用predict_video.py，修改以下参数：
**model_path对应logs文件夹下面的权值文件，classes_path是model_path对应分的类**。  
```python
yolox_weight = 'weights/best_epoch_weights.pth'
classes_path = './model_data/self_classes.txt'
input_shape = [640, 640]
phi = 's'
confidence = 0.5
nms_iou = 0.3
cuda = 'cuda:0'
```


# 4. LicensePlate_OCR 车牌识别
## 训练步骤
CCPD数据集转化见：主目录下tools文件夹下CCPD2lpr.py
注：以下的涉及到的代码都是在LicensePlate_OCR/lpr_net 目录下

1. 数据集的准备

使用 CCPD2lpr.py制作



2. 修改 train_lpr.py以下主要参数：
```python
parser.add_argument('--max_epoch', default=500, help='epoch to train the network')
parser.add_argument('--img_size', default=[94, 24], help='the image size')
parser.add_argument('--train_img_dirs',
                    default=r"E:\lg\BaiduSyncdisk\project\person_code\chepai_OCR\data\traindata\LPR_traindata",
                    help='the train images path')
parser.add_argument('--test_img_dirs', default=r"E:\lg\BaiduSyncdisk\project\person_code\chepai_OCR\data\traindata\LPR_traindata",
                    help='the test images path')
parser.add_argument('--train_batch_size', default=128, help='training batch size.')
parser.add_argument('--pretrained_model', default=r'E:\lg\BaiduSyncdisk\project\person_code\chepai_OCR\code\LicensePlate_OCR\lpr_net\runs\best.pth', help='no pretrain')
parser.add_argument('--save_folder', default=r'./runs/', help='Location to save checkpoint models')

```
3. 运行train_lpr.py 进行模型训练，模型保存在 runs 文件夹下


## 预测

predict_lpr.py
修改以下参数：
```python
weights = r'D:\lg\BaiduSyncdisk\project\person_code\chepai_OCR\code\LicensePlate_OCR\lpr_net\runs\best.pth'
parser.add_argument('--test_img_dirs', default=r"D:\lg\BaiduSyncdisk\project\person_code\chepai_OCR\data\traindata\LPR_traindata\green", help='the test images path')
parser.add_argument('--img_size', default=[94, 24], help='the image size')
```