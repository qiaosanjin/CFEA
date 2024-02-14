#数据集的类别
NUM_CLASSES = 21

#训练时batch的大小
BATCH_SIZE = 32

#训练轮数
NUM_EPOCHS= 200

#训练完成，精度和损失文件的保存路径,默认保存在trained_models下
#TRAINED_MODEL = '/data3/qiaoxin/CFEANet-Submitted-to-IEEE-JSTARS-main_qx/trained_models-nwpu28/data_record.pth'
TRAINED_MODEL = '/data3/qiaoxin/codeset/CFEANet-Submitted-to-IEEE-JSTARS-main_qx1/trained_models-ucm55/data_record.pth'

#数据集的存放位置
#TRAIN_DATASET_DIR = '/data3/qiaoxin/WHU-RS19/64/train'
#VALID_DATASET_DIR = '/data3/qiaoxin/WHU-RS19/64/val'
#TRAIN_DATASET_DIR = '/data3/qiaoxin/UCMerced_LandUse/82/train'
#VALID_DATASET_DIR = '/data3/qiaoxin/UCMerced_LandUse/82/val'
TRAIN_DATASET_DIR = '/data3/qiaoxin/UCMerced_LandUse/55/train'
VALID_DATASET_DIR = '/data3/qiaoxin/UCMerced_LandUse/55/val'
datset_dir = '/data3/qiaoxin/UCMerced_LandUse'
#TRAIN_DATASET_DIR = '/data3/qiaoxin/AID/28/train'
#VALID_DATASET_DIR = '/data3/qiaoxin/AID/28/val'
#TRAIN_DATASET_DIR = '/data3/qiaoxin/dataset/RSSCN7/28/train'
#VALID_DATASET_DIR = '/data3/qiaoxin/dataset/RSSCN7/28/val'
#TRAIN_DATASET_DIR = '/data3/qiaoxin/OPTIMAL-31/82/train'
#VALID_DATASET_DIR = '/data3/qiaoxin/OPTIMAL-31/82/val'

# -*- coding:utf-8 -*- 
class DefaultConfigs(object):
    data_root = '/data3/qiaoxin' # 数据集的根目录 因为train和valid的txt里边已经有UCMerced_LandUse/Images,所以去掉  /data3/qiaoxin/UCMerced_LandUse/Images 
    model = 'CFEANet' # ResNet18, ResNet34, ResNet50, ResNet101, ResNet152 使用的模型
    freeze = True # 是否冻结卷基层

    seed = 1000 # 固定随机种子
    num_workers = 12 # DataLoader 中的多线程数量
    num_classes = 21 # 分类类别数
    num_epochs = 10
    batch_size = 16
    lr = 0.01 # 初始lr
    width = 256 # 输入图像的宽
    height = 256 # 输入图像的高
    iter_smooth = 105 # 打印&记录log的频率

    resume = False #
    #checkpoint = 'ResNet152.pth' # 训练完成的模型名
    #checkpoint = 'model-bifpn-ecda99.29.pth'
                 #'CFEANet-model-bifpn-none attention99.52.pth'
    #checkpoint = 'ResNet18.pth' #qiaoxin
    checkpoint = 'model-bifpn2-ecda4-concate4-99.52.pth'

config = DefaultConfigs()
