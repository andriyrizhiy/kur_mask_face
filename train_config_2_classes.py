#-*-coding:utf-8-*-

import os
import numpy as np
from easydict import EasyDict as edict

config = edict()
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"          ##if u use muti gpu set them visiable there and then set config.TRAIN.num_gpu

config.TRAIN = edict()

#### below are params for dataiter
config.TRAIN.process_num = 3                      ### process_num for data provider
config.TRAIN.prefetch_size = 100                  ### prefect Q size for data provider

config.TRAIN.num_gpu = 1                        ##match with   os.environ["CUDA_VISIBLE_DEVICES"]
config.TRAIN.batch_size = 16                    ###A big batch size may achieve a better result, but the memory is a problem
config.TRAIN.log_interval = 10
config.TRAIN.epoch = 20   ###just keep training , evaluation shoule be take care by yourself,
config.TRAIN.epoch_start = 1
                                               ### generally 10,0000 iters is enough

config.TRAIN.train_set_size=7321              ###widerface train size
config.TRAIN.val_set_size=1149                 ###widerface val size

config.TRAIN.iter_num_per_epoch = config.TRAIN.train_set_size // config.TRAIN.num_gpu // config.TRAIN.batch_size
config.TRAIN.val_iter=config.TRAIN.val_set_size// config.TRAIN.num_gpu // config.TRAIN.batch_size

config.TRAIN.lr_value_every_step = [0.00001,0.0001,0.0005,0.0001,0.00001,0.000001]        ##warm up is used
config.TRAIN.lr_decay_every_step = [500,1000,6000,7500,9000]

config.TRAIN.weight_decay_factor = 5.e-4                  ##l2 regular
config.TRAIN.vis=False                                    ##check data flag

config.TRAIN.norm='BN'    ##'GN' OR 'BN'
config.TRAIN.lock_basenet_bn=False
config.TRAIN.frozen_stages=-1   ##no freeze

config.TEST = edict()
config.TEST.parallel_iterations=8
config.TEST.score_thres = 0.05
config.TEST.iou_thres = 0.3
config.TEST.max_detect = 1500

config.DATA = edict()
config.DATA.root_path=''
config.DATA.train_txt_path='2_classes_mask_dataset/train.txt'
config.DATA.val_txt_path='2_classes_mask_dataset/val.txt'
config.DATA.num_category=2                                  ###face 1  voc 20 coco 80
config.DATA.num_class = config.DATA.num_category + 1        # +1 background

config.DATA.PIXEL_MEAN = [123., 116., 103.]                 ###rgb
config.DATA.PIXEL_STD = [58., 57., 57.]

config.DATA.hin = 320  # input size
config.DATA.win = 320
config.DATA.max_size=[config.DATA.hin,config.DATA.win]  ##h,w
config.DATA.cover_small_face=5                          ###cover the small faces

config.DATA.mutiscale=False                #if muti scale set False  then config.DATA.MAX_SIZE will be the inputsize
config.DATA.scales=(320)

# anchors -------------------------
config.ANCHOR = edict()
config.ANCHOR.rect=True
config.ANCHOR.rect_longer=True       ####    make anchor h/w=1.5
config.ANCHOR.ANCHOR_STRIDE = 16
config.ANCHOR.ANCHOR_SIZES = (16,32,64, 128, 256, 512)   # sqrtarea of the anchor box
config.ANCHOR.ANCHOR_STRIDES = (4, 8,16, 32, 64, 128)  # strides for each FPN level. Must be the same length as ANCHOR_SIZES
config.ANCHOR.ANCHOR_RATIOS = (1., 4.) ######           1:2 in size,
config.ANCHOR.POSITIVE_ANCHOR_THRESH = 0.35
config.ANCHOR.NEGATIVE_ANCHOR_THRESH = 0.35
config.ANCHOR.AVG_MATCHES=20
config.ANCHOR.super_match=True


from lib.core.anchor.anchor import Anchor

config.ANCHOR.achor=Anchor()


# # basemodel ---------------------- fddb 0.983
# config.MODEL = edict()
# config.MODEL.continue_train=False ### revover from a trained model
# config.MODEL.model_path = './model'  # save directory
# config.MODEL.net_structure='resnet_v1_50'#resnet_v1_50,resnet_v1_101,mobilenet
# config.MODEL.pretrained_model='resnet_v1_50.ckpt'
# config.MODEL.fpn_dims=[256,512,1024,2048,512,256]
# config.MODEL.fem_dims=512

###resnet_v1_101 as basemodel
# config.MODEL = edict()
# config.MODEL.continue_train=False ### revover from a trained model
# config.MODEL.model_path = './model/'  # save directory
# config.MODEL.net_structure='resnet_v1_101' ######'resnet_v1_50,resnet_v1_101,mobilenet
# config.MODEL.pretrained_model='resnet_v1_101.ckpt'
# config.MODEL.fpn_dims=[256,512,1024,2048,512,256]
# config.MODEL.fem_dims=512

#vgg as basemodel. if vgg, set config.TRAIN.norm ='None', achieves fddb 0.987
# config.MODEL = edict()
# config.TRAIN.norm='None'
# config.MODEL.l2_norm=[10,8,5]
# config.MODEL.continue_train=False ### revover from a trained model
# config.MODEL.model_path = './model/'  # save directory
# config.MODEL.net_structure='vgg_16'
# config.MODEL.pretrained_model='vgg_16.ckpt'
# config.MODEL.fpn_dims=[256,512,512,1024,512,256]
# config.MODEL.fem_dims=512


# ##mobilenet as basemodel, mobile naet is not that fast
config.MODEL = edict()
config.MODEL.continue_train=False ### revover from a trained model
config.MODEL.model_path = './mobilenetv2_base_2_classes_fix/'  # save directory
config.MODEL.net_structure='MobilenetV2' ######'resnet_v1_50,resnet_v1_101,MobilenetV2

# config.MODEL.net_structure='vgg_16' ######'resnet_v1_50,resnet_v1_101,MobilenetV2
# config.MODEL.l2_norm=[10,8,5]

config.MODEL.pretrained_model=None
# config.MODEL.pretrained_model='mobilenet_v2_1.4_224.ckpt'
# config.MODEL.pretrained_model='/root/andrey/face_detector/DSFD-tensorflow/model/epoch_97L2_0.0005.ckpt'
# config.MODEL.pretrained_model='/media/newdrive/andrew/DSFD-tensorflow_experiments/model_endlersaning/epoch_40L2_0.0005.ckpt'
config.MODEL.fpn_dims=[96,96,288,160,256,256]
config.MODEL.fem_dims=256


config.MODEL.fpn=True
config.MODEL.dual_mode=True
config.MODEL.maxout=True
config.MODEL.max_negatives_per_positive= 3.0

