import os, sys

path = os.path.split(__file__)[0]
# print("abs path is %s" %(os.path.abspath()))

config = {
    'batch_size' : 4,
    'val_batch_size':2,
    'epochs':200,
    'seed' : 1,
    'lr':0.1,
    'weight_decay':1e-10,
    'print_freq':500,
    'start_epoch':0,
    'cuda' : True,
    'gpus' :1,
    'workers':1,

    'model':'ColorizationNet',
    'bachnorm':True,
    'pretrained':True,


    'save' :'D:\\test\\work\\try2\\',

    'image_folder_train' : 'D:\\test\\test\\train\\' ,
    'image_folder_val' :'D:\\Colorization-master\\Colorization-resnet\\test_image\\',

    'log_frequency': 1, #frequency for the number of epotch
    'save_iamge':'D:\\demo\\work\\try2\\img\\'
}

