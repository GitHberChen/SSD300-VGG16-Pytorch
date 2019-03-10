# config.py
import os.path

# gets home dir cross platform
# ~ 帐户的 home 目录
# 算是个常见的符号，代表使用者的 home 目录：cd ~；
# 也可以直接在符号后加上某帐户的名称：cd ~user
# 或者当成是路径的一部份：~/bin
# ~+ 当前的工作目录，这个符号代表当前的工作目录

HOME = './'
# HOME = os.path.join('')  # 不合理
# HOME = '/Users/chenlinwei/Desktop/计算机学习资料/20181020SSD/ssd.pytorch-master/'
# 把path中包含的"~"和"~user"转换成用户目录
# ===> '/Users/chenlinwei'


# for making bounding boxes pretty
# 边框颜色设置？
COLORS = ((255, 0, 0, 128), (0, 255, 0, 128), (0, 0, 255, 128),
          (0, 255, 255, 128), (255, 0, 255, 128), (255, 255, 0, 128))

# 训练集图片三通道的均值？
MEANS = (104, 117, 123)

# SSD300 CONFIGS
voc = {
    'num_classes': 21,
    'lr_steps': (80000, 100000, 120000),
    'max_iter': 120000,
    'feature_maps': [38, 19, 10, 5, 3, 1],
    'min_dim': 300,
    'steps': [8, 16, 32, 64, 100, 300],
    'min_sizes': [30, 60, 111, 162, 213, 264],
    'max_sizes': [60, 111, 162, 213, 264, 315],
    'aspect_ratios': [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
    'variance': [0.1, 0.2],
    'clip': True,
    'name': 'VOC',
}

coco = {
    'num_classes': 201,
    'lr_steps': (280000, 360000, 400000),
    'max_iter': 400000,
    'feature_maps': [38, 19, 10, 5, 3, 1],
    'min_dim': 300,
    'steps': [8, 16, 32, 64, 100, 300],
    'min_sizes': [21, 45, 99, 153, 207, 261],
    'max_sizes': [45, 99, 153, 207, 261, 315],
    'aspect_ratios': [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
    'variance': [0.1, 0.2],
    'clip': True,
    'name': 'COCO',
}
