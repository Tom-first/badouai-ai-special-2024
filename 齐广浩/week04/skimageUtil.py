import cv2
import numpy as np
from PIL import Image
from skimage.util import random_noise

# def random_noise(image, mode = 'gaussian', seed = None, clip = True, **kwargs) 使用skimage库实现图像的噪声实现
"""
功能实现：为浮点型图片添加各种随机噪声
image：输入图片（将会被转换成浮点数）ndarray型
mode：可选择，str类型，表示要添加的噪声类型
    gaussian：高斯噪声
    localvar：高斯分布的加性噪声，在图像的每个点出具有指定的局部方差
    poisson：泊松噪声
    salt：盐噪声
    pepper：椒噪声
    s&p：椒盐噪声
    specklce：均匀噪声（均值mean方差variance），out= image+n*image
    

seed：可选，int类型，如果选择的话，在生成噪声之前会先设置随机种子，以避免出现伪随机
clip：可选，bool型，如果是True，在添加均值，泊松以及高斯噪声后，会将图片裁剪到合适范围内，如果是False，则输出矩阵的值
mean：可选，float型，主要是高斯噪声和均值噪声中的mean参数，默认是0
var： 可选，float型，高斯噪声和均值噪声中的方差，默认值是0.01（非标准差）
local_vars:可选，ndarray型，用于定义每个像素点的局部方差，来localvar中使用
amount：可选，float型，是椒盐噪声所占比例，默认值0.05
salt_vs_pepper:可选，float型，椒盐噪声中椒盐比例，值越大，表示盐噪声越多，默认值0.5，表示椒盐等量
----------
返回值：ndarray型，且值在[0,1]或者[-1,1]之间，取决于是否是有符号数
"""


def NoiseImgAchieve(img, typeNoise):
    noise_img = random_noise(img, mode=typeNoise)
    return noise_img


if __name__ == '__main__':
    img = cv2.imread("../lenna.png")
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 高斯噪声接口调用实现
    typeNoiseGaussian = "gaussian"
    Gaussian_img = NoiseImgAchieve(img, typeNoiseGaussian)
    Guassian_img_gray = NoiseImgAchieve(img_gray, typeNoiseGaussian)
    cv2.imshow("resource", img)
    cv2.imshow("gaussian", Gaussian_img)
    cv2.imshow("GrayGaussian", Guassian_img_gray)

    # 椒盐噪声接口调用实现
    typeNoiseSP = "s&p"
    sAndP_img = NoiseImgAchieve(img, typeNoiseSP)
    sAndP_img_gray = NoiseImgAchieve(img_gray, typeNoiseSP)
    cv2.imshow("s&p", sAndP_img)
    cv2.imshow("Grays&p", sAndP_img_gray)

    cv2.waitKey(0)
