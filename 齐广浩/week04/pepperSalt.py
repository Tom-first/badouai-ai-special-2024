import cv2
import random
"""
给一幅数字图像加上椒盐噪声的处理顺序：
1）指定信噪比SNR（信号和噪声所占比例），其取值范围在[0-1]之间
2）计算总像素数目SP，得到要加噪的像素数目NP= SP*SNR
3）随机获取要加噪的每个像素位置P（i，j）
4）指定像素值为255或0
5）重复3,4两个步骤完成所有NP个像素的加噪
"""


def PepperSalt(src, percetage):
    pst_img = src.copy()
    pst_img_num = int(percetage*src.shape[0]*src.shape[1])
    for i in range(pst_img_num):
        randX = random.randint(0, pst_img.shape[0]-1)
        randY = random.randint(0, pst_img.shape[1]-1)
        # 随机取值，一半设置成白色像素，一半设置成黑丝像素
        if random.random() <= 0.5:
            pst_img[randX, randY] = 0
        else:
            pst_img[randX, randY] = 255
    return pst_img


if __name__ == '__main__':
    img = cv2.imread("../lenna.png")
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    Pst_img = PepperSalt(img_gray, 0.8)
    cv2.imshow("Source", img_gray)
    cv2.imshow("PepperSalt",Pst_img)
    cv2.waitKey(0)

