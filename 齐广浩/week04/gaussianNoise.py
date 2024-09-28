import cv2
import random
"""
高斯噪声实现
A.输入参数sigma和mean
B.生成高斯随机数
C.根据输入像素计算出输出像素
D.重新将像素放缩在[0-255]之间
E.循环所有像素
F.输出图像
"""


def GuassianNoise(img_src, mean, sigma, percetage):
    nosie_img = img_src.copy()
    nosieNum = int(percetage * img_src.shape[0] * img_src.shape[1])
    for i in range(nosieNum):
        # 随机生成randX和randY
        # 每次取一个点来进行高斯处理
        # 生成高斯随机数与单个点个相加
        randX = random.randint(0, img_src.shape[0]-1)
        randY = random.randint(0, img_src.shape[1]-1)

        # 对灰度图进行高斯噪声处理
        nosie_img[randX, randY] = nosie_img[randX, randY] + random.gauss(mean, sigma)
        # 假如对彩色图像进行高斯噪声处理
        # 需要逐个通道处理 (BGR)，防止溢出
        """
        for channel in range(img_src.shape[2]):
            new_value = noise_img[randX, randY, channel] + noise
            # 限制像素值在 [0, 255] 范围内
            new_value = np.clip(new_value, 0, 255)
            noise_img[randX, randY, channel] = new_value
        """
        if nosie_img[randX, randY] < 0:
            nosie_img[randX, randY] = 0
        elif nosie_img[randX, randY] > 255:
            nosie_img[randX, randY] = 255
    return nosie_img


if __name__ == '__main__':
    img = cv2.imread("../lenna.png")
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_guass = GuassianNoise(img_gray, 10, 5, 0.8)
    # img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imshow("source", img_gray)
    cv2.imshow("guass_img", img_guass)
    cv2.waitKey(0)
