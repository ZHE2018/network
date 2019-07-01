from PIL import Image
import numpy as np
import PIL


def get_RGB(num: float):
    """
    将灰度转化为RGB色彩
    :param num: (0,1)的小数，表示灰度
    :return: tuple(R,G,B)
    """
    c = int(255 * num)
    return c, c, c


def get_img(data: [], size=5):
    """
    读取图片数据并将其转化为一个img 对象，实现数据可视化
    :param data: 一个包含784个(0,1)中小数的列表，表示图片(灰度)
    :param size: 生成图片的放大倍数，由于28*28的图片过小，故输出时放大5倍
    :return: 一个Image 对象，可以调用show方法显示
    """
    if len(data) != 784:
        raise Exception('数据大小必须为784！')
    img = Image.new("RGB", (28 * size, 28 * size))  # 创建一个28*28的size被像素图片
    for i in range(28):
        for j in range(28):
            for x in range(size):
                for y in range(size):
                    img.putpixel((i * size + x, j * size + y), get_RGB(data[i * 28 + j]))
    img = img.transpose(PIL.Image.TRANSPOSE)
    return img


if __name__ == "__main__":
    test_data = np.load('test_data.npy', allow_pickle=True)
    count = int(np.random.rand() * len(test_data))
    data, value = test_data[count]
    print(count, " : ", value)
    get_img(data).show()
