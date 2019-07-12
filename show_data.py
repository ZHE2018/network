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


def datas_to_image(data: [], width=75, show_log=True, sort=True):
    """
    从数据集中生成一个PIL.Image 对象
    数据集中的数据项将会按照水平，从左至右依次排列，
    当数据项数大于 width^2 时，每行排列数据项由 width 指定，
    否则，自动计算
    排列行数（图片高度）自动计算
    :param data: 数据集
    :param width: 生成图片水平排列数据数量
    :param show_log: 是否显示进度
    :param sort: 是否对数据集排序
    :return: PIL.Image 对象
    """
    if sort:
        data = list(data)
        data.sort(key=lambda t: t[1])
    if len(data) < width ** 2:
        width = int(np.ceil(np.sqrt(len(data))))
        height = width
    else:
        height = int(np.ceil(len(data) / width))
    if show_log:
        print("宽：{} 高：{}".format(width, height))

    img = Image.new('RGB', (28 * width, 28 * height), "#FF0000")
    img_ptr = img.load()
    ax, ay = 0, 0
    for i in range(len(data)):
        for x in range(28):
            for y in range(28):
                c = data[i][0][x * 28 + y] * 255
                c = int(c)
                img_ptr[ay * 28 + y, ax * 28 + x] = (c, c, c)
        ay += 1
        if ay == width:
            if show_log:
                print('行 {}/{} 完成'.format(ax + 1, height))
            ax += 1
            ay = 0
    return img


def get_img(data_item: [], size=5):
    """
    读取单个图片数据并将其转化为一个img 对象，实现数据可视化
    :param data_item: 一个包含784个(0,1)中小数的列表，表示图片(灰度)
    :param size: 生成图片的放大倍数，由于28*28的图片过小，故输出时放大5倍
    :return: 一个Image 对象，可以调用show方法显示
    """
    if len(data_item) != 784:
        raise Exception('数据大小必须为784！')
    img = Image.new("RGB", (28 * size, 28 * size))  # 创建一个28*28的size被像素图片
    for i in range(28):
        for j in range(28):
            for x in range(size):
                for y in range(size):
                    img.putpixel((i * size + x, j * size + y), get_RGB(data_item[i * 28 + j]))
    img = img.transpose(PIL.Image.TRANSPOSE)
    return img


if __name__ == "__main__":
    test_data = np.load('test_data.npy', allow_pickle=True)
    datas_to_image(list(test_data)).show()
