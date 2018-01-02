#读取人脸库olivettiface，并储存为pkl文件

import numpy
from PIL import Image
import pickle
import pylab


def putDataAndSave(filename):
    #读取原始图片，并转化为数组，将灰度值由0~256转化到0~1
    img = Image.open(filename)
    img_ndarray = numpy.asarray(img, dtype='float32') /256 #一个在内存中占分别32和64个bits
    '''
        图片大小为1190*942，一共20*20个人脸，每张人脸图大小为（1190/20）* (943/20)，即每张人脸的大小为57*47=2679
        将全部400个样本存储你为一个400*2679的数组，每一行代表一个人脸图，并且0~9，10~19，20~29……行表示同个人的人脸（角度不同）
        另外，用olivettiface_label表示每一个样本的类别，它是400维的向量，有0~39共有40类，代表40个不同的人
    '''
    olivettifaces = numpy.empty((400,2679))
    for row in range(20):
        for colum in range(20):
            olivettifaces[row*20 + colum] = numpy.ndarray.flatten(img_ndarray[row*57:(row+1)*57, colum*47:(colum+1)*47])


    #给图像数组标记类别
    olivettifaces_label = numpy.empty(400)
    for label in range(40):
        olivettifaces_label[label*10:label*10+10] = label
    olivettifaces_label = olivettifaces_label.astype(int)


    #把得到的数组放到*.pkl数组中
    write_file = open('olivettifaces.pkl','wb')
    pickle.dump(olivettifaces,write_file,-1)
    pickle.dump(olivettifaces_label,write_file,-1)
    write_file.close()

def showImageOnline():
    read_file = open('olivettifaces.pkl','rb')
    faces = pickle.load(read_file)
    read_file.close()
    img1 = faces[144].reshape(57,47)
    pylab.imshow(img1)
    pylab.gray()
    pylab.show()
    print(faces)



if __name__ == "__main__":
    putDataAndSave('olivettifaces.gif')

