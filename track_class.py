import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import data, exposure, img_as_float


##存储单个跟踪训练样本
class track_Sample:
    def __init__(self, ID=0, IMG=None):
        ##image : BGR
        self.id = ID
        self.image = IMG
        if type(self.image) != type(np.array([])):
            raise TypeError("Image must be a numpy array")


##存储跟踪样本组
class track_Squence:
    id = 0
    Samples = list()

    def __init__(self, ID=0):
        self.id = ID
        self.Samples = list()

    def append(self, Sample):
        if type(Sample) != type(track_Sample(0, np.array([]))):
            raise TypeError("Sample must be track_Sample class")
        if Sample.image.dtype != np.float32().dtype:
            raise TypeError("Sample.image must be float32")
        self.Samples.append(Sample)

    def Squence_check(self):
        for i in range(len(self.Samples)):
            if self.Samples[i].id != self.id:
                raise TypeError("ID number must be equal")
        print("check success")

    def Samples_Augment(self, rate=0.5):
        np.random.seed(1)
        indx = np.random.randint(0, len(self.Samples), int(rate * len(self.Samples)))
        for i in indx:
            img = self.Samples[i].image
            for Gamma in np.logspace(-1, 0.3, 5):
                gam = exposure.adjust_gamma(img, Gamma)
                sample = track_Sample(self.id, gam)
                self.append(sample)


## 存储整个跟踪数据库
class track_dataset:
    track_dataset = list()

    def append(self, Squence):
        if type(Squence) != type(track_Squence(0)):
            raise TypeError("Sample must be track_Sample class")
        self.track_dataset.append(Squence)


## 图片水平翻转
def imflip(img):
    [H, W, C] = img.shape
    flip_img = img.copy()
    for i in range(H):
        for j in range(W):
            flip_img[i, j] = img[i, W - j - 1]
    return flip_img


## 获得三元组数据
def get_triple(Batch, trackData_list):
    if len(trackData_list) < Batch:
        raise TypeError("The length of trackData_list must longer than bach size")
    batchData = list()
    posData = list()
    negData = list()

    squenceIndx = np.random.randint(0, len(trackData_list), Batch)
    for i in squenceIndx:
        sampleIndx = np.random.randint(0, len(trackData_list[i].Samples), 1)
        posIndx = np.random.randint(0, len(trackData_list[i].Samples), 1)
        while (posIndx == sampleIndx):
            print("sampleIndx==posIndx")
            posIndx = np.random.randint(0, len(trackData_list[i].Samples), 1)

        negsIndx = np.random.randint(0, len(trackData_list), 1)
        while (i == negsIndx):
            print("squenceIndx==negIndx")
            negsIndx = np.random.randint(0, len(trackData_list), 1)
        negIndx = np.random.randint(0, len(trackData_list[negsIndx[0]].Samples), 1)

        neg = trackData_list[negsIndx[0]].Samples[negIndx[0]]
        data = trackData_list[i].Samples[sampleIndx[0]]
        pos = trackData_list[i].Samples[posIndx[0]]

        negData.append(neg)
        batchData.append(data)
        posData.append(pos)

    return batchData, posData, negData

## 获得训练batch三元组数据->N*H*W*C
def get_triple_CNN(Batch, trackData_list):
    if len(trackData_list) < Batch:
        raise TypeError("The length of trackData_list must longer than bach size")
    batchData = list()
    posData = list()
    negData = list()

    squenceIndx = np.random.randint(0, len(trackData_list), Batch)
    for i in squenceIndx:
        sampleIndx = np.random.randint(0, len(trackData_list[i].Samples), 1)
        posIndx = np.random.randint(0, len(trackData_list[i].Samples), 1)
        while (posIndx == sampleIndx):
            print("sampleIndx==posIndx")
            posIndx = np.random.randint(0, len(trackData_list[i].Samples), 1)

        negsIndx = np.random.randint(0, len(trackData_list), 1)
        while (i == negsIndx):
            print("squenceIndx==negIndx")
            negsIndx = np.random.randint(0, len(trackData_list), 1)

        negIndx = np.random.randint(0, len(trackData_list[negsIndx[0]].Samples), 1)
        neg = trackData_list[negsIndx[0]].Samples[negIndx[0]]
        data = trackData_list[i].Samples[sampleIndx[0]]
        pos = trackData_list[i].Samples[posIndx[0]]
        negData.append(neg)
        batchData.append(data)
        posData.append(pos)

    Batch_data = np.reshape(batchData[0].image, (1, 128, 64, 3))
    Batch_pos = np.reshape(posData[0].image, (1, 128, 64, 3))
    Batch_neg = np.reshape(negData[0].image, (1, 128, 64, 3))

    # Batch_data = np.concatenate((Batch_data, np.reshape(data[1].image, (1, 128, 64, 3))))

    for i in range(1, Batch):
        Batch_data = np.concatenate((Batch_data, np.reshape(batchData[i].image, (1, 128, 64, 3))))
        Batch_pos = np.concatenate((Batch_pos, np.reshape(posData[i].image, (1, 128, 64, 3))))
        Batch_neg = np.concatenate((Batch_neg, np.reshape(negData[i].image, (1, 128, 64, 3))))

    return Batch_data, Batch_pos, Batch_neg


if __name__ == "__main__":
    # IMG = cv2.imread('lena.jpg')
    IMG = cv2.imread('10.png')
    img = np.float32(img_as_float(IMG))  ##transform to float32

    # for Gamma in np.logspace(-1, 0.3, 5):
    #     gam = exposure.adjust_gamma(img, Gamma)
    #     cv2.imshow('1', gam)
    #     cv2.waitKey(0)
    #     cv2.destroyAllWindows()

    sample = track_Sample(0, img)
    squence = track_Squence(ID=0)
    squence.append(sample)
    squence.Samples_Augment(rate=1)
    dataset = track_dataset()
    dataset.append(squence)

    sample1 = track_Sample(1, img)
    squence1 = track_Squence(ID=1)
    squence1.append(sample1)
    squence1.Samples_Augment(rate=1)
    dataset.append(squence1)

    [Data,Pos,Neg]=get_triple(1,dataset.track_dataset)