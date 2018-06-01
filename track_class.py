import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import data, exposure, img_as_float


class track_Sample:
    def __init__(self, ID=0, IMG=None):
        ##image : BGR
        self.id = ID
        self.image = IMG
        if type(self.image) != type(np.array([])):
            raise TypeError("Image must be a numpy array")


class track_Squence:
    id = 0
    Samples = list()

    def __init__(self, ID=0):
        self.id = ID

    def __del__(self):
        self.id=0
        self.Samples=list()

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


class track_dataset:
    track_dataset=list()
    def append(self,Squence):
        if type(Squence) != type(track_Squence(0)):
            raise TypeError("Sample must be track_Sample class")
        self.track_dataset.append(Squence)


def imflip(img):
    [H, W, C] = img.shape
    flip_img = img.copy()
    for i in range(H):
        for j in range(W):
            flip_img[i, j] = img[i, W - j - 1]
    return flip_img


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
    dataset=track_dataset()
    dataset.append(squence)
