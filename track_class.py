import cv2
import numpy as np
import pickle


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

    def append(self, Sample):
        if type(Sample) != type(track_Sample(0,np.array([]))):
            raise TypeError("Sample must be track_Sample class")
        self.Samples.append(Sample)

    def Squence_check(self):
        for i in range(len(self.Samples)):
            if self.Samples[i].id !=self.id:
                raise TypeError("ID number must be equal")
        print("check success")


if __name__ == "__main__":
    track_list = list()
    track_samples = list()
    IMG = cv2.imread('10.png')
    sample = track_Sample(0, IMG)
    squence=track_Squence(ID=0)
    squence.append(sample)
    #track_samples.append(sample)

    # output = open('data.pkl', 'wb')
    # # Pickle dictionary using protocol 0.
    # pickle.dump(track_sample, output)
    # output.close()
    # output = open('data.pkl', 'rb')
    # x = pickle.load(output)
