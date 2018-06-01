from track_class import *
import os
import pickle
import cv2
import matplotlib.pyplot as plt

if __name__ == '__main__':
    path = 'G:\Tracking\Market\\bounding_box_train\\'
    files = os.listdir(path)
    files.sort()
    init_id = int(files[0].split('_')[0])
    dataset = track_dataset()
    Squence = track_Squence(init_id)
    for num in range(len(files)-1):
        name = files[num].split('_')
        id = int(name[0])
        I = cv2.imread(path + files[num])
        I = np.float32(I / 255.0)
        # plt.imshow(I)
        Sample = track_Sample(id, I)
        if init_id != id:
            init_id = id
            dataset.track_dataset.append(Squence)
            Squence = track_Squence(init_id)
            Squence.Samples=list()
            Squence.Samples.append(Sample)
        else:
            Squence.Samples.append(Sample)
    output = open('Market_Dataset.pkl', 'wb')
    # Pickle dictionary using protocol 0.
    pickle.dump(dataset.track_dataset, output,1)
    output.close()
    # output = open('Market_Dataset.pkl', 'rb')
    # x = pickle.load(output)