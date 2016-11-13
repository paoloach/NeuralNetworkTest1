import numpy as np
from scipy import random
from scipy import misc

class BachData:
    def __init__(self, data, all_label, label_index, num_classes, bach_size=200):
        self.num_classes = int(num_classes)
        self.bach_size = bach_size
        self.data = data
        self.all_label = all_label
        self.label_index = label_index
        for i in range(len(label_index)):
            a = random.random_integers(0,len(label_index)-1,2)
            label_index[a[0]],label_index[a[1]] = label_index[a[1]],label_index[a[0]]
        self.currentIndex = 0

    def get_num_classes(self):
        return self.num_classes

    def get_bach_data(self):
        data = []
        labels_sparse = []
        label = []
        upper = self.currentIndex + self.bach_size
        remain = 0
        if upper > self.label_index.shape[0]:
            remain = int(upper - self.label_index.shape[0])
            upper = self.label_index.shape[0]

        for index in self.label_index[self.currentIndex:upper]:
            sparse_label = np.zeros([self.num_classes])
            sparse_label[self.all_label[index]] = 1
            if len(data) == 0:
                data = np.array([self.data[index]])
                labels_sparse = np.array([sparse_label])
                label = np.array([self.all_label[index]])
            else:
                data = np.append(data, [self.data[index]], 0)
                labels_sparse = np.append(labels_sparse, [sparse_label], 0)
                label = np.append(label, [self.all_label[index]], 0)
        self.currentIndex = upper
        if remain > 0:
            for index in self.label_index[0:remain]:
                sparse_label = np.zeros([self.num_classes])
                sparse_label[self.all_label[index]] = 1
                data = np.append(data, [self.data[index]], 0)
                labels_sparse = np.append(labels_sparse, [sparse_label], 0)
                label = np.append(label, [self.all_label[index]], 0)
            self.currentIndex = remain

        return data, labels_sparse, label
