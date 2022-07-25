import numpy as np


class Buffer:
    def __init__(self, capacity, image_shape):
        self.images = np.zeros([capacity, image_shape[0], image_shape[1], image_shape[2]])
        self.labels = np.zeros(capacity)
        self.current_index = 0
        self.capacity = capacity
        self.is_full = False

    def add(self, image, label):
        if self.current_index == self.capacity:
            self.is_full = True
            self.current_index = 0
        self.images[self.current_index] = image
        self.labels[self.current_index] = label
        self.current_index += 1

    def sample(self, N):
        """
        Slice buffer into N mini-batches
        :param N: number of mini-batches
        :return:
        """
        image_mini_batches = []
        label_mini_batches = []

        if self.is_full:
            indices = np.arange(self.capacity)
        else:
            indices = np.arange(self.current_index)

        np.random.shuffle(indices)
        shuffled_images = self.images[indices]
        shuffled_labels = self.labels[indices]
        mini_batch_size = int(self.capacity / N)
        for i in range(N):
            image_mini_batches.append(shuffled_images[i * mini_batch_size:(i + 1) * mini_batch_size])
            label_mini_batches.append(shuffled_labels[i * mini_batch_size:(i + 1) * mini_batch_size])

        return image_mini_batches, label_mini_batches
