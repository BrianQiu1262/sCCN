import pickle as pkl
import tensorflow as tf
import numpy as np
import os


class Dataset(object):
    def __init__(self, path_to_file, device, batch_size=None):

        os.environ["CUDA_VISIBLE_DEVICES"] = device

        super(Dataset, self).__init__()

        with open(path_to_file, 'rb') as f:
            data_dict = pkl.load(f)

        data = np.array(data_dict['data'])
        labels = np.array([i for i in data_dict['label']])
        act = np.array([a for a in data_dict['activations']]).astype(np.float32)

        self.length = len(data_dict['data'])
        self.dataset = tf.data.Dataset.from_tensor_slices((data, labels, act)).shuffle(buffer_size=self.length, reshuffle_each_iteration=True).batch(batch_size)

    def __len__(self):
        return self.length

