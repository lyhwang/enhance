import scipy
from glob import glob
import numpy as np


class DataLoader():
    def __init__(self, dataset_name, img_res=(512, 512)):
        self.dataset_name = dataset_name
        self.img_res = img_res

    def load_data(self, domain, batch_size=1, is_testing=False):
        data_type = "train%s" % domain if not is_testing else "test%s" % domain
        path = glob('./datasets/%s/%s/*' % (self.dataset_name, data_type))

        batch_images = np.random.choice(path, size=batch_size)

        imgs = []
        for img_path in batch_images:
            img = self.imread(img_path)
            if not is_testing:
                img = scipy.misc.imresize(img, self.img_res)

                if np.random.random() > 0.5:
                    img = np.fliplr(img)
            else:
                img = scipy.misc.imresize(img, self.img_res)
            imgs.append(img)

        imgs = np.array(imgs)/127.5 - 1.

        return imgs

    def load_batch(self, batch_size=1, is_testing=False):
        data_type = "train" if not is_testing else "val"
        path_I = glob('./datasets/%s/%sI/*' % (self.dataset_name, data_type))
        self.n_batches = int(len(path_I) / batch_size)
        total_samples = self.n_batches * batch_size

        # Sample n_batches * batch_size from each path list so that model sees all
        # samples from both domains
        path_I = np.random.choice(path_I, total_samples, replace=False)

        for i in range(self.n_batches-1):
            batch_I = path_I[i*batch_size:(i+1)*batch_size]
            imgs_I = []
            for img_I in batch_I:
                img_I = self.imread(img_I)

                img_I = scipy.misc.imresize(img_I, self.img_res)

                if not is_testing and np.random.random() > 0.5:
                        img_I = np.fliplr(img_I)

                imgs_I.append(img_I)

            imgs_I = np.array(imgs_I)/127.5 - 1.

            yield imgs_I

    def load_img(self, path):
        img = self.imread(path)
        img = scipy.misc.imresize(img, self.img_res)
        img = img/127.5 - 1.
        return img[np.newaxis, :, :, :]

    def load_lowlightimg(self, path):
        img = self.imread(path)
        imgwidth = img.shape[1]
        imgheight = img.shape[0]
        img = scipy.misc.imresize(img, self.img_res)
        img = img/127.5 - 1.
        return imgwidth, imgheight, img[np.newaxis, :, :, :]

    def imread(self, path):
        return scipy.misc.imread(path, mode='RGB').astype(np.float)