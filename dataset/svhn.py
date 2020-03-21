import datetime, math, sys, time, os, tarfile
import numpy as np
from scipy import linalg
from scipy.io import loadmat
import glob, argparse
import pickle
from chainer import cuda
from six.moves import xrange  # pylint: disable=redefined-builtin
from six.moves import urllib
import copy

DATA_URL_TRAIN = 'http://ufldl.stanford.edu/housenumbers/train_32x32.mat'
DATA_URL_TEST = 'http://ufldl.stanford.edu/housenumbers/test_32x32.mat'

IMAGE_SIZE = 32
NUM_CLASSES = 10
NUM_EXAMPLES_TRAIN = 73257
NUM_EXAMPLES_TEST = 26032


def maybe_download_and_extract(data_dir):
    """Download and extract the tarball from Alex's website."""
    dest_directory = data_dir
    if not os.path.exists(dest_directory):
        os.makedirs(dest_directory)
    filepath_train_mat = os.path.join(data_dir, 'train_32x32.mat')
    filepath_test_mat = os.path.join(data_dir, 'test_32x32.mat')
    if not os.path.exists(filepath_train_mat) or not os.path.exists(filepath_test_mat):
    #if True:
        def _progress(count, block_size, total_size):
            sys.stdout.write('\r>> Downloading %.1f%%' % (float(count * block_size) / float(total_size) * 100.0))
            sys.stdout.flush()
        urllib.request.urlretrieve(DATA_URL_TRAIN, filepath_train_mat, _progress)
        urllib.request.urlretrieve(DATA_URL_TEST, filepath_test_mat, _progress)

        # Training set
        print("Loading training data...")
        train_data = loadmat(data_dir + '/train_32x32.mat')
        train_images = (-127.5 + train_data['X'].astype(np.float32)) / 255.
        train_images = train_images.transpose((3, 2, 0, 1))
        train_images = train_images.reshape([train_images.shape[0], -1])
        train_labels = train_data['y'].flatten().astype(np.int32)
        train_labels[train_labels == 10] = 0

        # Test set
        print("Loading test data...")
        test_data = loadmat(data_dir + '/test_32x32.mat')
        test_images = (-127.5 + test_data['X'].astype(np.float32)) / 255.
        test_images = test_images.transpose((3, 2, 0, 1))
        test_images = test_images.reshape((test_images.shape[0], -1))
        test_labels = test_data['y'].flatten().astype(np.int32)
        test_labels[test_labels == 10] = 0

        np.savez('{}/train'.format(data_dir), images=train_images, labels=train_labels)
        np.savez('{}/test'.format(data_dir), images=test_images, labels=test_labels)

def extract_specific_category_data(category, images, labels, N=None):
    ind = np.where(labels == category)[0]
    if N is not None:
        ind = ind[0:N]
    extracted_images = images[ind]
    extracted_labels = labels[ind]
    images_extracted_from = np.delete(images, ind, 0)
    labels_extracted_from = np.delete(labels, ind)
    return (extracted_images, extracted_labels), (images_extracted_from, labels_extracted_from)


def load_svhn(data_dir):
    maybe_download_and_extract(data_dir)
    train_data = np.load('{}/train.npz'.format(data_dir))
    test_data = np.load('{}/test.npz'.format(data_dir))
    return (train_data['images'], train_data['labels']), (test_data['images'], test_data['labels'])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, help='', default='svhn')
    parser.add_argument('--num_labeled_examples', type=int, default=1000)
    parser.add_argument('--num_valid_examples', type=int, default=200)
    args = parser.parse_args()
    
    examples_per_class = int(args.num_labeled_examples / 10)
    category_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    for dataset_seed in range(1, 11):
        dirpath = os.path.join(args.data_dir, 'seed' + str(dataset_seed))
        if not os.path.exists(dirpath):
            os.makedirs(dirpath)

        rng = np.random.RandomState(dataset_seed)
        rand_ix = rng.permutation(NUM_EXAMPLES_TRAIN)
        print(rand_ix)
        (train_images, train_labels), (test_images, test_labels) = load_svhn(args.data_dir)
        _train_images, _train_labels = copy.deepcopy(train_images)[rand_ix], copy.deepcopy(train_labels)[rand_ix]
        
        N_l = examples_per_class * len(category_list)
        labeled_train_images = []
        labeled_train_labels = []
        count = 0
        for i in category_list:
            (ext_images, ext_labels), (_train_images, _train_labels) = \
                extract_specific_category_data(i, _train_images, _train_labels, N=examples_per_class)
            labeled_train_images.append(ext_images)
            labeled_train_labels.append(ext_labels)
        labeled_train_images = np.concatenate(labeled_train_images, 0).astype(np.float32)
        labeled_train_labels = np.concatenate(labeled_train_labels, 0).astype(np.int64)
        

        print("N_l:{}, N_ul:{}".format(labeled_train_images.shape[0], train_images.shape[0]))
        np.savez('{}/labeled_train'.format(dirpath), images=labeled_train_images, labels=labeled_train_labels)
        np.savez('{}/unlabeled_train'.format(dirpath), images=train_images,
                 labels=train_labels)  # Do not use labels on training phase.
        np.savez('{}/test'.format(dirpath), images=test_images, labels=test_labels)

        # Dataset for validation
        train_images_valid, train_labels_valid = \
            labeled_train_images[args.num_valid_examples:], labeled_train_labels[args.num_valid_examples:]
        test_images_valid, test_labels_valid = \
            labeled_train_images[:args.num_valid_examples], labeled_train_labels[:args.num_valid_examples]
        unlabeled_train_images_valid = np.concatenate(
            (train_images_valid, _train_images), axis=0)
        unlabeled_train_labels_valid = np.concatenate(
            (train_labels_valid, _train_labels), axis=0)
        np.savez('{}/labeled_train_valid'.format(dirpath), images=train_images_valid, labels=train_labels_valid)
        np.savez('{}/unlabeled_train_valid'.format(dirpath),
                 images=unlabeled_train_images_valid,
                 labels=unlabeled_train_labels_valid)  # Do not use labels on training phase.
        np.savez('{}/test_valid'.format(dirpath), images=test_images_valid, labels=test_labels_valid)
