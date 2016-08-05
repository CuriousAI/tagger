#!/usr/bin/env python
# coding=utf-8

from __future__ import division, print_function, unicode_literals
import os
import os.path
import numpy as np
import h5py
from fuel.converters.base import fill_hdf5_file
np.random.seed(104174)

# update to local for easy debugging
data_dir = './'

square = np.array(
    [[1, 1, 1, 1, 1, 1, 1],
     [1, 1, 1, 1, 1, 1, 1],
     [1, 1, 0, 0, 0, 1, 1],
     [1, 1, 0, 0, 0, 1, 1],
     [1, 1, 0, 0, 0, 1, 1],
     [1, 1, 1, 1, 1, 1, 1],
     [1, 1, 1, 1, 1, 1, 1]])

triangle = np.array(
    [[0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0],
     [0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0],
     [0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0],
     [0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0],
     [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
     [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])

shapes = [square, triangle, triangle[::-1, :].copy()]


def generate_shapes_image(width, height, nr_shapes=3):
    img = np.zeros((height, width))
    grp = np.zeros_like(img)
    k = 1

    for i in range(nr_shapes):
        shape = shapes[np.random.randint(0, len(shapes))]
        sy, sx = shape.shape
        x = np.random.randint(0, width-sx+1)
        y = np.random.randint(0, height-sy+1)
        region = (slice(y,y+sy), slice(x,x+sx))
        img[region][shape != 0] += 1
        grp[region][shape != 0] = k
        k += 1

    grp[img > 1] = 0
    img = img != 0
    return img, grp


# Definition of the number of instances in training set.
np.random.seed(265076)
nr_train_examples = 50000   # 50000
nr_valid_examples = 10000   # 10000
nr_test_examples = 10000    # 10000
nr_single_examples = 10000  # 10000

width = 20
height = 20
nr_shapes = 3

data = np.zeros((nr_train_examples, height, width), dtype=np.float32)
grps = np.zeros_like(data, dtype=np.uint8)
for i in range(nr_train_examples):
    data[i], grps[i] = generate_shapes_image(width, height, nr_shapes)

data_valid = np.zeros((nr_valid_examples, height, width), dtype=np.float32)
grps_valid = np.zeros_like(data_valid, dtype=np.uint8)
for i in range(nr_valid_examples):
    data_valid[i], grps_valid[i] = generate_shapes_image(width, height, nr_shapes)

data_test = np.zeros((nr_test_examples, height, width), dtype=np.float32)
grps_test = np.zeros_like(data_test, dtype=np.uint8)
for i in range(nr_test_examples):
    data_test[i], grps_test[i] = generate_shapes_image(width, height, nr_shapes)

data_single = np.zeros((nr_single_examples, height, width), dtype=np.float32)
grps_single = np.zeros_like(data_single, dtype=np.uint8)
for i in range(nr_single_examples):
    data_single[i], grps_single[i] = generate_shapes_image(width, height, 1)


targets = np.zeros((nr_train_examples, 1), dtype=np.uint8)
targets_valid = np.zeros((nr_valid_examples, 1), dtype=np.uint8)
targets_test = np.zeros((nr_test_examples, 1), dtype=np.uint8)
targets_single = np.zeros((nr_single_examples, 1), dtype=np.uint8)

codes = np.zeros((nr_train_examples, nr_shapes+1, width*height), dtype=np.uint8)
codes_valid = np.zeros((nr_valid_examples, nr_shapes+1, width*height), dtype=np.uint8)
codes_test = np.zeros((nr_test_examples, nr_shapes+1, width*height), dtype=np.uint8)
codes_single = np.zeros((nr_single_examples, nr_shapes+1, width*height), dtype=np.uint8)


# make masks one-hot but set 1/nr_shapes for unscored pixels
A = np.eye(nr_shapes + 1, dtype=np.float32)
A[0] = 1.0 / nr_shapes
A = A[:, 1:]

grps = np.swapaxes(np.swapaxes(A[grps], 3, 2), 2, 1)
grps_valid = np.swapaxes(np.swapaxes(A[grps_valid], 3, 2), 2, 1)
grps_single = np.swapaxes(np.swapaxes(A[grps_single], 3, 2), 2, 1)
grps_test = np.swapaxes(np.swapaxes(A[grps_test], 3, 2), 2, 1)


split = (('train', 'features', data.reshape((-1, width*height))),
         ('train', 'mask', grps.reshape((-1, nr_shapes, width*height))),
         ('train', 'targets', targets),
         ('train', 'codes', codes),
         ('valid', 'features', data_valid.reshape((-1, width*height))),
         ('valid', 'mask', grps_valid.reshape((-1, nr_shapes, width*height))),
         ('valid', 'targets', targets_valid),
         ('valid', 'codes', codes_valid),
         ('test', 'features', data_test.reshape(-1, width*height)),
         ('test', 'mask', grps_test.reshape((-1, nr_shapes, width*height))),
         ('test', 'targets', targets_test),
         ('test', 'codes', codes_test),
         ('single', 'features', data_single.reshape(-1, width*height)),
         ('single', 'mask', grps_single.reshape((-1, nr_shapes, width*height))),
         ('single', 'targets', targets_single),
         ('single', 'codes', codes_single))

for n, m, d in split:
    print(n, m, d.shape)

h5file = h5py.File(os.path.join(data_dir, 'shapes.h5f'), mode='w')

fill_hdf5_file(h5file, split)


h5file.attrs['description'] = """
Shapes Problem
==============

Binary images containing 3 random shapes each. Introduced in [1] to investigate
binding in deep networks.
All images are of size 1 x {height} x {width}.
There are {nr_train_examples} training examples and {nr_test_examples} test
examples with {nr_shapes} random shapes each.
There are also {nr_single_examples} examples with just a single random shape.
There are three different shapes: ['square', 'up-triangle', 'down-triangle'].

[1] David P. Reichert and Thomas Serre,
    Neuronal Synchrony in Complex-Valued Deep Networks, ICLR 2014
""".format(nr_shapes=nr_shapes,
           height=height,
           width=width,
           nr_train_examples=nr_train_examples,
           nr_test_examples=nr_test_examples,
           nr_single_examples=nr_single_examples)

h5file.flush()
h5file.close()


def compress_fuel(source_filename, target_filename):
    source = h5py.File(os.path.join(data_dir, source_filename), mode='r')
    target = h5py.File(os.path.join(data_dir, target_filename), mode='w')
    for data in source:
        print('converting {}'.format(data))
        target.create_dataset(data, data=source[data][:], compression='gzip')
    for attr in source.attrs:
        target.attrs[attr] = source.attrs[attr]

compress_fuel('shapes.h5f', 'shapes50k_20x20_compressed.h5')
