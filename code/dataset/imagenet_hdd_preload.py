from tensorpack.dataflow import MultiProcessRunnerZMQ, MultiProcessRunner, MultiThreadRunner, LMDBSerializer, dataset
import numpy as np
import os

class BinaryILSVRC12(dataset.ILSVRC12Files):
    def __iter__(self):
        for fname, label in super(BinaryILSVRC12, self).__iter__():
            with open(fname, 'rb') as f:
                jpeg = f.read()
            jpeg = np.asarray(bytearray(jpeg), dtype='uint8')
            yield [jpeg, label]


if not os.path.exists("./imagenet/ILSVRC-train.lmdb"):
    ds0 = BinaryILSVRC12('imagenet', 'train')
    #ds1 = MultiThreadRunner(ds0, num_prefetch= 100, num_thread=1)
    LMDBSerializer.save(ds0, './imagenet/ILSVRC-train.lmdb')

if not os.path.exists("./imagenet/ILSVRC-val.lmdb"):
    ds0 = BinaryILSVRC12('imagenet', 'val')
    #ds1 = MultiThreadRunner(ds0, num_prefetch=100, num_thread=1)#MultiProcessRunnerZMQ(ds0, num_proc=1)
    LMDBSerializer.save(ds0, './imagenet/ILSVRC-val.lmdb')
