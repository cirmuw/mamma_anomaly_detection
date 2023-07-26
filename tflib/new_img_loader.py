"""
Load images and provide splits (train_normal, test_normal, and test_anom(alous)) as arrays

Copyright (c) 2018 Thomas Schlegl
Copyright (c) 2023 Bianca Burger    loading of gray value ranges

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""



from glob import glob
import numpy as np
import numpy.matlib 
import os
import pdb
import scipy.misc
import time
import scipy.io
from itertools import izip


give_all_patches=True   ############################################


trainset_path     =  #"path-to-folder-holding-normal-training-images"
trainset_val_path = #"path-to-folder-holding-normal-validation-images"
test_normal_path  = #"path-to-folder-holding-normal-test-images"
test_anom_path    = #"path-to-folder-holding-anom-test-images"

trainset_range_path=#"path-to-folder-holding-ranges-of-normal-training-images"
trainset_val_range_path= #"path-to-folder-holding-ranges-of-normal-validation-images"
test_normal_range_path= #"path-to-folder-holding-ranges-of-normal-test-images"
test_anom_range_path=#"path-to-folder-holding-ranges-of-anom-test-images"


def get_files(data_set): # add ranges to return
        if data_set == 'train_normal':
            return sorted(glob(os.path.join(trainset_path, "*.png"))) , sorted(glob(os.path.join(trainset_range_path, "*.mat")))
        if data_set == 'valid_normal':
            return sorted(glob(os.path.join(trainset_val_path, "*.png"))) , sorted(glob(os.path.join(trainset_val_range_path, "*.mat")))
        elif data_set == 'test_normal':
            return sorted(glob(os.path.join(test_normal_path, "*.png"))) , sorted(glob(os.path.join(test_normal_range_path, "*.mat")))
        elif data_set == 'test_anom':
            return sorted(glob(os.path.join(test_anom_path, "*.png"))) , sorted(glob(os.path.join(test_anom_range_path, "*.mat")))

def get_nr_training_samples(batch_size):
    files = glob(os.path.join(trainset_path, "*.png"))
    total_nr_samples = len(files)
    nr_training_samples = total_nr_samples - np.mod(total_nr_samples, batch_size)

    return nr_training_samples

def get_nr_samples(data_set, batch_size):
    files,ranges = get_files(data_set)
    total_nr_samples = len(files)
    if not give_all_patches:
       nr_samples = total_nr_samples - np.mod(total_nr_samples, batch_size)
    elif give_all_patches and np.mod(total_nr_samples,batch_size)>0:
       nr_samples=total_nr_samples - np.mod(total_nr_samples, batch_size)+batch_size
    elif give_all_patches and np.mod(total_nr_samples,batch_size)==0:
       nr_samples=total_nr_samples
    
    return nr_samples

def get_nr_test_samples(batch_size):
    return ( get_nr_samples('test_normal', batch_size),
             get_nr_samples('test_anom', batch_size) 
            )

def make_generator(data_set, batch_size):
    epoch_count = [1]
    def get_epoch():
        images = np.zeros((batch_size, 1, 64, 64), dtype='int32')
        mini = np.zeros((batch_size, 1, 64, 64), dtype='int32')
        maxi = np.zeros((batch_size, 1, 64, 64), dtype='int32')
        files,ranges = get_files(data_set)
        assert(len(files) > 0)

        random_state = np.random.RandomState(epoch_count[0])
        index=list(range(len(files)))
        random_state.shuffle(index)
        ranges2=[ranges[i] for i in index]
        files2=[files[i] for i in index]
        epoch_count[0] += 1
        for n, (f,r) in enumerate(zip(files2,ranges2)):
            image = scipy.misc.imread(f, mode='L')
            max=numpy.matlib.repmat(np.array(scipy.io.loadmat(r)['Max'][0][0]),64,64)
            min=numpy.matlib.repmat(np.array(scipy.io.loadmat(r)['Min'][0][0]),64,64)
            
            if np.random.rand()>=0.5:                                
                image = image[:,::-1]
            images[n % batch_size] = np.expand_dims( image, 0)
            mini[n % batch_size]=np.expand_dims(min,0)
            maxi[n % batch_size]=np.expand_dims(max,0)
            if n > 0 and n % batch_size == 0:
                yield np.copy(images),np.copy(mini),np.copy(maxi)
    return get_epoch


def make_ad_generator(data_set, batch_size):
    def get_epoch():
        patchID=np.zeros((batch_size,1),dtype='int32')
        images = np.zeros((batch_size, 1, 64, 64), dtype='int32')
        mini = np.zeros((batch_size, 1, 64, 64), dtype='int32')
        maxi = np.zeros((batch_size, 1, 64, 64), dtype='int32')
        files, ranges = get_files(data_set)
        nr_files = len(files)
        assert(nr_files > 0)

        for n, (f,r) in enumerate(zip(files,ranges)):
            image = scipy.misc.imread(f, mode='L')
            max=numpy.matlib.repmat(np.array(scipy.io.loadmat(r)['Max'][0][0]),64,64)
            min=numpy.matlib.repmat(np.array(scipy.io.loadmat(r)['Min'][0][0]),64,64)
            head, tail=os.path.split(f)
            id,ext=os.path.splitext(tail)
            
            images[n % batch_size] = np.expand_dims( image, 0)
            mini[n % batch_size]=np.expand_dims(min,0)
            maxi[n % batch_size]=np.expand_dims(max,0)
            patchID[n%batch_size]=int(id)
            
            if (n+1) % batch_size == 0:
                yield np.copy(images), np.copy(mini), np.copy(maxi), np.copy(patchID) 
                
            elif (n+1)==nr_files:
                final_btchsz = (n%batch_size)+1
                if give_all_patches ==False:
                    yield np.copy(images[:final_btchsz]), np.copy(mini[:final_btchsz]), np.copy(maxi[:final_btchsz]), np.copy(patchID[:final_btchsz])
                elif give_all_patches==True:
                   add=batch_size-final_btchsz
                   newID=nr_files+np.array(range(add))+1
                   patchID[final_btchsz:]=np.expand_dims(newID,1)
                   yield np.copy(images), np.copy(mini), np.copy(maxi), np.copy(patchID) 
    return get_epoch


def load(batch_size, run_type):
    if 'train' in run_type: 
        train_normal_img, train_normal_min, train_normal_max=izip(*make_generator('train_normal', batch_size)())
        valid_normal_img, valid_normal_min, valid_normal_max=izip(*make_generator('valid_normal', batch_size)())
        return (
            train_normal_img, train_normal_min, train_normal_max,
            valid_normal_img, valid_normal_min, valid_normal_max
        )
    elif run_type=='anomaly_score':
        test_normal_img, test_normal_min, test_normal_max,test_normal_ID=izip(*make_ad_generator('test_normal', batch_size)())
        test_anom_img, test_anom_min, test_anom_max, test_anom_ID=izip(*make_ad_generator('test_anom', batch_size)())
       
        return (
             test_normal_img, test_normal_min, test_normal_max,test_normal_ID,
             test_anom_img, test_anom_min, test_anom_max,test_anom_ID                      
        )


if __name__ == '__main__':
    train_normal_img, train_normal_min, train_normal_max,valid_normal_img, valid_normal_min, valid_normal_max = load(16, 'encoder_train')
    t0 = time.time()
    for n, batch in enumerate(train_normal_img, start=1):
        print "{}\t{}".format(str(time.time() - t0), batch[0][0,0,0,0])
        if n == 1000:
            break
        t0 = time.time()
