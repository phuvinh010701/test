from tensorflow.keras.utils import Sequence, load_img, img_to_array
import numpy as np
import tensorflow as tf
import imgaug.augmenters as iaa

seq = iaa.Sequential([
    iaa.Fliplr(0.5),
    iaa.Crop(percent=(0, 0.1)),
])


def age_to_class(nums):
    return np.floor(int(nums) / 10)

class DataGenerator(Sequence):
    def __init__(self,
                 folder_images,
                 file_label,
                 batch_size=32,
                 dim=(96, 96, 3),
                 n_channels=3,
                 gender_classes=1,
                 age_classes=9,
                 train=True,
                 shuffle=True):

        self.dim = dim
        self.folder_images = folder_images
        self.batch_size = batch_size
        self.n_channels = n_channels
        self.gender_classes = gender_classes
        self.age_classes = age_classes
        self.path_imgs = []
        self.genders = []
        self.ages = []
        self.train=train
        
        with open(file_label) as f:
            lines = f.readlines()
        f.close()

        for line in lines:
            data = line.strip().split(' ')
            path = data[0]
            age = path[6:8]
            age = age_to_class(age)
            gender = data[1]

            self.path_imgs.append(folder_images + path)
            self.genders.append(int(gender))
            self.ages.append(age)

        self.shuffle = shuffle
        self.img_indexes = np.arange(len(self.path_imgs))
        self.on_epoch_end()
        
    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.img_indexes) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temps = [self.img_indexes[k] for k in indexes]

        # Generate data
        X, y_gender = self.__data_generation(list_IDs_temps)
        # print(X.shape)
        # print(y_gender.shape)
        # print(y_age.shape)
        return X, tf.keras.utils.to_categorical(y_gender, num_classes=9, dtype='float32')
    
    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.path_imgs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
        
    def __data_generation(self, list_IDs_temps):
        X = np.empty((self.batch_size, *self.dim))
        y_gender = []
        y_age = []
        for i, ID in enumerate(list_IDs_temps):
            img = load_img(self.path_imgs[ID], target_size=self.dim[:2])
            
            img = img_to_array(img)
            X[i,] = img
            # X = (X/255).astype('float32')
            y_gender.append(self.genders[ID])
            y_age.append(self.ages[ID])

        # X = X[:,:,:,:,np.newaxis]
        return X, np.array(y_age)

