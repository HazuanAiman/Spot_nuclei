# -*- coding: utf-8 -*-
"""
Created on Fri Apr 15 9:32:16 2022

@author: User
"""

#https://www.kaggle.com/competitions/data-science-bowl-2018/data
import tensorflow as tf
from tensorflow_examples.models.pix2pix import pix2pix
from IPython.display import clear_output
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np
import cv2
import os

#prepare a list for image and mask
images = []
masks = []
train_dir = r"C:\Users\User\.keras\datasets\data-science-bowl-2018-2\train"
test_dir = r"C:\Users\User\.keras\datasets\data-science-bowl-2018-2\test"

#%%
#load the image
def load_images(directory):
    images = []

    for image_file in os.listdir(directory):
        img = cv2.imread(os.path.join(directory, image_file))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (128,128))
        images.append(img)
    return images

train_image_dir = os.path.join(train_dir,'inputs')
train_images = load_images(train_image_dir)
test_image_dir = os.path.join(test_dir, 'inputs')
test_images = load_images(test_image_dir)

#load the masks
def load_masks(directory):
    masks = []
    for mask_file in os.listdir(directory):
        mask = cv2.imread(os.path.join(directory, mask_file), cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (128,128))
        masks.append(mask)
    return masks
    
train_mask_dir = os.path.join(train_dir,'masks')
train_masks = load_masks(train_mask_dir)
test_mask_dir = os.path.join(test_dir, 'masks')
test_masks = load_masks(test_mask_dir)

#%%
#convert images and masks list into numpy array
train_images_np = np.array(train_images)
train_masks_np = np.array(train_masks)
test_images_np = np.array(test_images)
test_masks_np = np.array(test_masks)

#%%
#Check some examples on train images and masks
plt.figure(figsize=(10,4))
for i in range(1,4):
    plt.subplot(1,3,i)
    img_plot = train_images[i]
    plt.imshow(img_plot)
    plt.axis('off')
plt.show()

plt.figure(figsize=(10,4))
for i in range(1,4):
    plt.subplot(1,3,i)
    mask_plot = train_masks[i]
    plt.imshow(mask_plot, cmap='gray')
    plt.axis('off')
plt.show()

#%%
#Data preprocessing
#expand the mask dimension
train_masks_np_exp = np.expand_dims(train_masks_np, axis=-1)
test_masks_np_exp = np.expand_dims(test_masks_np, axis=-1)

#check the mask output
print(train_masks[0].min(), train_masks[0].max())

#%%
#change the mask value (1.Normalize the value, 2.Encode into numerical encoding)
converted_train_masks = np.ceil(train_masks_np_exp/255)
converted_train_masks = 1 - converted_train_masks
converted_test_masks = np.ceil(test_masks_np_exp/255)
converted_test_masks = 1 - converted_test_masks

#%%
#normalize the images
converted_train_images = train_images_np/255.0
converted_test_images = test_images_np/255.0


#%%
#do train-test split
x_train, x_val, y_train, y_val = train_test_split(converted_train_images, converted_train_masks, test_size=0.2, random_state=12345)

#%%
#convert the numpy array data into tensor slice
train_x = tf.data.Dataset.from_tensor_slices(x_train)
val_x = tf.data.Dataset.from_tensor_slices(x_val)
test_x = tf.data.Dataset.from_tensor_slices(converted_test_images)
train_y = tf.data.Dataset.from_tensor_slices(y_train)
val_y = tf.data.Dataset.from_tensor_slices(y_val)
test_y = tf.data.Dataset.from_tensor_slices(converted_test_masks)

#%%
#zip tensor slice into dataset
train = tf.data.Dataset.zip((train_x, train_y))
val = tf.data.Dataset.zip((val_x, val_y))
test = tf.data.Dataset.zip((test_x, test_y))

#%%
#convert into prefetch dataset
BATCH_SIZE = 16
AUTOTUNE = tf.data.AUTOTUNE
BUFFER_SIZE = 1000
STEPS_PER_EPOCH = 800//BATCH_SIZE
VALIDATION_STEPS = 200//BATCH_SIZE
train = train.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()
train = train.prefetch(buffer_size=AUTOTUNE)
val = val.batch(BATCH_SIZE).repeat()
val = val.prefetch(buffer_size=AUTOTUNE)
test = test.batch(BATCH_SIZE).prefetch(buffer_size=AUTOTUNE)

#%%
#Prepare model
#creating a modified version of U-net
base_model = tf.keras.applications.MobileNetV2(input_shape=[128,128,3], include_top=False)

#use the activations of those layers
layer_names = [
    'block_1_expand_relu',   # 64x64
    'block_3_expand_relu',   # 32x32
    'block_6_expand_relu',   # 16x16
    'block_13_expand_relu',  # 8x8
    'block_16_project',      # 4x4
]

base_model_outputs = [base_model.get_layer(name).output for name in layer_names]

#Create the feature extraction model
down_stack = tf.keras.Model(inputs=base_model.input, outputs=base_model_outputs)

down_stack.trainable = False

#Define the upsampling stack
up_stack = [
    pix2pix.upsample(512, 3),  # 4x4 -> 8x8
    pix2pix.upsample(256, 3),  # 8x8 -> 16x16
    pix2pix.upsample(128, 3),  # 16x16 -> 32x32
    pix2pix.upsample(64, 3),   # 32x32 -> 64x64
]

#Function to create the entire modified U-net
def unet_model(output_channels:int):
  inputs = tf.keras.layers.Input(shape=[128, 128, 3])

  #Downsampling through the model
  skips = down_stack(inputs)
  x = skips[-1]
  skips = reversed(skips[:-1])

  #Upsampling and establishing the skip connections
  for up, skip in zip(up_stack, skips):
    x = up(x)
    concat = tf.keras.layers.Concatenate()
    x = concat([x, skip])

  #This is the last layer of the model
  last = tf.keras.layers.Conv2DTranspose(
      filters=output_channels, kernel_size=3, strides=2,
      padding='same')  #64x64 -> 128x128

  x = last(x)

  return tf.keras.Model(inputs=inputs, outputs=x)

#Define the model
OUTPUT_CLASSES = 2
model = unet_model(output_channels=OUTPUT_CLASSES)

#%%
#Compile the model and display the model structure
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
model.summary()

#%%
#Create a function to display some examples
def display(display_list):
    plt.figure(figsize=(15,15))
    title = ['Input Image','True Mask','Predicted Mask']
    
    for i in range(len(display_list)):
        plt.subplot(1, len(display_list),i+1)
        plt.title(title[i])
        plt.imshow(tf.keras.utils.array_to_img(display_list[i]))
        plt.axis('off')
        
    plt.show()
    
for images, masks in train.take(2):
    sample_image, sample_mask = images[0], masks[0]
    display([sample_image, sample_mask])
    
#%%
#Create a function to process predicted mask
def create_mask(pred_mask):
    pred_mask = tf.argmax(pred_mask,axis=-1)
    pred_mask = pred_mask[..., tf.newaxis]
    return pred_mask

#Create a function to display prediction
def show_predictions(dataset=None,num=1):
    if dataset:
        for image, mask in dataset.take(num):
            pred_mask = model.predict(image)
            display([image[0],mask[0],create_mask(pred_mask)[0]])
    else:
        display([sample_image,sample_mask,create_mask(model.predict(sample_image[tf.newaxis,...]))[0]])

#Custom callback to display result during training
class DisplayCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        clear_output(wait=True)
        show_predictions()
        print('\n Sample prediction after epoch {}\n'.format(epoch+1))

#Tensorboard callback
import datetime
log_dir = r'X:\Users\User\Tensorflow Deep Learning\Tensorboard\databowl_' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") 
tb_callback = tf.keras.callbacks.TensorBoard(log_dir,histogram_freq=1,profile_batch=0)
es_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience=5,verbose=1)

#Start to do training
EPOCH = 100

history = model.fit(train,epochs=EPOCH,steps_per_epoch=STEPS_PER_EPOCH,
                    validation_steps=VALIDATION_STEPS,
                    validation_data=val,
                    callbacks=[DisplayCallback(),tb_callback,es_callback])

#%%
#Deploy model by using the show_prediction functions created before
show_predictions(test,3)