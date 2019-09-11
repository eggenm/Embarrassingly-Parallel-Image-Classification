import os
import numpy as np
import pandas as pd
from io import BytesIO
from PIL import Image
from pathlib import Path
import tensorflow as tf
import functools
from tensorflow.contrib.slim.nets import resnet_v1
slim = tf.contrib.slim

label_to_number_dict = {'NA':0,
                       'Not_HCSA':1,
                       'HCSA':2}

def get_nlcd_id(my_filename):
    ''' Extracts the true label  '''
    folder, _ = os.path.split(my_filename)
    return(label_to_number_dict[os.path.basename(folder)])

n_workers = 10

#############   DIRECTORIES  ################################
#############################################################

model_dir = "C:\\Users\\ME\\Dropbox\\HCSproject\\data\\PoC\\trainedModels\\tf\\models\\"
dataset_dir = "C:\\Users\\ME\\Dropbox\\HCSproject\\data\\PoC\\app_kalbar_cntk\\tiles\\balanced_validation_set\\Not_HCSA\\"


#mage_rdd = sc.binaryFiles('{}/*/*.png'.format(dataset_dir), minPartitions=n_workers).coalesce(n_workers)

##############              FUNCTIONS                  ##########################
#################################################################################

def get_network_fn(num_classes, weight_decay=0.0, is_training=False):
    arg_scope = resnet_v1.resnet_arg_scope(weight_decay=weight_decay)
    func = resnet_v1.resnet_v1_50

    @functools.wraps(func)
    def network_fn(images):
        with slim.arg_scope(arg_scope):
            return func(images, num_classes, is_training=is_training)

    if hasattr(func, 'default_image_size'):
        network_fn.default_image_size = func.default_image_size
    return (network_fn)


def mean_image_subtraction(image, means):
    num_channels = image.get_shape().as_list()[-1]
    channels = tf.split(image, num_channels, 2)
    for i in range(num_channels):
        channels[i] -= means[i]
    return (tf.concat(channels, 2))


def get_preprocessing():
    def preprocessing_fn(image, output_height=120, output_width=120):
        image = tf.expand_dims(image, 0)
        resized_image = tf.image.resize_bilinear(image, [output_height, output_width], align_corners=False)
        resized_image = tf.squeeze(resized_image)
        resized_image.set_shape([output_height, output_width, 3])

        image = tf.to_float(resized_image)
        return (mean_image_subtraction(image, [123.68, 116.78, 103.94]))

    return (preprocessing_fn)


def tf_run_worker(files):
    results = []

    with tf.Graph().as_default():
        network_fn = get_network_fn(num_classes=3, is_training=False)
        image_preprocessing_fn = get_preprocessing()

        current_image = tf.placeholder(tf.uint8, shape=(120, 120, 3))

        preprocessed_image = image_preprocessing_fn(current_image, 120, 120)
        image = tf.expand_dims(preprocessed_image, 0)
        logits, _ = network_fn(image)
        print('logits  ', logits)
        predictions = tf.argmax(logits, 1)
        print('predictions  ', predictions)

        with tf.Session() as sess:
            my_saver = tf.train.Saver()
            print(model_dir)
            my_saver.restore(sess, tf.train.latest_checkpoint(model_dir))

            coord = tf.train.Coordinator()
            print(files)
            try:
                for file in files:
                    print(file)
                    imported_image_np = np.asarray(Image.open(file), dtype=np.uint8)
                    #print(imported_image_np)
                    result = sess.run(predictions, feed_dict={current_image: imported_image_np})
                    #true_label = get_nlcd_id(file[0])
                    print("TRUE  " , 'HCSA  ', "   RESULT   ", result[0])
                    results.append([file, 'HCSA', result[0]])
                    #break
            finally:
                coord.request_stop()
    return (results)

basepath = Path(dataset_dir)
files_in_basepath = (entry for entry in basepath.iterdir() if entry.is_file())
if files_in_basepath:
    print('files_in_basepath:  ', files_in_basepath)
tf_run_worker(files_in_basepath);

#files_in_basepath.tf_run_worker();