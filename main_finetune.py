import sys
import argparse

from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_addons as tfa
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger
import atexit

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
import glob
import random

from scheduler import WarmUpCosine

import tfimm
from models_vit import *


def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--batch_size', default=64, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=50, type=int)

    # Model parameters
    parser.add_argument('--input_size', default=224, type=int,
                        help='images input size')

    # Optimizer parameters
    parser.add_argument('--lr', type=float, default=1e-4, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--warmup_epoch_percentage', type=int, default=.1, metavar='N',
                        help='epochs to warmup LR')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='weight decay (default: 1e-4)')

    # Dataset parameters
    parser.add_argument('--data_path', default='/data/yuka/projects/RETFound_MAE/data/', type=str,
                        help='dataset path')
    parser.add_argument('--nb_classes', default=1000, type=int,
                        help='number of the classification types')

    parser.add_argument('--output_dir', default='./output_dir/',
                        help='path where to save, empty for no saving')

    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')
    parser.add_argument('--eval', action='store_true',
                        help='Perform evaluation only')

    # * Cutmix params
    parser.add_argument('--cutmix', type=float, default=0.,
                        help='cutmix alpha, cutmix enabled if > 0.')

    # * Finetuning params
    parser.add_argument('--finetune', default='',type=str,
                        help='finetune from checkpoint')
    parser.add_argument('--task', default='',type=str,
                        help='finetune from checkpoint')
    parser.add_argument('--global_pool', action='store_true')
    parser.set_defaults(global_pool=True)
    parser.add_argument('--cls_token', action='store_false', dest='global_pool',
                        help='Use class token instead of global pool for classification')

    opt = parser.parse_args()

    return opt

### Prepare datasets
def prepare_data(dir, is_train=True):

    ds = tf.keras.utils.image_dataset_from_directory(
        dir,
        label_mode='categorical',
        labels="inferred",
        color_mode='rgb',
        shuffle=is_train,
        seed=SEED,
        image_size=(IMAGE_SIZE, IMAGE_SIZE),
        batch_size=BATCH_SIZE,
    )
    normalization_layer = tf.keras.layers.Rescaling(1./255)

    if CUTMIX_ALPHA > 0.:
        proc_ds_one = ds.map(
            lambda x, y: (normalization_layer(x), y),
            num_parallel_calls=AUTO,
        )
        proc_ds_two = ds.map(
            lambda x, y: (normalization_layer(x), y),
            num_parallel_calls=AUTO,
        )

        # Combine two shuffled datasets from the same training data.
        train_ds = tf.data.Dataset.zip((proc_ds_one, proc_ds_two))
        proc_ds = train_ds.map(
            cutmix,
            num_parallel_calls=AUTO,
        )
    else:
        proc_ds = ds.map(
            lambda x, y: (normalization_layer(x), y),
            num_parallel_calls=AUTO,
        )

    return proc_ds.prefetch(AUTO)




def sample_beta_distribution(size, concentration=0.2):
    gamma_1_sample = tf.random.gamma(shape=[size], alpha=concentration)
    gamma_2_sample = tf.random.gamma(shape=[size], alpha=concentration)
    return gamma_1_sample / (gamma_1_sample + gamma_2_sample)

@tf.function
def get_box(lambda_value):
    cut_rat = tf.math.sqrt(1.0 - lambda_value)

    cut_w = IMAGE_SIZE * cut_rat  # rw
    cut_w = tf.cast(cut_w, tf.int32)

    cut_h = IMAGE_SIZE * cut_rat  # rh
    cut_h = tf.cast(cut_h, tf.int32)

    cut_x = tf.random.uniform((1,), minval=0, maxval=IMAGE_SIZE, dtype=tf.int32)  # rx
    cut_y = tf.random.uniform((1,), minval=0, maxval=IMAGE_SIZE, dtype=tf.int32)  # ry

    boundaryx1 = tf.clip_by_value(cut_x[0] - cut_w // 2, 0, IMAGE_SIZE)
    boundaryy1 = tf.clip_by_value(cut_y[0] - cut_h // 2, 0, IMAGE_SIZE)
    bbx2 = tf.clip_by_value(cut_x[0] + cut_w // 2, 0, IMAGE_SIZE)
    bby2 = tf.clip_by_value(cut_y[0] + cut_h // 2, 0, IMAGE_SIZE)

    target_h = bby2 - boundaryy1
    if target_h == 0:
        target_h += 1

    target_w = bbx2 - boundaryx1
    if target_w == 0:
        target_w += 1

    return boundaryx1, boundaryy1, target_h, target_w


@tf.function
def cutmix(train_ds_one, train_ds_two):
    ### Cutmix augmentation
    ### implementation from https://keras.io/examples/vision/cutmix/
    (image1, label1), (image2, label2) = train_ds_one, train_ds_two
    label1 = tf.cast(label1, tf.float32)
    label2 = tf.cast(label2, tf.float32)

    # Get a sample from the Beta distribution
    lambda_value = sample_beta_distribution(1, [CUTMIX_ALPHA])

    # Define Lambda
    lambda_value = lambda_value[0][0]

    # Get the bounding box offsets, heights and widths
    boundaryx1, boundaryy1, target_h, target_w = get_box(lambda_value)

    # Get a patch from the second image (`image2`)
    crop2 = tf.image.crop_to_bounding_box(
        image2, boundaryy1, boundaryx1, target_h, target_w
    )
    # Pad the `image2` patch (`crop2`) with the same offset
    image2 = tf.image.pad_to_bounding_box(
        crop2, boundaryy1, boundaryx1, IMAGE_SIZE, IMAGE_SIZE
    )
    # Get a patch from the first image (`image1`)
    crop1 = tf.image.crop_to_bounding_box(
        image1, boundaryy1, boundaryx1, target_h, target_w
    )
    # Pad the `image1` patch (`crop1`) with the same offset
    img1 = tf.image.pad_to_bounding_box(
        crop1, boundaryy1, boundaryx1, IMAGE_SIZE, IMAGE_SIZE
    )

    # Modify the first image by subtracting the patch from `image1`
    # (before applying the `image2` patch)
    image1 = image1 - img1
    # Add the modified `image1` and `image2`  together to get the CutMix image
    image = image1 + image2

    # Adjust Lambda in accordance to the pixel ration
    lambda_value = 1 - (target_w * target_h) / (IMAGE_SIZE * IMAGE_SIZE)
    lambda_value = tf.cast(lambda_value, tf.float32)

    # Combine the labels of both images
    label = lambda_value * label1 + (1 - lambda_value) * label2
    return image, label

# For computing some classification metrics
def eval(keras_model, val_ds, task):

    if not os.path.exists(task):
        os.makedirs(task)

    # Create x and y tensors
    x_valid = None
    y_valid = None

    for x, y in iter(val_ds):
        if x_valid is None:
            x_valid = x.numpy()
            y_valid = y.numpy()
        else:
            x_valid = np.concatenate((x_valid, x.numpy()), axis=0)
            y_valid = np.concatenate((y_valid, y.numpy()), axis=0)

    # Generate predictions
    y_pred = keras_model.predict(val_ds)

    # Calculate confusion matrix
    confusion_mtx = tf.math.confusion_matrix(
        np.argmax(y_valid, axis=1), np.argmax(y_pred, axis=1)
    )
    confusion_mtx = confusion_mtx.numpy()
    # Plot the confusion matrix
    class_names = [
        "Double Layer Sign",
        "Drusen"
        ]
    plt.figure(figsize=(10, 8))
    confusion_mtx_norm = confusion_mtx/(1.*np.array([np.sum(confusion_mtx, axis=1)]).T)
    plt.imshow(confusion_mtx_norm, cmap=plt.cm.Blues)

    for i in range(confusion_mtx.shape[1]):
        for j in range(confusion_mtx.shape[0]):
            c = confusion_mtx_norm[j,i]
            plt.text(i, j, '%.2f'%c, va='center', ha='center', fontsize=16)
    plt.xlabel("Actual", fontsize=16)
    plt.ylabel("Prediction", fontsize=16)
    plt.title("Validation Confusion Matrix", fontsize=16)
    plt.savefig("%s/confusion_matrix.png"%task)

    for i, label in enumerate(class_names):
        precision = confusion_mtx[i, i] / np.sum(confusion_mtx[:, i])
        recall = confusion_mtx[i, i] / np.sum(confusion_mtx[i, :])
        print(
            "{0:15} Precision:{1:.2f}%; Recall:{2:.2f}%".format(
                label, precision * 100, recall * 100
            )
        )



def main(opt):

    ### Prepare datasets
    train_ds = prepare_data(DS_TRAIN_PATH)
    val_ds = prepare_data(DS_VALID_PATH, is_train=False)
    num_batches = tf.data.experimental.cardinality(train_ds)
    num_samples = num_batches.numpy() * BATCH_SIZE

    ### Set optimization parameters
    steps = int((num_samples // BATCH_SIZE) * opt.epochs)
    warmup_steps = int(steps * opt.warmup_epoch_percentage)
    scheduled_lrs = WarmUpCosine(
        learning_rate_base=opt.lr,
        total_steps=steps,
        warmup_learning_rate=0.0,
        warmup_steps=warmup_steps,
    )

    strategy = tf.distribute.MirroredStrategy()
    print('Number of devices: {}'.format(strategy.num_replicas_in_sync))
    # Open a strategy scope.
    with strategy.scope():
        # Define the Keras model
        if opt.global_pool:
            keras_model = tfimm.create_model( # apply global pooling withoug class token
                "vit_large_patch16_224_mae",
                nb_classes = opt.nb_classes
                )
        else:
            keras_model = tfimm.create_model( # Use class token
                "vit_large_patch16_224",
                nb_classes = opt.nb_classes
                )
        # Load weights (should load before adding augmentation layers)
        if not opt.eval:
            if opt.finetune:
                keras_model.load_weights(opt.finetune, skip_mismatch=True, by_name=True)
                print("Load pre-trained checkpoint from: %s" %opt.finetune)
        if opt.cutmix == 0.:
            # add simple augmentation layers (should be after loading weights)
            data_augmentation = keras.Sequential(
                [tf.keras.layers.RandomFlip("horizontal"),
               tf.keras.layers.RandomRotation(0.2),
               tf.keras.layers.RandomCrop(IMAGE_SIZE, IMAGE_SIZE),
                ])
            inputs = tf.keras.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3))
            x = data_augmentation(inputs)
            outputs = keras_model(x)
            keras_model = keras.Model(inputs, outputs)

        optimizer = tfa.optimizers.AdamW(learning_rate=scheduled_lrs, weight_decay=opt.weight_decay)
        loss_func = keras.losses.CategoricalCrossentropy(from_logits=True)
        checkpoint = ModelCheckpoint('%s/model-{epoch:03d}-{accuracy:03f}-{val_accuracy:03f}.h5'%(opt.output_dir+opt.task),
            verbose=1,
            monitor='val_loss',
            save_best_only=True,
            save_weights_only=True,
            mode='auto'
        )
        csv_logger = CSVLogger('%s/log.csv'%(opt.output_dir+opt.task))
        if not opt.eval:
            keras_model.summary()
            keras_model.compile(
                optimizer=optimizer, loss=loss_func, metrics=["accuracy"]
            )
        else:
            checkpoints = sorted(glob.glob('%s/*.h5'%(opt.output_dir+opt.resume))) # the last one is the best one
            print("Load pre-trained checkpoint from %s."%checkpoints[-1])
            keras_model.load_weights(checkpoints[-1])
            keras_model.compile(optimizer=optimizer, loss=loss_func, metrics=["categorical_accuracy"])

    if not opt.eval:
        keras_model.fit(train_ds, validation_data=val_ds, epochs=opt.epochs, callbacks=[checkpoint, csv_logger])
    else:
        loss, accuracy = keras_model.evaluate(val_ds)
        print(f"Accuracy on the test set: {accuracy}%.")
        eval(keras_model, val_ds, opt.output_dir+opt.task) # compute confusion matrix, precision/recall, etc.

    atexit.register(strategy._extended._collective_ops._pool.close) # Close multiprocessing ThreadPool for MirroredStrategy (this is necessary some combination of TF and python (e.g. TF2.7 with Py3.9))



if __name__ == '__main__':

    opt = parse_option()

    # Setting seeds for reproducibility.
    SEED = 42
    tf.keras.utils.set_random_seed(SEED)

    # Define hyperparameters
    AUTO = tf.data.experimental.AUTOTUNE
    DS_TRAIN_PATH = opt.data_path+'/train'
    DS_VALID_PATH = opt.data_path+'/val'
    IMAGE_SIZE = opt.input_size
    BATCH_SIZE = opt.batch_size
    CUTMIX_ALPHA = opt.cutmix

    if not os.path.isdir(opt.output_dir+opt.task):
        os.makedirs(opt.output_dir+opt.task)


    main(opt)
