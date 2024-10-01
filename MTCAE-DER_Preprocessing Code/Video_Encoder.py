import os, warnings
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf

from decord import VideoReader
from tensorflow import keras
from keras import layers

input_size = 224
num_frame = 16
sampling_rate = 1
batch_size = 16

def create_uc_dataframe(path):
    data = []
    label2id = {label:i for i, label in enumerate(class_uc_folders)}

    for class_name in os.listdir(path):
        class_dir = os.path.join(path, class_name)
        for video_file in os.listdir(class_dir):
            video_path = os.path.join(class_dir, video_file)
            data.append({
                'video_path': os.path.abspath(video_path),
                'label': label2id[class_name],
                'class_name': class_name
            })

    df = pd.DataFrame(data)
    df = df.sample(frac=1).reset_index(drop=True)
    return df

def read_video(file_path):
    vr = VideoReader(file_path.numpy().decode('utf-8'))
    frames = vr.get_batch(range(len(vr))).asnumpy()
    return format_frames(
        frames,
        output_size=(input_size, input_size)
    )

def format_frames(frame, output_size):
    frame = tf.image.convert_image_dtype(frame, tf.uint8)
    frame = tf.image.resize(frame, size=list(output_size))
    return frame

def load_video(file_path, label):
    video = tf.py_function(func=read_video, inp=[file_path], Tout=tf.float32)
    video.set_shape([None, None, None, 3])
    return video, tf.cast(label, dtype=tf.float32)

def uniform_temporal_subsample(
    x, num_samples, clip_idx, total_clips, frame_rate=1, temporal_dim=-4
):
    t = tf.shape(x)[temporal_dim]
    max_offset = t - num_samples * frame_rate
    step = max_offset // total_clips
    offset = clip_idx * step
    indices = tf.linspace(
        tf.cast(offset, tf.float32),
        tf.cast(offset + (num_samples-1) * frame_rate, tf.float32),
        num_samples
    )
    indices = tf.clip_by_value(indices, 0, tf.cast(t - 1, tf.float32))
    indices = tf.cast(tf.round(indices), tf.int32)
    return tf.gather(x, indices, axis=temporal_dim)


def clip_generator(
    image, num_frames=32, frame_rate=1, num_clips=1, crop_size=224
):
    clips_list = []
    for i in range(num_clips):
        frame = uniform_temporal_subsample(
            image, num_frames, i, num_clips, frame_rate=frame_rate, temporal_dim=0
        )
        clips_list.append(frame)

    video = tf.stack(clips_list)
    video = tf.reshape(
        video, [num_clips*num_frames, crop_size, crop_size, 3]
    )
    return video

processing_model = keras.Sequential(
    [
        layers.Rescaling(scale=1./255.),
        layers.Normalization(
            mean=[0.485, 0.456, 0.406],
            variance=[np.square(0.225), np.square(0.225), np.square(0.225)]
        )
    ]
)

def create_dataloader(df, batch_size, shuffle=True, drop_reminder=True):
    ds = tf.data.Dataset.from_tensor_slices(
        (df['video_path'].values, df['label'].values)
    )
    ds = ds.repeat()
    ds = ds.shuffle(8 * batch_size) if shuffle else ds
    ds = ds.map(load_video, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.map(
        lambda x, y: (clip_generator(x, num_frame, sampling_rate, num_clips=1), y),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    ds = ds.batch(batch_size, drop_remainder=drop_reminder)
    ds = ds.map(lambda x, y: (processing_model(x), y), num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds

def read_video_inference(file_path):
    vr = VideoReader(file_path)
    frames = vr.get_batch(range(len(vr))).asnumpy()
    return format_frames(
        frames,
        output_size=(input_size, input_size)
    )

if __name__ == "__main__":

    finetune_videomae = keras.models.load_model(
        f'Models\\TFVideoMAE_L_K400_16x224_FT', compile=False
    )
    finetune_videomae.trainable = False

    EncoderNet = keras.Sequential(name="TFVideoMAE_Encoder")
    for layer in finetune_videomae.layers[:-3]:
        EncoderNet.add(layer)

    train_path = 'Data\\RAVDESS\\RAVDESS_16F\\Video\\train'

    class_uc_folders = os.listdir(train_path)

    train_df = create_uc_dataframe(train_path)
    train_ds = create_dataloader(train_df, batch_size, shuffle=True)

    video_path = 'Data\\RAVDESS\\RAVDESS_16F\\Video\\test\\Angry\\01-01-05-02-01-01-01.mp4'
    sample_ds = read_video_inference(video_path)
    sample_ds = clip_generator(sample_ds, num_frame, sampling_rate, num_clips=1)
    sample_ds = processing_model(sample_ds)
    # print(sample_ds.shape)

    y_pred = EncoderNet(sample_ds, training=False)
    y_pred = np.array(y_pred)
    y_pred = np.matrix(y_pred)
    # print(y_pred, y_pred.shape)

    outputs_path = 'Output.txt'
    np.savetxt(outputs_path, y_pred)

    y = np.loadtxt(outputs_path)
    print(y, y.shape)
