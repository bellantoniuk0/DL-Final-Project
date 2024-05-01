import cv2
import numpy as np
import tempfile
import tensorflow as tf
from data_processing import process_frames, preprocess_video, preprocess_frame
import hyp
import os

#load models

def action_classifier(frames, c3d_path):
    print('Loading C3D model from: ', c3d_path)
    model_C3D = tf.keras.models.load_model(c3d_path)
    # Prepare input frames
    frames2keep = np.linspace(0, frames.shape[2] - 1, 16, dtype=int)
    X = np.zeros((1, 16, 112, 112, 3))
    ctr = 0
    for i in frames2keep:
        X[0, ctr, :, :, :] = frames[:, :, i, :, :]
        ctr += 1
    print('X shape: ', X.shape)

    # Preprocess input
    X = X * 255
    X = np.flip(X, axis=4) 

    # Perform prediction
    prediction = model_C3D.predict(X)

    # Print top predictions
    top_inds = np.argsort(prediction[0])[::-1][:5]  # Reverse sort and take five largest items
    print('\nTop 5:')
    print('Top inds: ', top_inds)
    
    return top_inds[0]

#run c3d on the frames on the trained classified model
#save output

#run the altered

#make predictions and run the regressor 

def main():
    video_file = '../data/diving_samples_len_151_lstm/054.avi'
    c3d_path = '/Users/student/Desktop/1470/DL-Final-Project/models/C3D.py'
    frames = preprocess_video(video_file, input_resize=(171, 128), H=112)
    print('Frames shape: ', frames.shape)
    action_classifier(frames, c3d_path)

if __name__ == '__main__':
    main()
