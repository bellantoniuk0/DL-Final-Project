import os
import cv2
import numpy as np
import tensorflow as tf
from data_processing import process_frames, preprocess_video, preprocess_frame
from C3D import C3D, save_c3d_model_weights

# Constants
VIDEO_FILE = '../data/diving_samples_len_151_lstm/054.avi'
C3D_SAVE_PATH = 'C3D_model'


# Load models
def action_classifier(frames, c3d_path):
    print('Loading C3D model from:', c3d_path)
    
    model_C3D = C3D()  # Create a new instance of the model
    model_C3D.load_weights(c3d_path)  # Load the saved weights

    print('Frames shape in action_classifier:', frames.shape)

    # Prepare input frames
    frames2keep = np.linspace(0, frames.shape[2] - 1, 16, dtype=int)
    print('frames2keep:', frames2keep)
    print('Length of frames2keep:', len(frames2keep))  # New print statement

    X = np.zeros((1, 16, 112, 112, 3))
    ctr = 0

    print('X shape just before loop:', X.shape)
    
    for i in frames2keep:
        print('Inside loop - Value of i:', i)
        temp_slice = frames[:, :, i, :, :]
        print('Inside loop - Shape of temp_slice:', temp_slice.shape)
        X[0, ctr, :, :, :] = temp_slice
        ctr += 1

    print('X shape after loop:', X.shape)

    # Preprocess input
    X = X * 255
    X = np.flip(X, axis=4)
    print('X shape after preprocessing:', X.shape)

    # Perform prediction
    prediction = model_C3D.predict(X)

    # Print top predictions
    top_inds = np.argsort(prediction[0])[::-1][:5]  # Reverse sort and take five largest items
    print('\nTop 5:')
    print('Top inds:', top_inds)

    return top_inds[0]

def main():
    frames = preprocess_video(VIDEO_FILE, input_resize=(171, 128), H=112)
    print('Frames shape:', frames.shape)

    # Train or load and train your C3D model before saving
    # Assuming you have already instantiated and trained your C3D model
    c3d_model = C3D()
    # Train your model here if needed

    # Save the trained C3D model weights
    save_c3d_model_weights(c3d_model, C3D_SAVE_PATH)

    # Now use the saved model path for classification
    action_classifier(frames, C3D_SAVE_PATH)

if __name__ == '__main__':
    main()