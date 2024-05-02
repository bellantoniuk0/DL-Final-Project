import os
from C3D_alt import save_c3d_alt_model_weights
from C3D_alt import C3D_altered
from score_regressor import save_score_reg_model_weights
from fc6 import save_fc6_model_weights
from dive_classifier import save_dive_classifier_model_weights
from score_regressor import ScoreRegressor
from dive_classifier import DiveClassifier
from data_processing import load_training_data, load_validation_data
import cv2
from fc6 import my_fc6
import numpy as np
import tensorflow as tf
from data_processing import process_frames, preprocess_video, preprocess_frame
from C3D import C3D, save_c3d_model_weights

# Constants
VIDEO_FILE = '../data/diving_samples_len_151_lstm/054.avi'
C3D_SAVE_PATH = 'C3D_model'
C3D_ALT_PATH = 'C3D_alt_model'
DIVE_CLASS_PATH = 'dive_class_model'
FC6_PATH = 'fc6_model'
SCORE_REG_PATH = 'score_re_model'


def action_classifier(frames, c3d_path):
    
    model_C3D = C3D()  # Create a new instance of the model
    model_C3D.load_weights(c3d_path)  # Load the saved weights

    # Prepare input frames
    frames2keep = np.linspace(0, frames.shape[0] - 1, 16, dtype=int)

    X = np.zeros((1, 16, 112, 112, 3))
    ctr = 0
    
    for i in frames2keep:
      temp_slice = frames[i, :, :, :]
      temp_slice = np.transpose(temp_slice, (1, 2, 0))
      X[0, ctr, :, :, :] = temp_slice
      ctr += 1

    # Preprocess input
    X = X * 255
    X = np.flip(X, axis=4)

    # Perform prediction
    prediction = model_C3D.predict(X)

    # Print top predictions
    top_inds = np.argsort(prediction[0])[::-1][:5]  
    print('\nTop 5:')
    print('Top inds:', top_inds)

    return top_inds[0]

def action_scoring(video):
    pass

def dataloading(vf):
    pass

def main():
    frames = preprocess_video(VIDEO_FILE, input_resize=(171, 128), H=112)

    c3d_model = C3D()
    c3d_alt_model = C3D_altered()
    dive_class_model = DiveClassifier()
    fc6_model = my_fc6()
    score_re_model = ScoreRegressor()

    # Train your model here if needed

    # Save the trained C3D model weights
    save_c3d_model_weights(c3d_model, C3D_SAVE_PATH)
    save_c3d_alt_model_weights(c3d_alt_model, C3D_ALT_PATH)
    save_dive_classifier_model_weights(dive_class_model, DIVE_CLASS_PATH)
    save_fc6_model_weights(fc6_model, FC6_PATH)
    save_score_reg_model_weights(score_re_model, SCORE_REG_PATH)

    # Now use the saved model path for classification
    action_classifier(frames, C3D_SAVE_PATH)

if __name__ == '__main__':
    main()