import os
from C3D_alt import save_c3d_alt_model_weights
from C3D_alt import C3D_altered
from fc6 import save_fc6_model_weights
from score_regressor import save_score_reg_model_weights
from dive_classifier import save_dive_classifier_model_weights
from score_regressor import ScoreRegressor
from dive_classifier import DiveClassifier
import cv2
from fc6 import my_fc6
import numpy as np
import tensorflow as tf
from data_processing import process_frames, preprocess_video, preprocess_frame
from C3D import C3D, save_c3d_model_weights

# Constants
<<<<<<< HEAD
VIDEO_FILE = '../data/diving_samples_len_151_lstm' #all videos
=======
VIDEO_TRAIN_FOLDER = '../data/diving_samples_training' #rn just one video but needs to be all videos 
VIDEO_TEST_FOLDER = '../data/diving_samples_testing'
>>>>>>> 2f97ae64627251843a58868008933943cf6b13e3
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

def action_scoring(video, c3d_alt_path, fc6_path, score_reg_path):
    # Load the altered C3D backbone
    model_CNN = C3D_altered()
    model_CNN.load_weights(c3d_alt_path)  # Load the saved weights
    
    # Load the fc6 layer
    model_my_fc6 = my_fc6()
    model_my_fc6.load_weights(fc6_path)  # Load the saved weights
    
    # Load the score regressor
    model_score_regressor = ScoreRegressor()
    model_score_regressor.load_weights(score_reg_path)
    
    # Process video frames
    clip_feats = []
    print('len of video: ', len(video))
    print('video shape: ', video.shape)
    for i in np.arange(0, video.shape[2], 16):
        print('i: ', i)
        clip = video[:, :, i:i + 16, :, :]
        clip_feats_temp = model_CNN(clip)
        clip_feats.append(clip_feats_temp.numpy())
    
    clip_feats_avg = np.mean(clip_feats, axis=0)
    
    sample_feats_fc6 = model_my_fc6(clip_feats_avg)
    temp_final_score = model_score_regressor(sample_feats_fc6) * 17
    
    pred_scores = temp_final_score.numpy().flatten().tolist()

    return pred_scores

def load_training_data():
    training_data = []

    # Loop through all the video files in the folder
    for video_file in os.listdir(VIDEO_TRAIN_FOLDER):
        # Process the video file
        video_file_path = os.path.join(VIDEO_TRAIN_FOLDER, video_file)
        print(f"Processing video file: {video_file_path}")

        with open(video_file_path, 'rb') as video_file:
            frames = preprocess_video(video_file, input_resize=(171, 128), H=112)
            training_data.append(frames)

    return training_data

def load_validation_data():
    test_data = []

    # Loop through all the video files in the folder
    for video_file in os.listdir(VIDEO_TEST_FOLDER):
        # Process the video file
        video_file_path = os.path.join(VIDEO_TEST_FOLDER, video_file)
        print(f"Processing video file: {video_file_path}")

        with open(video_file_path, 'rb') as video_file:
            frames = preprocess_video(video_file, input_resize=(171, 128), H=112)
            test_data.append(frames)
    return test_data


# Want to have a for loop, hand split into train/test
# split 300 train 70 test, take 300 and iterate over those
# then check against 70

def main():
    # #this needs to be changed to be all video files 
    # frames = preprocess_video(VIDEO_FILE, input_resize=(171, 128), H=112)
    # video = preprocess_video(VIDEO_FILE, input_resize=(171, 128), H=112)
    load_validation_data()

    c3d_model = C3D()
    #TRAIN HERE 
    c3d_alt_model = C3D_altered()
    #TRAIN HERE 
    dive_class_model = DiveClassifier()
    #TRAIN HERE
    fc6_model = my_fc6()
    #TRAIN HERE
    score_re_model = ScoreRegressor()
    #TRAIN HERE 


    # Save the trained C3D model weights
    save_c3d_model_weights(c3d_model, C3D_SAVE_PATH)
    save_c3d_alt_model_weights(c3d_alt_model, C3D_ALT_PATH)
    save_dive_classifier_model_weights(dive_class_model, DIVE_CLASS_PATH)
    save_fc6_model_weights(fc6_model, FC6_PATH)
    save_score_reg_model_weights(score_re_model, SCORE_REG_PATH)

    # Now use the saved model path for classification
    action_classifier(frames, C3D_SAVE_PATH)
    # run action scorring 
    action_scoring(video, C3D_ALT_PATH, FC6_PATH, SCORE_REG_PATH)

    #SUM AND PRINT FINAL ACTION SCORE

if __name__ == '__main__':
    main()
