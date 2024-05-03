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
import matplotlib.pyplot as plt

# Constants
VIDEO_TRAIN_FOLDER = '../data/diving_samples_training' #rn just one video but needs to be all videos 
VIDEO_TEST_FOLDER = '../data/diving_samples_testing'
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
    print('want to have top 5 here') 
    # print('\nTop 5:')
    # print('Top inds:', top_inds)

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
    print('want to have len and shape of vid here')
    # print('len of video: ', len(video))
    # print('video shape: ', video.shape)
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
    labels = []
    label_to_int = {}  # Dictionary to map labels to integers
    current_label = 0  # Start assigning labels from 0

    for video_file in os.listdir(VIDEO_TRAIN_FOLDER):
        video_file_path = os.path.join(VIDEO_TRAIN_FOLDER, video_file)
        #print(f"Processing video file: {video_file_path}")

        with open(video_file_path, 'rb') as file:
            frames = preprocess_video(file, input_resize=(171, 128), H=112)
            training_data.append(frames)

            # If this label has not been seen before, assign it a new integer
            if video_file not in label_to_int:
                label_to_int[video_file] = current_label
                current_label += 1

            labels.append(label_to_int[video_file])

    return np.array(training_data), np.array(labels)


def load_validation_data():
    validation_data = []
    labels = []
    label_to_int = {}
    current_label = 0

    for video_file in os.listdir(VIDEO_TEST_FOLDER):
        video_file_path = os.path.join(VIDEO_TEST_FOLDER, video_file)
        #print(f"Processing video file: {video_file_path}")

        with open(video_file_path, 'rb') as file:
            frames = preprocess_video(file, input_resize=(171, 128), H=112)
            validation_data.append(frames)

            if video_file not in label_to_int:
                label_to_int[video_file] = current_label
                current_label += 1

            labels.append(label_to_int[video_file])

    return np.array(validation_data), np.array(labels)

loss_function = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
optimizer = tf.keras.optimizers.Adam()
# maybe softmax layer in fc6, stable softmax
def train_model(model, train_data, val_data, epochs, model_name):
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        # Training loop - iterate over the batches
       
        for step, (x_batch, y_batch) in enumerate(train_data):
            with tf.GradientTape() as tape:
                if (model_name == 'fc6' or model_name == 'score_regressor'):
                    x_batch = tf.reduce_mean(x_batch, axis=[1, 2, 3])
                # x_batch = tf.reduce_mean(x_batch, axis=[1, 2, 3])
                predictions = model(x_batch, training=True)
                loss = loss_function(y_batch, predictions)
            grads = tape.gradient(loss, model.trainable_weights)
            optimizer.apply_gradients(zip(grads, model.trainable_weights))
            print(f"Step {step}, Loss: {loss.numpy()}")

        # Validation loop
        for x_batch, y_batch in val_data:
            if (model_name == 'fc6' or model_name == 'score_regressor'):
                x_batch = tf.reduce_mean(x_batch, axis=[1, 2, 3])
            # x_batch = tf.reduce_mean(x_batch, axis=[1, 2, 3])
            val_predictions = model(x_batch, training=False)
            val_loss = loss_function(y_batch, val_predictions)
        print(f"Validation Loss: {val_loss.numpy()}")


def main():
    # #this needs to be changed to be all video files 
    # frames = preprocess_video(VIDEO_FILE, input_resize=(171, 128), H=112)
    # video = preprocess_video(VIDEO_FILE, input_resize=(171, 128), H=112)

    # SAVE THESE TO A CSV FILE TO TAKE LESS TIME TO READ LOL
    training_data, training_labels = load_training_data()
    validation_data, validation_labels = load_validation_data()
    # print('training_data: ', training_data.shape)
    # training_data:  (300, 160, 3, 112, 112)
    # print('training_labels: ', training_labels) 0-299 distinct
    #print('validation_data: ', validation_data.shape)
    # validation_data:  (70, 160, 3, 112, 112)
    # print('validation_labels: ', validation_labels) 0-99 distinct?

    # Create TensorFlow datasets
    train_dataset = tf.data.Dataset.from_tensor_slices((training_data, training_labels)).batch(10)
    val_dataset = tf.data.Dataset.from_tensor_slices((validation_data, validation_labels)).batch(10)


    c3d_model = C3D()    
    # Train the model
    train_model(c3d_model, train_dataset, val_dataset, epochs=10, model_name='c3d')
    save_c3d_model_weights(c3d_model, C3D_SAVE_PATH)
    
    c3d_alt_model = C3D_altered()
    #TRAIN HERE
    print('TRAINING C3D ALT MODEL')
    train_model(c3d_alt_model, train_dataset, val_dataset, epochs=10, model_name='c3d_alt')
    save_c3d_alt_model_weights(c3d_alt_model, C3D_ALT_PATH)
    
   
    fc6_model = my_fc6()
    #TRAIN HERE
    print('TRAINING FC6')
    loss_function_fc6 = tf.keras.losses.SparseCategoricalCrossentropy() # from_logits=True
    train_model(fc6_model, train_dataset, val_dataset, epochs=10, model_name='fc6')
    save_fc6_model_weights(fc6_model, FC6_PATH)
    

    # This does work! Loss is horrifically high but it works
    score_re_model = ScoreRegressor()
    #TRAIN HERE 
    print('TRAINING SCORE REGRESSOR')
    train_model(score_re_model, train_dataset, val_dataset, epochs=10, model_name='score_regressor')
    save_score_reg_model_weights(score_re_model, SCORE_REG_PATH)


    # Now use the saved model path for classification
    #action_classifier(frames, C3D_SAVE_PATH)
    # run action scorring 
    #action_scoring(video, C3D_ALT_PATH, FC6_PATH, SCORE_REG_PATH)

    #SUM AND PRINT FINAL ACTION SCORE

if __name__ == '__main__':
    main()

# TODO: want to be able to save data to a file, one of frames, one of video
# folder of full length videos
# folder of frames that correspond to each full length video
