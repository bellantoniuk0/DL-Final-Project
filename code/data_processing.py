import cv2 as cv
import numpy as np
import tempfile
import tensorflow as tf
import pandas as pd
import os


def read_video(video_path):
    """Reads video from a file and returns a VideoCapture object."""
    return cv.VideoCapture(video_path)

def process_frames(video_capture, input_resize, H):
    """Process all frames of the video and return a tensor of processed frames."""
    frames = []
    while True:
        ret, frame = video_capture.read()
        if not ret:
            break
        frame = preprocess_frame(frame, input_resize, H)
        frames.append(frame)
    
    frames = tf.stack(frames)
    return frames

def center_crop(img, dim):
    """Returns center cropped image."""
    width, height = img.shape[1], img.shape[0]
    crop_width = dim[0] if dim[0] < img.shape[1] else img.shape[1]
    crop_height = dim[1] if dim[1] < img.shape[0] else img.shape[0]
    mid_x, mid_y = width // 2, height // 2
    cw2, ch2 = crop_width // 2, crop_height // 2
    crop_img = img[mid_y - ch2:mid_y + ch2, mid_x - cw2:mid_x + cw2]
    return crop_img

def preprocess_frame(frame, input_resize, H):
    """Convert color, resize, crop, and convert to tensor."""
    frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    frame = cv.resize(frame, input_resize, interpolation=cv.INTER_LINEAR)
    frame = center_crop(frame, (H, H))
    frame = tf.convert_to_tensor(frame, dtype=tf.float32)
    frame = (frame / 255.0 - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]  # Normalize
    return frame

def pad_frames(frames, C, H):
    """Pad frames to ensure the count is divisible by 16."""
    padding_count = (16 - frames.shape[0] % 16) % 16
    if padding_count > 0:
        # Ensure the padding shape matches frames excluding the batch dimension
        padding = tf.zeros((padding_count, H, H, 3), dtype=frames.dtype)
        frames = tf.concat([frames, padding], axis=0)
    return frames

def preprocess_video(video_file, input_resize=(171, 128), H=112):
    """Process video file into a tensor suitable for model input."""
    if video_file != "sample":
        with tempfile.NamedTemporaryFile(delete=False) as tfile:
            tfile.write(video_file.read())
            video_path = tfile.name
    else:
        video_path = "054.avi"
    # print('VIDEO FILE AHHHH', video_path)
    vf = read_video(video_path)
    frames = process_frames(vf, input_resize, H)

    vf.release()
    cv.destroyAllWindows()

    frames = pad_frames(frames, 3, H)
    #print('Frames shape before transpose: ', frames.shape)
    frames = tf.transpose(frames, [0, 3, 1, 2])  # Change to (batch, channel, height, width)
    frames = tf.cast(frames, tf.float64)  # Convert to double
    # print(f"Processed video shape: {frames.shape}")
    #  frames.shape for all = (160, 3, 112, 112)
    return frames


# Saving data to dataframe/csv
# def tensor_to_dataframe(tensor):
#     """Convert a tensor into a pandas DataFrame."""
#     # Flatten the tensor and convert it to a numpy array
#     array = tensor.numpy().flatten() <-- want to reshape not flatten
#     # Create a DataFrame
#     df = pd.DataFrame(array, columns=['ProcessedData'])
#     return df

# def save_to_csv(df, filepath):
#     """Save the DataFrame to a CSV file."""
#     df.to_csv(filepath, index=False)
#     print(f"Data saved to {filepath}")


def main(directory_path, output_csv_path):
    # Initialize a DataFrame to hold all the processed data
    all_data = pd.DataFrame()

    # Loop through each video file in the directory
    for filename in os.listdir(directory_path):
        if filename.endswith(".avi"):  # Check if the file is an AVI video
            video_file_path = os.path.join(directory_path, filename)
            print(f"Processing video: {video_file_path}")

            # Open the video file
            with open(video_file_path, 'rb') as video_file:
                processed_tensor = preprocess_video(video_file)
                # Convert the processed tensor to DataFrame
                # processed_data_df = tensor_to_dataframe(processed_tensor)
                # Append the processed data of the current video to the main DataFrame
                # all_data = pd.concat([all_data, processed_data_df], ignore_index=True)

    # After all videos are processed, save the DataFrame to CSV
    # save_to_csv(all_data, output_csv_path)
    print("All video processing complete and data saved.")


if __name__ == "__main__":
    directory_path = "../data/diving_samples_len_151_lstm"  # Path to the folder with videos
    output_csv_path = "../data/processed_video_data.csv"  # Path to save the combined CSV file
    main(directory_path, output_csv_path)
