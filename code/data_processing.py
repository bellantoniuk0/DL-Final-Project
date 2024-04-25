import cv2
import pandas as pd

def extract_frames(video_path, times, output_folder):
    """
    Extracts frames from a video at specific times.

    :param video_path: Path to the video file.
    :param times: List of times in seconds to extract frames.
    :param output_folder: Folder to save the extracted frames.
    """
    vidcap = cv2.VideoCapture(video_path)
    fps = vidcap.get(cv2.CAP_PROP_FPS)  # Get the frames per second of the video

    success, image = True, None
    for time in times:
        frameId = int(fps * time)  # Calculate the frame number (should be 151)
        vidcap.set(cv2.CAP_PROP_POS_FRAMES, frameId)
        success, image = vidcap.read()
        if success:
            cv2.imwrite(f"{output_folder}/frame_at_{time}_sec.jpg", image)  # Save frame as JPEG file

    vidcap.release()

def synchronize_frames_with_data(csv_file, video_folder, output_folder):
    """
    Synchronizes video frames with data from a CSV file.

    :param csv_file: Path to the CSV file containing times and other data.
    :param video_folder: Folder containing the videos.
    :param output_folder: Folder to save the extracted frames.
    """
    data = pd.read_csv(csv_file)

    for index, row in data.iterrows():
        video_path = f"{video_folder}/{row['video_name']}"
        times = [row['time_of_interest']]  # Time when the dive happens, you might adjust if you have multiple times
        extract_frames(video_path, times, output_folder)

# Example usage
csv_file = '.\data\data.csv'  # Your CSV file path
video_folder = '.\data\diving_samples_len_151_lstm'  # Folder where your videos are stored
output_folder = '.\data\diving_samples_output_frames'  # Folder where you want to save extracted frames

synchronize_frames_with_data(csv_file, video_folder, output_folder)
