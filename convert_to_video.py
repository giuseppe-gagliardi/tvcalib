import cv2
import os

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
dir_path = 'tmp/demo/bar_ben_004323_004353'
out = cv2.VideoWriter('tmp/demo/bar_ben_004323_004353/output.mp4', fourcc, 25.0, (1120, 540))

nr_frames = len([f for f in os.listdir(dir_path) if f.endswith('.png')])
# Iterate over frames and write them to video
frame_dir = "tmp/demo/bar_ben_004323_004353"
for i in range(1,nr_frames+1):
    filename = "{:04d}".format(i)+'.png'
    frame_path = os.path.join(frame_dir, filename)
    frame = cv2.imread(frame_path)
    print(filename, end='\r')
    out.write(frame)
out.release()