
from tvcalib.utils.objects_3d import SoccerPitchSN as SoccerPitch
import numpy as np
import cv2 as cv
import os
import json
from sn_segmentation.src.evaluate_extremities import scale_points

def overlap_inout(img1, img2):

    # Crea una maschera dei pixel non neri in img2
    gray = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)
    mask = cv.threshold(gray, 10, 255, cv.THRESH_BINARY)[1]
    mask_inv = cv.bitwise_not(mask)

    # Applica la maschera per sovrapporre solo i pixel colorati di img2 su img1
    img2_masked = cv.bitwise_and(img2, img2, mask=mask)
    img1_masked = cv.bitwise_and(img1, img1, mask=mask_inv)
    result = cv.add(img1_masked, img2_masked)

    return result

#params
threshold = 10
resolution_width = 960
resolution_height = 540

lines_palette = [0, 0, 0]
for line_class in SoccerPitch.lines_classes:
    lines_palette.extend(SoccerPitch.palette[line_class])

subset = 'test'
dataset_dir = os.path.join('data', 'datasets')
dataset_dir = os.path.join(dataset_dir, f"sncalib-{subset}")

num_points_lines = 4
num_points_circles = 8
radius = 4
maxdists = 30
output_dir = os.path.join('data', 'results')
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
output_dir = os.path.join(output_dir, f"np{num_points_lines}_nc{num_points_circles}_r{radius}_md{maxdists}")
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
prediction_dir = os.path.join('data', 'segment_localization')
prediction_dir = os.path.join(prediction_dir, f"np{num_points_lines}_nc{num_points_circles}_r{radius}_md{maxdists}")
#prediction_dir = os.path.join(prediction_dir, f"sn-baseline")
prediction_dir = os.path.join(prediction_dir, f"sncalib-{subset}")
if not os.path.exists(prediction_dir):
    print(prediction_dir)
    raise IsADirectoryError
frames = [f for f in os.listdir(dataset_dir) if ".jpg" in f]
annotation_files = [f for f in os.listdir(dataset_dir) if ".json" in f]
annotation_files.sort()
prediction_files = [f for f in os.listdir(prediction_dir) if ".json" in f]
prediction_files.sort()
""" output_prediction_folder = os.path.join('results', '${subset}')
if not os.path.exists(output_prediction_folder):
    os.makedirs(output_prediction_folder) """
frame_paths = dict()
for f in frames:
    frame_paths.update({f.split(".")[0]: os.path.join(dataset_dir, f)})

annotation_paths = dict()
for f in annotation_files:
    annotation_paths.update({f.split(".")[0]: os.path.join(dataset_dir, f)})

prediction_paths = dict()
for f in prediction_files:
    prediction_paths.update({f.split(".")[0]: os.path.join(prediction_dir, f)})
for frame in frames:
    #frame = frames[frame_index_int]
    prediction = dict()
    count = 0
    frame_path = os.path.join(dataset_dir, frame)
    frame_index = frame.split(".")[0]
    frame_index_int = int(frame_index)
    image = cv.imread(frame_path)
    #print(image.shape)
    if (not (int(prediction_files[frame_index_int].split('_')[1].split('.')[0]) == frame_index_int)):
        print (prediction_files[frame_index_int], frame_index_int)
    annotation_file = os.path.join(dataset_dir, annotation_files[frame_index_int])
    with open(annotation_file, 'r') as f:
        line_annotations = json.load(f)

    prediction_file = os.path.join(prediction_dir, prediction_files[frame_index_int])
    with open(prediction_file, 'r') as f:
        predictions = json.load(f)


    
    predictions = scale_points(predictions, resolution_width, resolution_height, frame_index_int)
    line_annotations = scale_points(line_annotations, resolution_width, resolution_height, frame_index_int)
    thickness = 4

    prediction = np.zeros((resolution_height, resolution_width,3)).astype('uint8')

    for k, class_name in enumerate(SoccerPitch.lines_classes):
        if class_name in predictions.keys(): 
            #print(k)       
            color = SoccerPitch.palette[class_name]
            #print("color",color)
            ptsArray = []
            for xy in predictions[class_name]:   
                y = int(xy['y'])
                x = int(xy['x'])
                ptsArray.append([x,y])
            #print("ptsArray",ptsArray)       
            if class_name == 'Circle central':
                pts = np.array(ptsArray,np.int32)
                pts = pts.reshape((-1, 1, 2))
                #print("pts",pts)
                prediction = cv.polylines(img=prediction, pts=[pts], color=color, isClosed=True, thickness=thickness)
                #print("prediction",prediction)       
            else :
                pts = np.array(ptsArray,np.int32)
                pts = pts.reshape((-1, 1, 2))
                #print("pts",pts)
                prediction = cv.polylines(img=prediction, pts=[pts], color=color, isClosed=False, thickness=thickness)
                #print("prediction",prediction)       

    g = prediction_file.split('/').pop().split('.')[0]
    
    output_file = os.path.join(output_dir, f'{g}.png')
    #print(output_file)
    #print(image.shape,prediction.shape)
    overlapped = overlap_inout(image, prediction)
    cv.imwrite(output_file, overlapped) 
    