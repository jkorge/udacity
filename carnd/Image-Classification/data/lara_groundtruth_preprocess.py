'''
Use: python lara_groundtruth_preprocess.py
Description:
    Removes fields from the ground truth file included in the laRA dataset download
    Resulting file (grouth_truth.txt) contains only the frame index, bounding box coords, and object class for each image
    Run the file in the same directory as the downloaded Lara_UrbanSeq1_GroundTruth_GT.txt file
'''
with open('Lara_UrbanSeq1_GroundTruth_GT.txt', 'r') as f:
    gt = f.readlines()

class_map = {'go': 'GREEN', 'warning': 'YELLOW', 'stop': 'RED', 'ambiguous': 'UNKNOWN'}
with open('ground_truth.txt', 'w') as f:
    for line in gt:
        if line[0] == "#":
            continue
        line = line.split()
        idx = line[2]
        x1 = line[3]
        x2 = line[4]
        y1 = line[5]
        y2 = line[6]
        color = class_map[line[10].split("'")[1]]
        f.write(' '.join([idx, x1, x2, y1, y2, color, '\n']))