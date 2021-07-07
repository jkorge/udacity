'''
Use: python to_annotations.py --output_path <path/to/annotations.yaml> --ground_truth_path <path/to/ground_truth.txt>
Description:
    Reads bounding box and classfication info for LaRA from ground truth txt file
    File format should be: frameindex x1 y1 x2 y2 class
        One line per frame containing an traffic light
    Outputs annotations into yaml file
Source: https://github.com/oflucas/Traffic-Light-Detection
'''
import tensorflow as tf

flags = tf.app.flags
flags.DEFINE_string('output_path', '', 'Path to output TYAML file')
flags.DEFINE_string('ground_truth_path', '', 'Path to file containing ground truth data')
FLAGS = flags.FLAGS

def write_image_yaml(o, idx, boxes):
    o.write('- annotations:\n')
    for box in boxes:
        o.write(box + '\n')

    tail = '''  class: image
  filename: Lara3D_UrbanSeq1_JPG/frame_{idx}.jpg'''.format(idx='%06d'%idx)
    o.write(tail + '\n')

def convert(fin, fout):
    images = {}
    with open(fin) as f:
        for line in f.readlines():
            idx_, x1, y1, x2, y2, name = line.strip().split()
            idx = int(idx_)
            dx = int(x2) - int(x1)
            dy = int(y2) - int(y1)
            if dx < 1 or dy < 1:
                continue
            
            box = '  - {class: %s, x_width: %d, xmin: %s, y_height: %d, ymin: %s}' % (name, dx, x1, dy, y1)
            images.setdefault(idx, []).append(box)

    
    with open(fout, 'w') as f:
        for idx, boxes in images.items():
            write_image_yaml(f, idx, boxes)

def main(_):
    convert(FLAGS.ground_truth_path, FLAGS.output_path)

if __name__ == '__main__':
    tf.app.run()