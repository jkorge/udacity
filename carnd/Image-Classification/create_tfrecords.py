'''
Use: python data_conversion --input_yaml input_file_name.yaml --output_path output_file_name.record
Description:
    Reads dictionary of examples ({string: feature} maps) from input_yaml
    Parses image filepath, object bounding box coords, and object label from dict
    Encodes image file data
    Stores encoded image and corresponding features in tf.train.Example object
    Serializes tfrecords to output_path
Source: https://github.com/oflucas/Traffic-Light-Detection
'''

import tensorflow as tf
import yaml
import os, sys
import io
from PIL import Image
from utils import dataset_util


flags = tf.app.flags
flags.DEFINE_string('output_path', '', 'Path to output TFRecord')
flags.DEFINE_string('input_yaml', '', 'Path to labeling YAML')
FLAGS = flags.FLAGS

LABEL_DICT =  {
    "GREEN" : 1,
    "RED" : 2,
    "YELLOW" : 3,
    "UNKNOWN" : 4,
    }

def create_tf_example(example):
    filename = example['filename'] # Filename of the image. Empty if image is not from file
    filename = filename.encode()

    with tf.gfile.GFile(example['filename'], 'rb') as fid:
        encoded_image = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_image)
    image = Image.open(encoded_jpg_io)
    width, height = image.size

    image_format = 'jpg'.encode() 

    xmins = [] # List of normalized left x coordinates in bounding box (1 per box)
    xmaxs = [] # List of normalized right x coordinates in bounding box
                # (1 per box)
    ymins = [] # List of normalized top y coordinates in bounding box (1 per box)
    ymaxs = [] # List of normalized bottom y coordinates in bounding box
                # (1 per box)
    classes_text = [] # List of string class name of bounding box (1 per box)
    classes = [] # List of integer class id of bounding box (1 per box)

    for box in example['annotations']:
        # adding box, one image may have multiple detected boxes
        if box['xmin'] + box['x_width'] > width or box['ymin']+ box['y_height'] > height:
            continue

        xmins.append(float(box['xmin']) / width)
        xmaxs.append(float(box['xmin'] + box['x_width']) / width)
        ymins.append(float(box['ymin']) / height)
        ymaxs.append(float(box['ymin']+ box['y_height']) / height)
        classes_text.append(box['class'].encode())
        classes.append(int(LABEL_DICT[box['class']]))

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_image),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))

    return tf_example


def main(_):
    
    writer = tf.python_io.TFRecordWriter(FLAGS.output_path)
    
    INPUT_YAML = FLAGS.input_yaml
    examples = yaml.load(open(INPUT_YAML, 'rb').read())

    len_examples = len(examples)
    print("Loaded ", len(examples), "examples")

    for i in range(len(examples)):
        examples[i]['filename'] = os.path.abspath(os.path.join(os.path.dirname(INPUT_YAML), examples[i]['filename']))
    
    counter = 0.
    for example in examples:
        tf_example = create_tf_example(example)
        writer.write(tf_example.SerializeToString())

        if counter % 10 == 0:
            print("Percent done", (counter/len_examples)*100)
        counter += 1.

    writer.close()



if __name__ == '__main__':
    tf.app.run()