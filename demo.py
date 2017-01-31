import sys

sys.path.append('./')

from yolo.net.yolo_tiny_net import YoloTinyNet 
import tensorflow as tf 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

classes_name = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train","tvmonitor"]


def process_predicts(predicts):
  # 2 Boxes per grid cell
  p_classes = predicts[0, :, :, 0:20]   # Probability of each class
  C = predicts[0, :, :, 20:22]          # Box confidences, 2 per grid cell
  coordinate = predicts[0, :, :, 22:]   # Coordinates and sizes
  coordinate = np.reshape(coordinate, (7, 7, 2, 4))
  p_classes = np.reshape(p_classes, (7, 7, 1, 20)) # Probability of each class in each grid cell given object
  C = np.reshape(C, (7, 7, 2, 1))       # Box confidences?

  P = C * p_classes     # (Shape: 7, 7, 2, 20)

  objects = []

  threshold = 0.05
  print('Max is: ', np.max(P))
  for grid_x in range(7):
      for grid_y in range(7):
          if np.max(P[grid_y, grid_x, :, :]) < threshold:
              continue

          print('Adding object:')
          print(grid_x, grid_y)
          print(np.max(P[grid_y, grid_x, :, :]))
          class_index = np.argmax(P[grid_y, grid_x, :, :])
          asd = P[grid_y, grid_x, :, :]
          class_index = np.unravel_index(class_index, asd.shape)
          class_num = class_index[1]
          box_num = class_index[0]
          print('Box: ', box_num)
          max_coordinate = coordinate[grid_y, grid_x, box_num, :]
          xcenter = max_coordinate[0]
          ycenter = max_coordinate[1]
          w = max_coordinate[2]
          h = max_coordinate[3]

          xcenter = (grid_x + xcenter) * (448.0/7.0)
          ycenter = (grid_y + ycenter) * (448.0/7.0)

          w = w * 448
          h = h * 448
          print('Size: ', w, h)
          print('Center: ', xcenter, ycenter)

          xmin = xcenter - w/2.0
          ymin = ycenter - h/2.0

          xmax = xmin + w
          ymax = ymin + h
          print('Class detected: ', classes_name[class_num])
          objects.append({
            'class_name': classes_name[class_num],
            'xmin': int(xmin),
            'ymin': int(ymin),
            'xmax': int(xmax),
            'ymax': int(ymax)
          })

  return objects

common_params = {'image_size': 448, 'num_classes': 20, 'batch_size': 1}
net_params = {'cell_size': 7, 'boxes_per_cell': 2, 'weight_decay': 0.0005}

net = YoloTinyNet(common_params, net_params, test=True)

image = tf.placeholder(tf.float32, (1, 448, 448, 3))
predicts = net.inference(image)

sess = tf.Session()

# Load and resize image
img = Image.open('helene.jpg')
resized_img = img.resize((448, 448), Image.ANTIALIAS)
np_img = np.array(resized_img)
np_img = np_img.astype(np.float32)
np_img = (np_img / 255.0) * 2 - 1
np_img = np.reshape(np_img, (1, 448, 448, 3))

saver = tf.train.Saver(net.trainable_collection)

saver.restore(sess,'models/pretrain/yolo_tiny.ckpt')
#saver.restore(sess,'models/train/model.ckpt-5000')

np_predict = sess.run(predicts, feed_dict={
    image: np_img,
    #net.dropout_prob: 1.0,
})

objects = process_predicts(np_predict)


def draw_object(ax, object):
    ax.add_patch(
        patches.Rectangle(
            (object['xmin'], object['ymin']),  # (x,y)
            object['xmax']-object['xmin'],  # width
            object['ymax']-object['ymin'],  # height
            fill=False,
            edgecolor='red',
            linewidth=2,
        )
    )
    center_x = object['xmin'] + (object['xmax']-object['xmin'])/2.0
    center_y = object['ymin'] + (object['ymax']-object['ymin'])/2.0
    print('Center: ', center_x, center_y)
    ax.add_patch(
        patches.Circle(
            (center_x, center_y),
            color='green'
        )
    )

# Draw image and box
f, ax = plt.subplots(1, 1, figsize=(20, 20))
ax.imshow(resized_img)
for object in objects:
    print(object)
    draw_object(ax, object)
plt.show()

sess.close()
