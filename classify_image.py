''''
    Author:Shuntos 
    2020 Apr 14
'''



import tensorflow as tf
from tensorflow.python.platform import gfile
import numpy as np
import os
import cv2
import time
path_to_pb = "./weights/froze_xxx.pb"
classes = ["plant","negative"]
dir_path = 'imgs'

def inference(image):
    text= None
    image_size=128
    num_channels=3
    images = []
    try:
        # Resizing the image to our desired size and preprocessing will be done exactly as done during training
        image = cv2.resize(image, (image_size, image_size),0,0, cv2.INTER_LINEAR)
        images.append(image)
        images = np.array(images, dtype=np.uint8)
        images = images.astype('float32')
        images = np.multiply(images, 1.0/255.0)         
        # The input to the network is of shape [None image_size image_size num_channels]. Hence we reshape.
        x_batch = images.reshape(1, image_size,image_size,num_channels)

        # Let us restore the saved model 
        with tf.Session() as sess:
            print("load graph")
            with gfile.FastGFile(path_to_pb,'rb') as f:
               graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            sess.graph.as_default()
            tf.import_graph_def(graph_def, name='')
            graph_nodes=[n for n in graph_def.node]
            names = []
            for t in graph_nodes:
              names.append(t.name)
            print(names)
                
            graph = tf.get_default_graph()
            # Now, let's get hold of the op that we can be processed to get the output.
            # In the original network y_pred is the tensor that is the prediction of the network
            y_pred = graph.get_tensor_by_name("y_pred:0")

            ## Let's feed the images to the input placeholders
            x= graph.get_tensor_by_name("x:0") 

            # Creating the feed_dict that is required to be fed to calculate y_pred 
            feed_dict_testing = {x: x_batch}
            result=sess.run(y_pred, feed_dict=feed_dict_testing)
            # Result is of this format [[probabiliy_of_classA probability_of_classB ....]]
            # Convert np.array to list
            a = result[0].tolist()
            text = classes[a.index(max(a))]
            print("Result",text)
              
    except Exception as e:
        print("Exception:",e)

    return text



if __name__ == "__main__":
    img = cv2.imread("imgs/d.jpg")
    inference(img)
