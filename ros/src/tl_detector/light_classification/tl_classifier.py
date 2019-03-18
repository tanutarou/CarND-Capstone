from styx_msgs.msg import TrafficLight
import tensorflow as tf
import numpy as np
import time

class TLClassifier(object):
    def __init__(self):
        #TODO load classifier
        PATH_TO_FROZEN_GRAPH = './frozen_graph/frozen_inference_graph.pb'
        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
          od_graph_def = tf.GraphDef()
          with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

        self.sess = tf.Session(graph=self.detection_graph)
        
        self.colormap = {1: TrafficLight.GREEN, 2: TrafficLight.RED, 3: TrafficLight.YELLOW, 4: TrafficLight.UNKNOWN}

    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """

        image_expanded = np.expand_dims(image, axis=0)
        #start = time.time()
        output_dict = self.run_inference_for_single_image(image, self.detection_graph)
        #print('time', time.time() - start)


        #TODO implement light color prediction
        lights = output_dict['detection_classes'][output_dict['detection_scores'] > 0.9]
        if len(lights) > 0:
            print('detected', self.colormap[lights[0]]
)
            return self.colormap[lights[0]]
        else:
            return TrafficLight.UNKNOWN

    def run_inference_for_single_image(self, image, graph):
      with graph.as_default():
        ops = tf.get_default_graph().get_operations()
        all_tensor_names = {output.name for op in ops for output in op.outputs}
        tensor_dict = {}
        for key in [
            'num_detections', 'detection_boxes', 'detection_scores',
            'detection_classes'
        ]:
          tensor_name = key + ':0'
          if tensor_name in all_tensor_names:
            tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
                tensor_name)
        image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

        output_dict = self.sess.run(tensor_dict,
                               feed_dict={image_tensor: np.expand_dims(image, 0)})

        output_dict['num_detections'] = int(output_dict['num_detections'][0])
        output_dict['detection_classes'] = output_dict['detection_classes'][0].astype(np.uint8)
        output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
        output_dict['detection_scores'] = output_dict['detection_scores'][0]
      return output_dict
