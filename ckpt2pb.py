import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"    
import tensorflow as tf
from tensorflow.python.framework import graph_util


def ckpt2pb(output_nodes, cpkt_meta, ckpt, pb_path):
    '''convert .ckpt to .pb file
    '''
    with tf.Graph().as_default():
        graph = tf.Graph()
        config = tf.ConfigProto(allow_soft_placement=True)

        with tf.Session(graph=graph, config=config) as sess:
            # restored model from ckpt
            saver = tf.train.import_meta_graph(cpkt_meta)
            saver.restore(sess, ckpt)

            # save freeze graph into .pb file
            graph_def = tf.get_default_graph().as_graph_def()
            constant_graph = graph_util.convert_variables_to_constants(sess, graph_def, [output_node for output_node in output_nodes])
            with tf.gfile.FastGFile(pb_path, mode='wb') as f:
                f.write(constant_graph.SerializeToString())
            print("convert done!")


if __name__=="__main__":
    output_nodes = ["softmax_linear/add"]
    cpkt_meta = '/home/cw/git/works_git/tf_mnist/logs/model.ckpt-1999.meta'
    ckpt = '/home/cw/git/works_git/tf_mnist/logs/model.ckpt-1999'
    pb_path = 'mnist.pb'

    ckpt2pb(output_nodes, cpkt_meta, ckpt, pb_path)


    
