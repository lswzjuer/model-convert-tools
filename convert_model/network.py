"""
Reference Model
Copyright (c) 2018 MobaiTech Inc 
Author: Abinash Mohanty
Revisions: 
"""

import numpy as np
import tensorflow as tf
from config import cfg
from roi_pooling_layer import roi_pooling_op as roi_pool_op
from lp import lp_op as lp_op
from saturate import saturate_op as saturate_op
from fp import fp_op as fp_op
from rpn_msr.proposal_layer_tf import proposal_layer as proposal_layer_py
from nearest_neighbor.nearest_neighbor import nearest_neighbor_layer as nearest_neighbor_layer_py
from utils.refModel_log import print_msg
import cPickle


DEFAULT_PADDING = 'SAME'

def include_original(dec):
        """ Meta decorator, which make the original function callable (via f._original() )"""
        def meta_decorator(f):
                decorated = dec(f)
                decorated._original = f
                return decorated
        return meta_decorator

@include_original
def layer(op):
        def layer_decorated(self, *args, **kwargs):
                name = kwargs.setdefault('name', self.get_unique_name(op.__name__))             # Automatically set a name if not provided.
                if len(self.inputs)==0:                                                                                                 # Figure out the layer inputs.
                        raise RuntimeError('No input variables found for layer %s.'%name)
                elif len(self.inputs)==1:
                        layer_input = self.inputs[0]
                else:
                        layer_input = list(self.inputs)
                layer_output = op(self, layer_input, *args, **kwargs)           # Perform the operation and get the output.
                self.layers[name] = layer_output                                                        # Add to layer LUT.
                self.feed(layer_output)                                                                         # This output is now the input for the next layer.
                return self                                                                                                     # Return self for chained calls.
        return layer_decorated


class Network(object):
        def __init__(self, inputs, isHardware=False, trainable=False):
                
                self.inputs = []
                self.layers = dict(inputs)
                self.trainable = trainable
                self._layer_map = {}
                self.isHardware = isHardware
                self.setup()

        def get_layer_map(self):
            return self._layer_map

        def setup(self):
                raise NotImplementedError('Must be subclassed.')

        def load(self, data_path, session, ignore_missing=False):
                """
                Initialize graph with pre-trained parameters. 
                """
                data_dict = np.load(data_path).item()
                for op_name in data_dict:
                        with tf.variable_scope(op_name, reuse=True):
                                for param_name, data in data_dict[op_name].iteritems():
                                        try:
                                                #print("op_name: %s param_name: %s\n" %(op_name, param_name))
                                                #print(data)
                                                var = tf.get_variable(param_name)
                                                session.run(var.assign(data))
                                                #print_msg("assign pretrain model "+param_name+ " to "+op_name,0)
                                        except ValueError:
                                                print_msg("ignore "+"Param: "+str(param_name)+" - OpName: "+str(op_name),3)
                                                if not ignore_missing:
                                                        raise
                print_msg("Model was successfully loaded from "+data_path ,3)


        def variable_summaries(self, var):
                """
                Attach a lot of summaries to a Tensor (for TensorBoard visualization).
                """
                with tf.name_scope('summaries'):
                        mean = tf.reduce_mean(var)
                        tf.summary.scalar('mean', mean)
                        with tf.name_scope('stddev'):
                                stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
                        tf.summary.scalar('stddev', stddev)
                        tf.summary.scalar('max', tf.reduce_max(var))
                        tf.summary.scalar('min', tf.reduce_min(var))
                        tf.summary.histogram('histogram', var)


        def feed(self, *args):
                assert len(args)!=0
                self.inputs = []
                for layer in args:
                        if isinstance(layer, basestring):
                                try:
                                        layer = self.layers[layer]
                                        print_msg(str(layer),0)
                                except KeyError:
                                        print_msg(str(self.layers.keys()),3)
                                        raise KeyError('Unknown layer name fed: %s'%layer)
                        self.inputs.append(layer)
                return self

        def get_output(self, layer):
                try:
                        layer = self.layers[layer]
                except KeyError:
                        print_msg(str(self.layers.keys()),3)
                        raise KeyError('Unknown layer name fed: %s'%layer)
                return layer

        def get_unique_name(self, prefix):
                id = sum(t.startswith(prefix) for t,_ in self.layers.items())+1
                return '%s_%d'%(prefix, id)

        def make_var(self, name, shape, initializer=None, trainable=False, regularizer=None):
                #print("    tf.get_variable(%s, %s, ..)\n" %(name, shape))
                return tf.get_variable(name, shape, initializer=initializer, trainable=self.trainable, regularizer=regularizer)

        def validate_padding(self, padding):
                assert padding in ('SAME', 'VALID')

        @layer 
        def pad_v1(self, input, pad_b, pad_r, name):
                """ Special Padding to make caffe and tensorflow similar"""
                paddings = [[0,0],[0, pad_b],[0, pad_r],[0,0]]
                return tf.pad(input, paddings, constant_values=-999999.0)       # Random Big Negative value

 
        @layer
        def pad(self, input, pad, name):
                """ Padding using tf.pad """
                paddings = [[0,0],[pad, pad],[pad, pad],[0,0]]
                return tf.pad(input, paddings)  
        
        @layer
        def convert_to_float(self, input, fl, name):
                # Convert int from fixed point to float representations. Needed sometimes. 
                input_fp =  fp_op.fp(input, fl)  
                return input_fp

        def lp_deconv(self, input, k, output_shape, strides, bw, fl, rs, padding):
                """
                Low precision convolution.      
                """

                c = tf.nn.conv2d_transpose(input, k, output_shape, strides=strides, padding=padding)   # Tensorflow
                return lp_op.lp(c, bw, rs)      # Do the right shift here and bit truncation and satturation here !

        def lp_conv(self, input, k, s_h, s_w, bw, fl, rs, padding):
                """
                Low precision convolution.      
                """
                c = tf.nn.conv2d(input, k, [1, s_h, s_w, 1], padding=padding)
                return lp_op.lp(c, bw, rs)      # Do the right shift here and bit truncation and satturation here !

        def saturate(self, input, bw):
                """
                Saturates tensors based on the bw value. For bw == 8, values are saturated in the range (-128,127). 
                Implementation also works for bw == 16. 
                """
                return saturate_op.sp(input, bw)

        @layer
        def conv(self, input, k_h, k_w, c_o, s_h, s_w, name, bw=cfg.WORD_WIDTH, fl=10, rs=0, biased=True,relu=True, padding=DEFAULT_PADDING):
                """ contribution by miraclebiu, and biased option"""
                self.validate_padding(padding)
                c_i = input.get_shape()[-1]
                if self.isHardware:     
                        convolve = lambda i, k: self.lp_conv(i, k, s_h, s_w, bw, fl, rs, padding)               # DLA
                else:
                        convolve = lambda i, k: tf.nn.conv2d(i, k, [1, s_h, s_w, 1], padding=padding)   # Tensorflow

                with tf.variable_scope(name) as scope:
                        init_weights = tf.contrib.layers.variance_scaling_initializer(factor=0.01, mode='FAN_AVG', uniform=False)
                        init_biases = tf.constant_initializer(0.0)
                        #print("TF DEBUG, scope: %s\n" %(name))
                        kernel = self.make_var('weights', [k_h, k_w, c_i, c_o], init_weights, self.trainable, \
                                                                   regularizer=self.l2_regularizer(0.0005))
                        if cfg.ENABLE_TENSORBOARD:
                                self.variable_summaries(kernel)
                        if biased:
                                biases = self.make_var('biases', [c_o], init_biases, self.trainable)
                                if cfg.ENABLE_TENSORBOARD:
                                        self.variable_summaries(biases)
                                conv = convolve(input, kernel)
                                if relu:
                                        bias = tf.nn.bias_add(conv, biases)
                                        if self.isHardware:
                                                bias_s = self.saturate(bias, cfg.WORD_WIDTH)    # New addition for saturation
                                                return tf.nn.relu(bias_s)
                                        return tf.nn.relu(bias)
                                bias_add = tf.nn.bias_add(conv, biases)
                                if self.isHardware:
                                        return self.saturate(bias_add, cfg.WORD_WIDTH)  # New addition for saturation
                                return bias_add
                        else:
                                conv = convolve(input, kernel)
                                if relu:
                                        return tf.nn.relu(conv)
                                return conv

        @layer
        def upconv(self, input, shape, c_o, name, bw=cfg.WORD_WIDTH, fl=10, rs=0, ksize=4, stride = 2, biased=False, relu=True, padding=DEFAULT_PADDING):
                
                """ up-conv"""
                self.validate_padding(padding)

                c_in = input.get_shape()[3].value
                in_shape = tf.shape(input)
                if shape is None:
                        # h = ((in_shape[1] - 1) * stride) + ksize - 2xpad
                        # w = ((in_shape[2] - 1) * stride) + ksize - 2xpad 
                        h = ((in_shape[1] ) * stride)
                        w = ((in_shape[2] ) * stride)
                        new_shape = [in_shape[0], h, w, c_o]
                else:
                        new_shape = [in_shape[0], shape[1], shape[2], c_o]
                output_shape = tf.stack(new_shape)

                filter_shape = [ksize, ksize, c_o, c_in]
                if self.isHardware:     
                        deconvolve = lambda i, k: self.lp_deconv(i, k, output_shape, [1, stride, stride, 1], bw, fl, rs, padding=DEFAULT_PADDING)               # DLA
                else:
                        deconvolve = lambda i, k: tf.nn.conv2d_transpose(i, k, output_shape, strides=[1, stride, stride, 1], padding=DEFAULT_PADDING)   # Tensorflow

                with tf.variable_scope(name) as scope:
                        # init_weights = tf.truncated_normal_initializer(0.0, stddev=0.01)
                        init_weights = tf.contrib.layers.variance_scaling_initializer(factor=0.01, mode='FAN_AVG', uniform=False)
                        filters = self.make_var('weights', filter_shape, init_weights, self.trainable, \
                                                                   regularizer=self.l2_regularizer(0.0005))
                        if cfg.ENABLE_TENSORBOARD:
                                self.variable_summaries(filters)

                        deconv = deconvolve(input, filters)
                        #deconv = tf.nn.conv2d_transpose(input, filters, output_shape, strides=[1, stride, stride, 1], padding=DEFAULT_PADDING, name=scope.name)
                        # coz de-conv losses shape info, use reshape to re-gain shape
                        deconv = tf.reshape(deconv, new_shape)

                        if biased:
                                init_biases = tf.constant_initializer(0.0)
                                biases = self.make_var('biases', [c_o], init_biases, self.trainable)
                                if cfg.ENABLE_TENSORBOARD:
                                        self.variable_summaries(biases)
                                if relu:
                                        bias = tf.nn.bias_add(deconv, biases)
                                        return tf.nn.relu(bias)
                                return tf.nn.bias_add(deconv, biases)
                        else:
                                if relu:
                                        return tf.nn.relu(deconv)
                                return deconv

        @layer
        def deconv(self, input, k_h,k_w, c_o,s_h,s_w,name, bw=cfg.WORD_WIDTH, fl=10, rs=0,
                   biased=True,relu=True, padding=DEFAULT_PADDING):

                """ deconv"""
                self.validate_padding(padding)

                c_in = input.get_shape()[3].value
                in_shape = tf.shape(input)
                # h = ((in_shape[1] - 1) * stride) + ksize - 2xpad
                # w = ((in_shape[2] - 1) * stride) + ksize - 2xpad
                h = ((in_shape[1]) * k_h)
                w = ((in_shape[2]) * k_w)
                new_shape = [in_shape[0], h, w, c_o]
                output_shape = tf.stack(new_shape)

                filter_shape = [k_h, k_w, c_o, c_in]
                if self.isHardware:
                        deconvolve = lambda i, k: self.lp_deconv(i, k, output_shape, [1, s_h, s_w, 1], bw, fl, rs,
                                                                 padding=DEFAULT_PADDING)  # DLA
                else:
                        deconvolve = lambda i, k: tf.nn.conv2d_transpose(i, k, output_shape,
                                                                         strides=[1, s_h, s_w, 1],
                                                                         padding=DEFAULT_PADDING)  # Tensorflow

                with tf.variable_scope(name) as scope:
                        # init_weights = tf.truncated_normal_initializer(0.0, stddev=0.01)
                        init_weights = tf.contrib.layers.variance_scaling_initializer(factor=0.01, mode='FAN_AVG',
                                                                                      uniform=False)
                        filters = self.make_var('weights', filter_shape, init_weights, self.trainable, \
                                                regularizer=self.l2_regularizer(0.0005))
                        if cfg.ENABLE_TENSORBOARD:
                                self.variable_summaries(filters)

                        deconv = deconvolve(input, filters)
                        # deconv = tf.nn.conv2d_transpose(input, filters, output_shape, strides=[1, stride, stride, 1], padding=DEFAULT_PADDING, name=scope.name)
                        # coz de-conv losses shape info, use reshape to re-gain shape
                        deconv = tf.reshape(deconv, new_shape)

                        if biased:
                                init_biases = tf.constant_initializer(0.0)
                                biases = self.make_var('biases', [c_o], init_biases, self.trainable)
                                if cfg.ENABLE_TENSORBOARD:
                                        self.variable_summaries(biases)
                                if relu:
                                        bias = tf.nn.bias_add(deconv, biases)
                                        return tf.nn.relu(bias)
                                return tf.nn.bias_add(deconv, biases)
                        else:
                                if relu:
                                        return tf.nn.relu(deconv)
                                return deconv

        @layer
        def relu(self, input, name):
                return tf.nn.relu(input, name=name)

        @layer
        def max_pool(self, input, k_h, k_w, s_h, s_w, name, padding=DEFAULT_PADDING):
                self.validate_padding(padding)
                return tf.nn.max_pool(input,
                                                          ksize=[1, k_h, k_w, 1],
                                                          strides=[1, s_h, s_w, 1],
                                                          padding=padding,
                                                          name=name)

        @layer
        def avg_pool(self, input, k_h, k_w, s_h, s_w, name, padding=DEFAULT_PADDING):
                self.validate_padding(padding)
                return tf.nn.avg_pool(input,
                                                          ksize=[1, k_h, k_w, 1],
                                                          strides=[1, s_h, s_w, 1],
                                                          padding=padding,
                                                          name=name)

        @layer
        def roi_pool(self, input, pooled_height, pooled_width, spatial_scale, name):
                # only use the first input
                if isinstance(input[0], tuple):
                        input[0] = input[0][0]

                if isinstance(input[1], tuple):
                        input[1] = input[1][0]

                return roi_pool_op.roi_pool(input[0], input[1],
                                                                        pooled_height,
                                                                        pooled_width,
                                                                        spatial_scale,
                                                                        name=name)[0]

        @layer
        def nearest_neighbor_layer(self, input, c_o, name, stride):
                #input[0]=input[0][0]
                print("NN DEBUG: input type: %s, shape: %s\n" %(type(input), tf.shape(input)))
                print(input)
                in_shape = tf.shape(input)
                h = ((in_shape[1])*stride)
                w = ((in_shape[2])*stride)
                new_shape = [in_shape[0],h,w,c_o]
               
                return tf.reshape(tf.py_func(nearest_neighbor_layer_py, [input[0],stride],[tf.float32]),new_shape,name=name)

        @layer
        def proposal_layer(self, input, feat_stride, anchor_scales, cfg_key, base_size, ratios, pre_nms_top_n, max_nms_topN, name, num_stddev, fl_cls_prob, fl_bbox_pred):
                if isinstance(input[0], tuple):
                        input[0] = input[0][0]
                        # input[0] shape is (1, H, W, Ax2)
                        # rpn_rois <- (1 x H x W x A, 5) [0, x1, y1, x2, y2]
                return tf.reshape(tf.py_func(proposal_layer_py,\
                                                                         [input[0],input[1],input[2], cfg_key, fl_cls_prob, fl_bbox_pred, feat_stride, anchor_scales, base_size, ratios, pre_nms_top_n, max_nms_topN, self.isHardware, num_stddev],\
                                                                         [tf.float32]),
                                                  [-1,5],name =name)

        @layer
        def reshape_layer(self, input, d, name):
                input_shape = tf.shape(input)
                if name == 'rpn_cls_prob_reshape':
                        #
                        # transpose: (1, AxH, W, 2) -> (1, 2, AxH, W)
                        # reshape: (1, 2xA, H, W)
                        # transpose: -> (1, H, W, 2xA)
                         return tf.transpose(tf.reshape(tf.transpose(input,[0,3,1,2]),
                                                                                        [       input_shape[0],
                                                                                                int(d),
                                                                                                tf.cast(tf.cast(input_shape[1],tf.float32)/tf.cast(d,tf.float32)*tf.cast(input_shape[3],tf.float32),tf.int32),
                                                                                                input_shape[2]
                                                                                        ]),
                                                                 [0,2,3,1],name=name)
                else:
                         return tf.transpose(tf.reshape(tf.transpose(input,[0,3,1,2]),
                                                                                [       input_shape[0],
                                                                                        int(d),
                                                                                        tf.cast(tf.cast(input_shape[1],tf.float32)*(tf.cast(input_shape[3],tf.float32)/tf.cast(d,tf.float32)),tf.int32),
                                                                                        input_shape[2]
                                                                                ]),
                                                                 [0,2,3,1],name=name)

        @layer
        def spatial_reshape_layer(self, input, d, name):
                input_shape = tf.shape(input)
                # transpose: (1, H, W, A x d) -> (1, H, WxA, d)
                return tf.reshape(input,\
                                                           [input_shape[0],\
                                                                input_shape[1], \
                                                                -1,\
                                                                int(d)])

        @layer
        def lrn(self, input, radius, alpha, beta, name, bias=1.0):
                return tf.nn.local_response_normalization(input,
                                                                depth_radius=radius,
                                                                alpha=alpha,
                                                                beta=beta,
                                                                bias=bias,
                                                                name=name)

        @layer
        def reshape_l(self, input, shape, name):
                return tf.reshape(input, shape)

        @layer
        def concat(self, inputs, axis, name):
                return tf.concat(axis=axis, values=inputs, name=name)

        @layer
        def dummy(self, input, name):
                """ Dummy Layer. Doesnt do anything"""
                return input 
        
        # Deprecated Implementation. Need 2 stage saturation. 
        #@layer
        #def fc(self, input, num_out, name, bw=cfg.WORD_WIDTH, fl=10, rs=0, relu=True):
        #       with tf.variable_scope(name) as scope:
        #               if isinstance(input, tuple):
        #                       input = input[0]

        #               input_shape = input.get_shape()
        #               if input_shape.ndims == 4:
        #                       dim = 1
        #                       print input_shape
        #                       for d in input_shape[1:].as_list():
        #                               dim *= d
        #                       feed_in = tf.reshape(tf.transpose(input,[0,3,1,2]), [-1, dim])
        #               else:
        #                       feed_in, dim = (input, int(input_shape[-1]))

        #               if name == 'bbox_pred':
        #                       init_weights = tf.truncated_normal_initializer(0.0, stddev=0.001)
        #                       init_biases = tf.constant_initializer(0.0)
        #               else:
        #                       init_weights = tf.truncated_normal_initializer(0.0, stddev=0.01)
        #                       init_biases = tf.constant_initializer(0.0)

        #               weights = self.make_var('weights', [dim, num_out], init_weights, self.trainable, \
        #                                                               regularizer=self.l2_regularizer(0.0005))
        #               biases = self.make_var('biases', [num_out], init_biases, self.trainable)
        #               if cfg.ENABLE_TENSORBOARD:
        #                       self.variable_summaries(biases)
        #                       self.variable_summaries(weights)

        #               op = tf.nn.relu_layer if relu else tf.nn.xw_plus_b
        #               fc = op(feed_in, weights, biases, name=scope.name)
        #               if self.isHardware: 
        #                       return lp_op.lp(fc, bw, rs)
        #               return fc

        @layer
        def fc(self, input, num_out, name, bw=cfg.WORD_WIDTH, fl=10, rs=0, relu=True):
                print("FC DEBUG: input: \n")
                print(input)
                print("FC DEBUG: done input\n")
                with tf.variable_scope(name) as scope:
                        if isinstance(input, tuple):
                                input = input[0]

                        input_shape = input.get_shape()
                        if input_shape.ndims == 4:
                                dim = 1
                                print(" FC  input_shape: %s\n" %(str(input_shape)))
                                #print input
                                print input_shape[1:]
                                for d in input_shape[1:].as_list():
                                        dim = dim * d
                                feed_in = tf.reshape(tf.transpose(input,[0,3,1,2]), [-1, dim])
                        else:
                                feed_in, dim = (input, int(input_shape[-1]))

                        if name == 'bbox_pred':
                                init_weights = tf.truncated_normal_initializer(0.0, stddev=0.001)
                                init_biases = tf.constant_initializer(0.0)
                        else:
                                init_weights = tf.truncated_normal_initializer(0.0, stddev=0.01)
                                init_biases = tf.constant_initializer(0.0)

                        weights = self.make_var('weights', [dim, num_out], init_weights, self.trainable, \
                                                                        regularizer=self.l2_regularizer(0.0005))
                        biases = self.make_var('biases', [num_out], init_biases, self.trainable)
                        biases_dummy = self.make_var('biases_dummy', [num_out], init_biases, self.trainable)    # Dummy bias to take care of saturation
                        if cfg.ENABLE_TENSORBOARD:
                                self.variable_summaries(biases)
                                self.variable_summaries(weights)
                        
                        fc = tf.nn.xw_plus_b(feed_in, weights, biases_dummy, name=scope.name)
                        if self.isHardware: 
                                fc = lp_op.lp(fc, bw, rs)
                        fc_biased = tf.add(fc, biases)  
        
                        if self.isHardware:
                                fc_biased_saturate = self.saturate(fc_biased, cfg.WORD_WIDTH)
                                if relu: 
                                        return tf.nn.relu(fc_biased_saturate)
                                return fc_biased_saturate

                        if relu:
                                return tf.nn.relu(fc_biased)
                        return fc_biased        

        @layer
        def softmax(self, input, name):
                input_shape = tf.shape(input)
                if name == 'rpn_cls_prob':
                        return tf.reshape(tf.nn.softmax(tf.reshape(input,[-1,input_shape[3]])),[-1,input_shape[1],input_shape[2],input_shape[3]],name=name)
                else:
                        tfsoftmax = tf.nn.softmax(input,name=name)
                        return tfsoftmax

        @layer
        def spatial_softmax(self, input, name):
                input_shape = tf.shape(input)
                return tf.reshape(tf.nn.softmax(tf.reshape(input, [-1, input_shape[3]])),
                                                  [-1, input_shape[1], input_shape[2], input_shape[3]], name=name)

        @layer
        def add(self,input,name, relu=True):
                """
                This is elementwise addition. 
                """

                # add saturate
                if relu:
                    x=tf.add(input[0],input[1], name=name)
                    if self.isHardware:
                        sat_s = self.saturate(x, cfg.WORD_WIDTH)    # New addition for saturation
                        return tf.nn.relu(sat_s)
                    return tf.nn.relu(x)
                if self.isHardware:
                    return self.saturate(x, cfg.WORD_WIDTH)    # New addition for saturation
                return tf.add(input[0],input[1], name=name)

        @layer
        def batch_normalization(self,input,name,relu=True, is_training=False):
                """contribution by miraclebiu"""
                if relu:
                        temp_layer=tf.contrib.layers.batch_norm(input,scale=True,center=True,is_training=is_training,scope=name)
                        return tf.nn.relu(temp_layer)
                else:
                        return tf.contrib.layers.batch_norm(input,scale=True,center=True,is_training=is_training,scope=name)

        @layer
        def negation(self, input, name):
                """ simply multiplies -1 to the tensor"""
                return tf.multiply(input, -1.0, name=name)

        @layer
        def bn_scale_combo(self, input, c_in, name, relu=True):
                """ PVA net BN -> Scale -> Relu"""
                with tf.variable_scope(name) as scope:
                        bn = self.batch_normalization._original(self, input, name='bn', relu=False, is_training=False)
                        if relu:
                                bn = tf.nn.relu(bn, name='relu')
                        return bn

        @layer
        def scale(self, input, c_in, name):
                with tf.variable_scope(name) as scope:
                        alpha = tf.get_variable('alpha', shape=[c_in, ], dtype=tf.float32,
                                                                        initializer=tf.constant_initializer(1.0), trainable=self.trainable,
                                                                        regularizer=self.l2_regularizer(0.00001))
                        beta = tf.get_variable('beta', shape=[c_in, ], dtype=tf.float32,
                                                                   initializer=tf.constant_initializer(0.0), trainable=self.trainable,
                                                                   regularizer=self.l2_regularizer(0.00001))
                        return tf.add(tf.multiply(input, alpha), beta)

        @layer
        def dropout(self, input, keep_prob, name):
                return tf.nn.dropout(input, keep_prob, name=name)

        def l2_regularizer(self, weight_decay=0.0005, scope=None):
                def regularizer(tensor):
                        with tf.name_scope(scope, default_name='l2_regularizer', values=[tensor]):
                                l2_weight = tf.convert_to_tensor(weight_decay,
                                                                           dtype=tensor.dtype.base_dtype,
                                                                           name='weight_decay')
                                return tf.multiply(l2_weight, tf.nn.l2_loss(tensor), name='value')
                return regularizer

        def smooth_l1_dist(self, deltas, sigma2=9.0, name='smooth_l1_dist'):
                with tf.name_scope(name=name) as scope:
                        deltas_abs = tf.abs(deltas)
                        smoothL1_sign = tf.cast(tf.less(deltas_abs, 1.0/sigma2), tf.float32)
                        return tf.square(deltas) * 0.5 * sigma2 * smoothL1_sign + \
                                                (deltas_abs - 0.5 / sigma2) * tf.abs(smoothL1_sign - 1)

