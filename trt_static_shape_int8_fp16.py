import numpy as np
import tensorrt as trt
import pycuda.autoinit
import pycuda.driver as cuda
import struct
import os
import sys
import time
import mxnet as mx
import cv2
import calibrator

logger = trt.Logger(trt.Logger.INFO)

batch_size = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 1
print("TRT batch_size:", batch_size)

i_gpu = int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2].isdigit() else 0
print("GPU in use:", cuda.Device(i_gpu).name())
cuda.Device(i_gpu).make_context()

time_step = 20
print("Time step:", time_step)

_, params, extra_params = mx.model.load_checkpoint('ocr', 869)

class TrtPredictor:
    @classmethod
    def read_float(cls, wf, n):
        l = []
        for _ in range(n):
            l.append(*struct.unpack('f', wf.read(4)))
        return np.array(l, dtype=np.float32)
    
    def __init__(self, time_step):
        self.time_step = time_step
        
        trt_file_path = "ocr_{}.trt".format(time_step)
#         if os.path.isfile(trt_file_path):
        if False:
            with open(trt_file_path, 'rb') as f:
                engine_str = f.read()
        else:
            engine_str = self.build_engine_str()
            with open(trt_file_path, 'wb') as f:
                f.write(engine_str)
        
        with trt.Runtime(logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(engine_str)
        self.context = self.engine.create_execution_context()
        
    def __del__(self):
        self.context = None
        self.engine = None
            
    def build_engine_str(self):
        with trt.Builder(logger) as builder, self.create_engine(builder) as engine:
            return engine.serialize()
    
    def infer(self, batch_size, d_input, d_output):    
        bindings = [int(d_input), int(d_output)]    
        self.context.execute(batch_size, bindings)

class TrtPredictor_Conv(TrtPredictor):
    def create_engine(self, builder):
        network = builder.create_network()
        data = network.add_input("data", trt.DataType.FLOAT, (1, 80, 32))
        bag = []

        w = params['convolution0_weight'].asnumpy().reshape(-1)
        b = params['convolution0_bias'].asnumpy().reshape(-1)
        bag += [w, b]
        conv0 = network.add_convolution(data, 32, (3,3), w, b)
        conv0.stride = (1, 1)
        conv0.padding = (1, 1)
        print('conv0', conv0.get_output(0).shape)

        g = params['batchnorm0_gamma'].asnumpy().reshape(-1)
        m = extra_params['batchnorm0_moving_mean'].asnumpy().reshape(-1)
        v = extra_params['batchnorm0_moving_var'].asnumpy().reshape(-1)
        scale = g / np.sqrt(v + 2e-5)
        shift = -m / np.sqrt(v + 2e-5) * g + params['batchnorm0_beta'].asnumpy().reshape(-1)
        power = np.ones(len(g), dtype=np.float32)
        bag += [scale, shift, power]
        batn0 = network.add_scale(conv0.get_output(0), trt.ScaleMode.CHANNEL, shift, scale, power)
        
        actv0 = network.add_activation(batn0.get_output(0), trt.ActivationType.RELU)
        pool0 = network.add_pooling(actv0.get_output(0), trt.PoolingType.MAX, (2, 2))
        pool0.stride = (2, 2)
        
        w = params['convolution1_weight'].asnumpy().reshape(-1)
        b = params['convolution1_bias'].asnumpy().reshape(-1)
        bag += [w, b]
        conv1 = network.add_convolution(pool0.get_output(0), 32, (3,3), w, b)
        conv1.stride = (1, 1)
        conv1.padding = (1, 1)
        print('conv1', conv1.get_output(0).shape)
        
        actv1 = network.add_activation(conv1.get_output(0), trt.ActivationType.RELU)
        pool1 = network.add_pooling(actv1.get_output(0), trt.PoolingType.MAX, (2, 2))
        pool1.stride = (2, 2)
        
        w = params['convolution2_weight'].asnumpy().reshape(-1)
        b = params['convolution2_bias'].asnumpy().reshape(-1)
        bag += [w, b]
        conv2 = network.add_convolution(pool1.get_output(0), 16, (3,3), w, b)
        conv2.stride = (1, 1)
        conv2.padding = (1, 1)
        print('conv2', conv2.get_output(0).shape)
        
        g = params['batchnorm1_gamma'].asnumpy().reshape(-1)
        m = extra_params['batchnorm1_moving_mean'].asnumpy().reshape(-1)
        v = extra_params['batchnorm1_moving_var'].asnumpy().reshape(-1)
        scale = g / np.sqrt(v + 2e-5)
        shift = -m / np.sqrt(v + 2e-5) * g + params['batchnorm1_beta'].asnumpy().reshape(-1)
        power = np.ones(len(g), dtype=np.float32)
        bag += [scale, shift, power]
        batn1 = network.add_scale(conv2.get_output(0), trt.ScaleMode.CHANNEL, shift, scale, power)
        
        actv2 = network.add_activation(batn1.get_output(0), trt.ActivationType.RELU)
        pool2 = network.add_pooling(actv2.get_output(0), trt.PoolingType.MAX, (1, 2))
        pool2.stride = (1, 2)
        
        w = params['convolution3_weight'].asnumpy().reshape(-1)
        b = params['convolution3_bias'].asnumpy().reshape(-1)
        bag += [w, b]
        conv3 = network.add_convolution(pool2.get_output(0), 16, (3,3), w, b)
        conv3.stride = (1, 1)
        conv3.padding = (1, 1)
        print('conv3', conv3.get_output(0).shape)
         
        actv3 = network.add_activation(conv3.get_output(0), trt.ActivationType.RELU)
        pool3 = network.add_pooling(actv3.get_output(0), trt.PoolingType.MAX, (1, 2))
        pool3.stride = (1, 2)
         
        w = params['convolution4_weight'].asnumpy().reshape(-1)
        b = params['convolution4_bias'].asnumpy().reshape(-1)
        bag += [w, b]
        conv4 = network.add_convolution(pool3.get_output(0), 16, (3,2), w, b)
        conv4.stride = (1, 1)
        conv4.padding = (1, 0)
        print('conv4', conv4.get_output(0).shape)
        
        shuf0 = network.add_shuffle(conv4.get_output(0))
        shuf0.first_transpose = (1, 0, 2)
        shuf0.reshape_dims = (self.time_step, 16)
        print('shuf0', shuf0.get_output(0).shape)
        
        network.mark_output(shuf0.get_output(0))
    
        builder.max_batch_size = 64
        builder.max_workspace_size = 500 << 20
        builder.int8_mode = True
        builder.int8_calibrator = calibrator.CaptcharEntropyCalibrator("ocr_int8.calib", 64)
    
        return builder.build_cuda_engine(network)    
        
class TrtPredictor_Lstm(TrtPredictor):
    def create_engine(self, builder):
        network = builder.create_network()
        data = network.add_input("data", trt.DataType.FLOAT, (self.time_step, 16))
        bag = []

        lstm = network.add_rnn_v2(data, 2, 100, self.time_step, trt.RNNOperation.LSTM)
        lstm.direction = trt.RNNDirection.BIDIRECTION
        for i in range(8):
            layer = i // 2
            isW = True if i % 2 == 0 else False             
            
            param_name = 'l{}_{}_'.format(layer, 'i2h' if isW else 'h2h')
            all_w = [w.reshape(-1) for w in np.split(params[param_name + 'weight'].asnumpy(), 4)]
            all_b = [w.reshape(-1) for w in np.split(params[param_name + 'bias'].asnumpy(), 4)]
            bag += [all_w, all_b]
        
            for i, g in zip(range(4), 
                    [trt.RNNGateType.INPUT, trt.RNNGateType.CELL, trt.RNNGateType.FORGET, trt.RNNGateType.OUTPUT]):
                lstm.set_weights_for_gate(layer, g, isW, all_w[i])
                lstm.set_bias_for_gate(layer, g, isW, all_b[i])
        
        shuf1 = network.add_shuffle(lstm.get_output(0))
        shuf1.reshape_dims = (-1, 1, 1, 200)
      
        n_char = 27
        w = params['pred_fc_weight'].asnumpy().reshape(-1)
        b = params['pred_fc_bias'].asnumpy().reshape(-1)
        bag += [w, b]
        fc = network.add_fully_connected(shuf1.get_output(0), n_char, w, b)
        
        print('fc', fc.get_output(0).shape)
        topk = network.add_topk(fc.get_output(0), trt.TopKOperation.MAX, 1, 2)
      
        network.mark_output(topk.get_output(1))
        topk.get_output(1).dtype = trt.DataType.INT32
    
        builder.max_batch_size = 64
        builder.max_workspace_size = 500 << 20
        builder.fp16_mode = True
    
        return builder.build_cuda_engine(network)

def read_img(path):
    """ Reads image specified by path into numpy.ndarray"""
    img = cv2.imread(path, 0)
    h = 32
    img = cv2.resize(img, ((img.shape[1] * h + img.shape[0] - 1) // img.shape[0], h)).transpose(1, 0)
    img = img.reshape(-1).reshape((1,) + img.shape)
    img = np.multiply(img, 1 / 255.0)
    return img.astype(np.float32)

np.set_printoptions(threshold=np.inf)

def predict(data):
    batch_size = len(data)
    print("TRT batch_size:", batch_size)
    
    data = np.array(data)
    d_input = cuda.mem_alloc(data.nbytes)
    cuda.memcpy_htod(d_input, data)

    d_tensor_into_lstm = cuda.mem_alloc(time_step * 16 * 4)
    output_index = np.empty((batch_size, time_step), dtype = np.int32)
    d_output = cuda.mem_alloc(output_index.nbytes)
    
    predictor_conv = TrtPredictor_Conv(time_step)
    predictor_lstm = TrtPredictor_Lstm(time_step)
    
    n_round = 1
    time0 = time.time()
    for _ in range(n_round):
        #start = time.time()
        predictor_conv.infer(batch_size, d_input, d_tensor_into_lstm)
        predictor_lstm.infer(batch_size, d_tensor_into_lstm, d_output)
        #print "tensorrt forward batch spend : {}".format((time.time() - start) / 1.0)
    print("TRT average:", (time.time() - time0) * 1.0 / n_round)

    cuda.memcpy_dtoh(output_index, d_output)
    print(output_index)
    for k in range(len(output_index)):
        cur = None
        seq = []
        for i in output_index[k]:
            if cur == i:
                continue
            seq.append(i)
            cur = i
        print([chr(ord('a') + i - 1) for i in seq if i != 0])

img = read_img("test.jpg")
predict([img] * batch_size)

cuda.Context.pop()