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

logger = trt.Logger(trt.Logger.INFO)

batch_size = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 1
print("TRT batch_size:", batch_size)

i_gpu = int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2].isdigit() else 0
print("GPU in use:", cuda.Device(i_gpu).name())
cuda.Device(i_gpu).make_context()

b_fp16 = len(sys.argv) > 3 and sys.argv[3].isdigit() and int(sys.argv[3]) == 1
print("Use FP16:", b_fp16)

time_step = int(sys.argv[4]) if len(sys.argv) > 4 and sys.argv[4].isdigit() else 20
print("Time step:", time_step)

class TrtPredictor:
    @classmethod
    def read_float(cls, wf, n):
        l = []
        for _ in range(n):
            l.append(*struct.unpack('f', wf.read(4)))
        return np.array(l, dtype=np.float32)
    
    def __init__(self, b_fp16, time_step):
        self.b_fp16 = b_fp16
        self.time_step = time_step
        
        trt_file_path = "ocr_{}.trt".format(time_step)
        if os.path.isfile(trt_file_path):
            with open(trt_file_path, 'rb') as f:
                engine_str = f.read()
        else:
            engine_str = self.build_engine_str(b_fp16)
            with open(trt_file_path, 'wb') as f:
                f.write(engine_str)
        
        with trt.Runtime(logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(engine_str)
        self.context = self.engine.create_execution_context()
        
    def __del__(self):
        self.context = None
        self.engine = None
        
    def create_engine(self, builder, b_fp16):
        _, params, extra_params = mx.model.load_checkpoint('ocr', 869)
        
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
        shuf0.reshape_dims = (-1, self.time_step, 16)
        
        lstm = network.add_rnn_v2(shuf0.get_output(0), 2, 100, self.time_step, trt.RNNOperation.LSTM)
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
        builder.fp16_mode = b_fp16
    
        return builder.build_cuda_engine(network)    
    
    def build_engine_str(self, b_fp16):
        with trt.Builder(logger) as builder, self.create_engine(builder, b_fp16) as engine:
            return engine.serialize()
    
    def infer(self, data, batch_size, d_input, d_output, output_index):    
        bindings = [int(d_input), int(d_output)]    
        cuda.memcpy_htod(d_input, data)
        self.context.execute(batch_size, bindings)
        cuda.memcpy_dtoh(output_index, d_output)


def read_img(path):
    """ Reads image specified by path into numpy.ndarray"""
    img = cv2.copyMakeBorder(cv2.resize(cv2.imread(path, 0), (80, 30)), 1, 1, 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    img = img.transpose(1, 0).reshape((1, 80, 32)).reshape(-1).reshape((1, 80, 32))
    img = np.multiply(img, 1 / 255.0)
    return img.astype(np.float32)

np.set_printoptions(threshold=np.inf)

def predict(data):
    batch_size = len(data)
    print("TRT batch_size:", batch_size)
    
    data = np.array(data)
    d_input = cuda.mem_alloc(data.nbytes)
    shape = (batch_size, time_step)
    output_index = np.empty(shape, dtype = np.int32)
    d_output = cuda.mem_alloc(output_index.nbytes)
    
    predictor = TrtPredictor(b_fp16, time_step)
    
    n_round = 100
    time0 = time.time()
    for _ in range(n_round):
        #start = time.time()
        predictor.infer(data, batch_size, d_input, d_output, output_index)
        #print "tensorrt forward batch spend : {}".format((time.time() - start) / 1.0) 
    print("TRT average:", (time.time() - time0) * 1.0 / n_round)
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