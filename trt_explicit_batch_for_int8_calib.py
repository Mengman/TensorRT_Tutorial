import numpy as np
import tensorrt as trt
import pycuda.autoinit
import pycuda.driver as cuda
import sys
import time
import mxnet as mx
import cv2
import calibrator

logger = trt.Logger(trt.Logger.INFO)

batch_size = 1
print("TRT batch_size:", batch_size)

max_batch_size = 16

i_gpu = 0
print("GPU in use:", cuda.Device(i_gpu).name())
cuda.Device(i_gpu).make_context()

_, params, extra_params = mx.model.load_checkpoint('ocr', 869)

class TrtPredictor:    
    def __init__(self, b_fp16):
        self.b_fp16 = b_fp16
        
        trt_file_path = "ocr_{}.trt".format("fp16" if b_fp16 else "fp32")
#         if os.path.isfile(trt_file_path):
        if False:
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
    
    def build_engine_str(self, b_fp16):
        with trt.Builder(logger) as builder, self.create_engine(builder, b_fp16) as engine:
            return engine.serialize()
    
class TrtPredictor_Conv(TrtPredictor):
    def create_engine(self, builder, b_fp16):
        network = builder.create_network(1)
#         data = network.add_input("data", trt.DataType.FLOAT, (-1, 1, -1, 32))
        data = network.add_input("data", trt.DataType.FLOAT, (1, 1, 80, 32))
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
        
        shape0 = network.add_shape(conv4.get_output(0)).get_output(0)
        print('shape0', shape0.shape)
        
        n0 = network.add_slice(shape0, (0,), (1,), (1,)).get_output(0)
        c0 = network.add_slice(shape0, (1,), (1,), (1,)).get_output(0)
        h0 = network.add_slice(shape0, (2,), (1,), (1,)).get_output(0)
        w0 = network.add_slice(shape0, (3,), (1,), (1,)).get_output(0)
        
        cw0 = network.add_elementwise(c0, w0, trt.ElementWiseOperation.PROD).get_output(0)
        shape1 = network.add_concatenation([n0, h0, cw0]).get_output(0)
        print(shape1.shape)
        
        shuf0 = network.add_shuffle(conv4.get_output(0))
        shuf0.first_transpose = (0, 2, 1, 3)
        shuf0.set_input(1, shape1)
        print('shuf0', shuf0.get_output(0).shape)
        
        network.mark_output(shuf0.get_output(0))
            
        op = builder.create_optimization_profile()
        op.set_shape('data', (1, 1, 1, 32), (1, 1, 80, 32), (max_batch_size, 1, 320, 32))
        config = builder.create_builder_config()
        config.add_optimization_profile(op)

        builder.max_workspace_size = 1 << 30
        builder.fp16_mode = b_fp16
        builder.int8_mode = True
        builder.int8_calibrator = calibrator.CaptcharEntropyCalibrator("ocr_int8.calib", 1)
    
#         return builder.build_engine(network, config)
        return builder.build_cuda_engine(network)
    
    def infer(self, input_shape, d_input, d_output):    
        bindings = [int(d_input), int(d_output)]
        self.context.set_binding_shape(0, input_shape)
        self.context.execute_async_v2(bindings, 0)

max_time_step = 80

class TrtPredictor_Lstm(TrtPredictor):
    def create_engine(self, builder, b_fp16):
        network = builder.create_network()
        data = network.add_input("data", trt.DataType.FLOAT, (1, max_time_step, 16))
        time_step = network.add_input("time_step", trt.DataType.INT32, (1,))
        bag = []

        lstm = network.add_rnn_v2(data, 2, 100, max_time_step, trt.RNNOperation.LSTM)
        lstm.direction = trt.RNNDirection.BIDIRECTION
        lstm.seq_lengths = time_step
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
    
        builder.max_batch_size = max_batch_size
        builder.max_workspace_size = 1 << 30
        builder.fp16_mode = b_fp16
    
        return builder.build_cuda_engine(network)

    def infer(self, batch_size, d_input, d_time_step, d_output):    
        bindings = [int(d_input), int(d_time_step), int(d_output)]
        self.context.execute_async(batch_size, bindings, 0)

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
    print('data shape:', data.shape)
    batch_size = data.shape[0]
    print("TRT batch_size:", batch_size)
    
    d_input = cuda.mem_alloc(data.nbytes)
    cuda.memcpy_htod(d_input, data)
    
    time_step = data.shape[2] // 4
    print('timestep:', time_step)
    h_time_step = np.array([time_step] * batch_size, np.int32)
    d_time_step = cuda.mem_alloc(h_time_step.nbytes)
    cuda.memcpy_htod(d_time_step, h_time_step)

    d_cnn_output = cuda.mem_alloc(batch_size * time_step * 16 * 4)
    d_lstm_input = cuda.mem_alloc(batch_size * max_time_step * 16 * 4)
    output = np.empty((batch_size, max_time_step), dtype = np.int32)
    d_output = cuda.mem_alloc(output.nbytes)
    
    predictor_conv = TrtPredictor_Conv(False)
    predictor_lstm = TrtPredictor_Lstm(False)
    
    n_round = 1
    time0 = time.time()
    for _ in range(n_round):
        #start = time.time()
        predictor_conv.infer(data.shape, d_input, d_cnn_output)
        
        m = cuda.Memcpy2D()
        m.src_pitch = time_step * 16 * 4
        m.dst_pitch = max_time_step * 16 * 4
        m.width_in_bytes = m.src_pitch
        m.height = batch_size
        m.set_src_device(d_cnn_output)
        m.set_dst_device(d_lstm_input)
        m(False)

        predictor_lstm.infer(batch_size, d_lstm_input, d_time_step, d_output)
        #print "tensorrt forward batch spend : {}".format((time.time() - start) / 1.0)
    cuda.Context.synchronize()
    print("TRT average:", (time.time() - time0) * 1.0 / n_round)

    cuda.memcpy_dtoh(output, d_output)
    print(output)
    for k in range(len(output)):
        cur = None
        seq = []
        for i in output[k]:
            if cur == i:
                continue
            seq.append(i)
            cur = i
        print([chr(ord('a') + i - 1) for i in seq if i != 0])

img = read_img("test.jpg")
predict(np.array([img] * batch_size, np.float32))

cuda.Context.pop()
