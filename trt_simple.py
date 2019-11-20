import numpy as np
import tensorrt as trt
import pycuda.autoinit
import pycuda.driver as cuda

i_gpu = 0
print("GPU in use:", cuda.Device(i_gpu).name())
cuda.Device(i_gpu).make_context()

#input
batch = 2
channel = 2
height = 5
width = 10

logger = trt.Logger(trt.Logger.WARNING)

builder = trt.Builder(logger)
builder.max_batch_size = 32
builder.max_workspace_size = 0 << 20
network = builder.create_network()

data = network.add_input("data", trt.DataType.FLOAT, (channel, height, width))

#output
c = 3
kh = 5
kw = 5
w0 = np.arange(channel * c * kh * kw, dtype=np.float32)
b0 = np.arange(c, dtype=np.float32)
conv0 = network.add_convolution(data, c, (kh, kw), trt.Weights(w0), trt.Weights(b0))
conv0.stride = (1, 1)
conv0.padding = (1, 1)

actv0 = network.add_activation(conv0.get_output(0), trt.ActivationType.TANH)

wh = 2
ww = 2
pool0 = network.add_pooling(actv0.get_output(0), trt.PoolingType.AVERAGE, (wh, ww))
pool0.stride = (wh, ww)
pool0.padding = (0, 0)

nOutput = 10
w1 = np.ones(nOutput * 12, dtype=np.float32)
b1 = np.ones(nOutput, dtype=np.float32)
fc = network.add_fully_connected(pool0.get_output(0), nOutput, trt.Weights(w1), trt.Weights(b1))

network.mark_output(fc.get_output(0))
engine = builder.build_cuda_engine(network)

h_input = np.empty(trt.volume(engine.get_binding_shape(0)), dtype=np.float32)
h_output = np.empty(batch * trt.volume(engine.get_binding_shape(1)), dtype=np.float32)
d_input = cuda.mem_alloc(h_input.nbytes)
d_output = cuda.mem_alloc(h_output.nbytes)

h_input[:] = range(h_input.size)

context = engine.create_execution_context()

cuda.memcpy_htod_async(d_input, h_input)
context.execute_async(batch, [int(d_input), int(d_output)], 0)
cuda.memcpy_dtoh(h_output, d_output)

output = h_output.reshape((batch,) + tuple(engine.get_binding_shape(1)))
print(output)

cuda.Context.pop()
