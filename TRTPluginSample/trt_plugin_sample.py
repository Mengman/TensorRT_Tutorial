from __future__ import print_function
import numpy as np
import tensorrt as trt
import pycuda.autoinit
import pycuda.driver as cuda
import ctypes
import os
import sys

logger = trt.Logger(trt.Logger.INFO)
ctypes.cdll.LoadLibrary('./AddPlugin.so')

def get_plugin_creator(plugin_name):
    trt.init_libnvinfer_plugins(logger, '')
    plugin_creator_list = trt.get_plugin_registry().plugin_creator_list
    plugin_creator = None
    for c in plugin_creator_list:
        if c.name == plugin_name:
            plugin_creator = c
    return plugin_creator

def build_engine(shape):
    plugin_creator = get_plugin_creator('AddPlugin')
    if plugin_creator == None:
        print('Plugin not found. Exiting')
        exit()

    builder = trt.Builder(logger)
    builder.max_batch_size = 1024
    builder.max_workspace_size = 1 << 20
    builder.fp16_mode = use_fp16
    network = builder.create_network()
    
    tensor = network.add_input('data', trt.DataType.FLOAT, shape)
    for _ in range(10):
        tensor = network.add_plugin_v2(
            [tensor], 
            plugin_creator.create_plugin('AddPlugin', trt.PluginFieldCollection([
                trt.PluginField('valueToAdd', np.array([10.0], dtype=np.float32), trt.PluginFieldType.FLOAT32)
            ]))
        ).get_output(0)
    
    network.mark_output(tensor)
    return builder.build_cuda_engine(network)
    
def run_trt(data):
    trt_file_path = 'sample.{}.trt'.format('fp16' if use_fp16 else 'fp32')
    if os.path.isfile(trt_file_path):
        with open(trt_file_path, 'rb') as f:
            engine_str = f.read()
        with trt.Runtime(logger) as runtime:
            engine = runtime.deserialize_cuda_engine(engine_str)
        print('Engine loaded')
    else:
        engine = build_engine(data.shape[1:])
        with open(trt_file_path, 'wb') as f:
            f.write(engine.serialize())
        print('Engine saved')

    context = engine.create_execution_context()
    
    d_data = cuda.mem_alloc(data.nbytes)
    output = np.zeros_like(data)
    d_output = cuda.mem_alloc(data.nbytes)
    
    cuda.memcpy_htod(d_data, data)
    bindings = [int(d_data), int(d_output)]    
    context.execute(data.shape[0], bindings)
    cuda.memcpy_dtoh(output, d_output)
    
    return output

use_fp16 = len(sys.argv) > 1 and sys.argv[1].isdigit() and int(sys.argv[1]) == 1
print('Use FP16:', use_fp16)
print(
    run_trt(
        np.ones((2, 4096, 4, 4), np.float32)
    )[0, 0]
)
