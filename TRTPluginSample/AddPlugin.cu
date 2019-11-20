// nvcc -g -arch=sm_70 -Xcompiler -fPIC -shared -o AddPlugin.so -I/home/gji/Devtools/TensorRT-5.1.2.2/include AddPlugin.cu -L/home/gji/Devtools/TensorRT-5.1.2.2/lib -lnvinfer

#include "AddPlugin.h"
#include "cuda_fp16.h"
#include <thread>

template<typename T>
__global__ void AddValue(T *pDst, T *pSrc, int n, T valueToAdd) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x >= n) return;
    pDst[x] = pSrc[x] + valueToAdd;
}

int LayerNormPlugin::enqueue(int nBatch, const void * const *inputs, void **outputs, void* workspace, cudaStream_t stream) {
    int n = nBatch;
    for (int i = 0; i < m.inputDim.nbDims; i++) {
        n *= m.inputDim.d[i];
    }
    if (m.dataType == nvinfer1::DataType::kFLOAT) {
        std::cout << "Running fp32 kernel" << std::endl;
        AddValue<<<(n + 1023) / 1024, 1024>>>((float *)outputs[0], (float *)inputs[0], n, m.valueToAdd);
    } else {
        std::cout << "Running fp16 kernel" << std::endl;
        AddValue<<<(n + 1023) / 1024, 1024>>>((__half *)outputs[0], (__half *)inputs[0], n, (__half)m.valueToAdd);
    }
    return 0;
}

REGISTER_TENSORRT_PLUGIN(AddPluginCreator);
