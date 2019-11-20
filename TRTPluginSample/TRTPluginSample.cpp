// nvcc -g -arch=sm_70 -I/home/gji/Devtools/TensorRT-5.1.2.2/include AddPlugin.cu TRTPluginSample.cpp -L/home/gji/Devtools/TensorRT-5.1.2.2/lib -lnvinfer

#include "AddPlugin.h"
#include <vector>
#include <iostream>
#include <iomanip>
#include <cuda_runtime_api.h>

using namespace nvinfer1;
using namespace std;

class Logger : public nvinfer1::ILogger
{
public:
    Logger(Severity severity = Severity::kINFO) : reportableSeverity(severity) {}

    void log(Severity severity, const char* msg) override
    {
        if (severity > reportableSeverity) return;

        switch (severity)
        {
        case Severity::kINTERNAL_ERROR: std::cerr << "INTERNAL_ERROR: "; break;
        case Severity::kERROR: std::cerr << "ERROR: "; break;
        case Severity::kWARNING: std::cerr << "WARNING: "; break;
        case Severity::kINFO: std::cerr << "INFO: "; break;
        default: std::cerr << "UNKNOWN: "; break;
        }
        std::cerr << msg << std::endl;
    }

    Severity reportableSeverity;
};
static Logger gLogger(ILogger::Severity::kINFO);

inline bool check(cudaError_t e, int iLine, const char *szFile) {
    if (e != cudaSuccess) {
        cout << "CUDA runtime API error " << cudaGetErrorName(e) << " at line " << iLine << " in file " << szFile << endl;
        return false;
    }
    return true;
}

#define ck(call) check(call, __LINE__, __FILE__)

const char* INPUT_BLOB_NAME = "input";
const char* OUTPUT_BLOB_NAME = "output";

void print(const vector<float> &v, int row, int col) {
    for (int i = 0; i < row; i++) {
        for (int j = 0; j < col; j++) {
            cout << fixed << setprecision(3) << v[i * col + j] << " ";
        }
        cout << endl;
    }
    cout << endl;
}

void printDim(Dims const &dim) {
    for (int i = 0; i < dim.nbDims; i++) {
        cout << dim.d[i] << " ";
    }
    cout << endl;
}

void fill(vector<float> &v) {
    for (int i = 0; i < v.size(); i++) {
        v[i] = i;
    }
}

ICudaEngine* buildEngine(int nBatch, int nChannel, int nHeight, int nWidth, float valueToAdd, IBuilder* builder) {
    const int maxBatchSize = 64;
    INetworkDefinition* network = builder->createNetwork();
    ITensor *aInputs[] = {
        network->addInput(INPUT_BLOB_NAME, DataType::kFLOAT, Dims3{nChannel, nHeight, nWidth}),
    };

    Weights w{DataType::kFLOAT, &valueToAdd, 1};
    LayerNormPlugin *plugin = new LayerNormPlugin(w);
    IPluginV2Layer *pluginLayer = network->addPluginV2(aInputs, sizeof(aInputs) / sizeof(aInputs[0]), *plugin);
    pluginLayer->getOutput(0)->setName(OUTPUT_BLOB_NAME);
    network->markOutput(*pluginLayer->getOutput(0));

    printDim(pluginLayer->getOutput(0)->getDimensions());

    builder->setMaxBatchSize(maxBatchSize);
    builder->setMaxWorkspaceSize(1 << 20);
    builder->setFp16Mode(false);
    ICudaEngine* engine = builder->buildCudaEngine(*network);
    network->destroy();

    return engine;
}

void doInference(vector<float> const &X, vector<float> &output,
        int nBatch, int nChannel, int nHeight, int nWidth, float valueToAdd) {
    IBuilder* builder = createInferBuilder(gLogger);
    ICudaEngine* engine = buildEngine(nBatch, nChannel, nHeight, nWidth, valueToAdd, builder);
    IExecutionContext* context = engine->createExecutionContext();
    
    assert(engine->getNbBindings() == 2);
    void* buffers[2];

    const int inputIndex = engine->getBindingIndex(INPUT_BLOB_NAME);
    const int outputIndex = engine->getBindingIndex(OUTPUT_BLOB_NAME);

    ck(cudaMalloc(&buffers[inputIndex], X.size() * sizeof(float)));
    ck(cudaMalloc(&buffers[outputIndex], output.size() * sizeof(float)));

    cudaStream_t stream;
    ck(cudaStreamCreate(&stream));

    ck(cudaMemcpyAsync(buffers[inputIndex], X.data(), X.size() * sizeof(float), cudaMemcpyHostToDevice, stream));
    context->enqueue(nBatch, buffers, stream, nullptr);
    ck(cudaMemcpyAsync(output.data(), buffers[outputIndex], output.size() * sizeof(float), cudaMemcpyDeviceToHost, stream));
    cudaStreamSynchronize(stream);

    cudaStreamDestroy(stream);
    ck(cudaFree(buffers[inputIndex]));
    ck(cudaFree(buffers[outputIndex]));
    
    builder->destroy();
}

int main(int argc, char** argv) {
    const int nBatch = 2, nChannel = 3, nHeight = 2, nWidth = 2;
    float valueToAdd = 100.0f;
    int iDevice = 0;

    if (argc >= 2) iDevice = atoi(argv[1]);
    ck(cudaSetDevice(iDevice));
    cudaDeviceProp prop = {};
    ck(cudaGetDeviceProperties(&prop, iDevice));
    cout << "Using " << prop.name << endl;

    vector<float> X(nBatch * nChannel * nHeight * nWidth), output(nBatch * nChannel * nHeight * nWidth);
    fill(X);

    doInference(X, output, nBatch, nChannel, nHeight, nWidth, valueToAdd);
    print(output, nBatch * nChannel * nHeight, nWidth);

    return 0;
}
