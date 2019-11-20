#include "NvInfer.h"
#include <iostream>
#include <cstring>
#include <assert.h>

using namespace std;

class LayerNormPlugin: public nvinfer1::IPluginV2 {
public:
    LayerNormPlugin(nvinfer1::Weights valueToAdd) {
        m.valueToAdd = *(float *)valueToAdd.values;
    }
    
    LayerNormPlugin(const void *buffer, size_t length) {
        memcpy(&m, buffer, sizeof(m));
    }
    virtual size_t getSerializationSize() const override {
        return sizeof(m);
    }
    virtual void serialize(void *buffer) const override {
        memcpy(buffer, &m, sizeof(m));
    }
    
    nvinfer1::IPluginV2* clone() const override {
        return new LayerNormPlugin(&m, sizeof(m));
    }

    bool supportsFormat(nvinfer1::DataType type, nvinfer1::PluginFormat format) const override {
        return (type == nvinfer1::DataType::kFLOAT || type == nvinfer1::DataType::kHALF) 
                && format == nvinfer1::PluginFormat::kNCHW;
    }
    int getNbOutputs() const override {
        return 1;
    }
    nvinfer1::Dims getOutputDimensions(int index, const nvinfer1::Dims* pInputDim, int nInputDim) override {
        return pInputDim[0];
    }

    void configureWithFormat(const nvinfer1::Dims* pInputDim, int nInputDim, const nvinfer1::Dims* pOutputDim, 
            int nOutputDim, nvinfer1::DataType dataType, nvinfer1::PluginFormat pluginFormat, int maxBatchSize) override {
        m.dataType = dataType;
        m.inputDim = pInputDim[0];
    }
    size_t getWorkspaceSize(int nBatch) const override {return 0;}
    int enqueue(int nBatch, const void * const *inputs, void **outputs, void* workspace, cudaStream_t stream) override;

    int initialize() override {return 0;}
    void terminate() override {}
    void destroy() override { delete this; }
    void setPluginNamespace(const char* szNamespace) override {}
    const char* getPluginNamespace() const override {return "";}
    const char* getPluginType() const override {return "AddPlugin";}
    const char* getPluginVersion() const override {return "0";}

private:
    struct {
        nvinfer1::DataType dataType;
        nvinfer1::Dims inputDim;
        float valueToAdd;
    } m;
};

class AddPluginCreator : public nvinfer1::IPluginCreator {
public:
    nvinfer1::IPluginV2* deserializePlugin(const char* name, const void* serialData, size_t serialLength) override {
        return new LayerNormPlugin(serialData, serialLength);
    }
    
    const char* getPluginName() const override {return "AddPlugin";}
    const char* getPluginVersion() const override {return "0";}

    void setPluginNamespace(const char* szNamespace) override {}
    const char* getPluginNamespace() const override {return "";}
    
    const nvinfer1::PluginFieldCollection* getFieldNames() override {
        std::cout << __FUNCTION__ << std::endl;
        return nullptr;
    }
    nvinfer1::IPluginV2* createPlugin(const char* name, const nvinfer1::PluginFieldCollection* fc) override {
        std::cout << __FUNCTION__ << std::endl;
        float valueToAdd = 0;
        for (int i = 0; i < fc->nbFields; i++) {
            if (!strcmp(fc->fields[i].name, "valueToAdd")) {
                valueToAdd = *(float *)fc->fields[i].data;
            }
        }
        return new LayerNormPlugin({nvinfer1::DataType::kFLOAT, &valueToAdd, 1});
    }
};
