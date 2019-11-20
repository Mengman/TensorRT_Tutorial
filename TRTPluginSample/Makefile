CUDA_PATH = /usr/local/cuda

GCC = g++
NVCC = $(CUDA_PATH)/bin/nvcc
CCFLAGS = -g -std=c++11 -DNDEBUG
INCLUDES += -I. -I$(CUDA_PATH)/include -I/home/gji/Devtools/TensorRT-6.0.1.5/include
LDFLAGS := -L/usr/local/cuda/lib64 -L/home/gji/Devtools/TensorRT-6.0.1.5/lib $(LDFLAGS)
LDFLAGS += -lnvinfer -lcudart

BIN = TRTPluginSample
SO = AddPlugin.so
OBJ = $(shell find .. -name '*.o')
DEP = $(OBJ:.o=.d)

all: $(BIN) $(SO) $(CUDA_BIN)

TRTPluginSample: TRTPluginSample.o AddPlugin.o
AddPlugin.so: AddPlugin.o

-include $(DEP)

clean:
	rm -rf $(BIN) $(SO) $(CUDA_BIN) $(OBJ) $(DEP)

%.o: %.cpp
	$(GCC) $(CCFLAGS) -fPIC -MD -MP $(INCLUDES) -o $@ -c $<

%.o: %.cu
	$(NVCC) $(CCFLAGS) -M -MT $@ $(INCLUDES) -o $(@:.o=.d) $<
	$(NVCC) $(CCFLAGS) $(INCLUDES) -Xcompiler -fPIC -arch=sm_70 -o $@ -c $<

$(SO):
	$(GCC) $(CCFLAGS) -shared -o $@ $+ $(LDFLAGS)

$(BIN):
	$(GCC) $(CCFLAGS) -o $@ $+ $(LDFLAGS)

test: all
	./TRTPluginSample
	python3 trt_plugin_sample.py
	python3 trt_plugin_sample.py 1
