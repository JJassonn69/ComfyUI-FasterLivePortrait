FROM nvidia/cuda:12.2.2-cudnn8-devel-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive
ARG TRT_VERSION=10.6.0.26
ARG CUDA_VERSION=12.6

ENV TensorRT_ROOT=/opt/TensorRT-${TRT_VERSION}
ENV LD_LIBRARY_PATH=${TensorRT_ROOT}/lib:${LD_LIBRARY_PATH}
ENV PATH=${TensorRT_ROOT}/bin:${PATH}

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip wget ca-certificates git cmake build-essential \
        libprotobuf-dev protobuf-compiler && \
    rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir numpy onnx onnxruntime

WORKDIR /opt
RUN wget --progress=dot:giga \
    https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/10.6.0/tars/TensorRT-${TRT_VERSION}.Linux.x86_64-gnu.cuda-${CUDA_VERSION}.tar.gz \
 && tar -xzf TensorRT-${TRT_VERSION}.Linux.x86_64-gnu.cuda-${CUDA_VERSION}.tar.gz \
 && rm TensorRT-${TRT_VERSION}.Linux.x86_64-gnu.cuda-${CUDA_VERSION}.tar.gz

 # after the TensorRT tar extraction
RUN echo "${TensorRT_ROOT}/lib" > /etc/ld.so.conf.d/tensorrt.conf \
&& ldconfig

RUN git clone https://github.com/SeanWangJS/grid-sample3d-trt-plugin.git
WORKDIR /opt/grid-sample3d-trt-plugin

ENV TensorRT_ROOT=/opt/TensorRT-${TRT_VERSION}
ENV LD_LIBRARY_PATH=${TensorRT_ROOT}/lib:$LD_LIBRARY_PATH

# RUN mkdir -p build && cd build && \
#     cmake .. \
#       -DCMAKE_BUILD_TYPE=Release \
#       -DTensorRT_ROOT=${TensorRT_ROOT} \
#       -DCMAKE_PREFIX_PATH=${TensorRT_ROOT} \
#       -DCMAKE_LIBRARY_PATH=${TensorRT_ROOT}/lib \
#       -DCMAKE_CXX_FLAGS="-I${TensorRT_ROOT}/include" \
#       -DCMAKE_CUDA_FLAGS="-I${TensorRT_ROOT}/include" && \
#     cmake --build . --parallel $(nproc)

WORKDIR /workspace

CMD ["/bin/bash"]
