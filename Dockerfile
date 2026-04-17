# can be set at build time with --build-arg BASE_IMAGE=your_image:tag
# for arm64 use: python:3.12.13-slim-bookworm
# for x86_64 use: nvidia/cuda:13.1.1-devel-ubuntu24.04 or nvidia/cuda:13.1.1-runtime-ubuntu24.04
# for CUDA 12: nvidia/cuda:12.8.1-runtime-ubuntu24.04
ARG BASE_IMAGE=nvidia/cuda:13.1.1-runtime-ubuntu24.04
FROM ${BASE_IMAGE}

ARG CUDAVER=130

# automatically filled by docker build
ARG TARGETARCH

# Set the working directory in the container
WORKDIR /app

SHELL ["/bin/bash", "-c"] 

# install require linux packages
COPY install_pkg.sh /app
RUN /app/install_pkg.sh

# Copy the current directory contents into the container at /app
COPY pyproject.toml LICENSE  README.md /app/

# create a python 12 venv	
RUN python3.12 -m venv venv
ENV PATH="venv/bin:$PATH"

# upgrade pip for --group support
RUN pip install -U pip   

# install the correct torch first 
RUN if [[ "$TARGETARCH" = "amd64" ]]; then \
        echo "Build for AMD64 with CUDA: $CUDAVER" && \
        if [[ "$CUDAVER" = "130" ]]; then \
          pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130; \
        else \
          pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128; \
        fi; \
    elif [[ "$TARGETARCH" = "arm64" ]]; then \
        echo "Build for ARM64 with NO CUDA ..." && \
        pip install torch torchvision torchaudio--index-url https://download.pytorch.org/whl/cpu; \
    else \
        echo "Unsupported architecture: $TARGETARCH" >&2 && exit 1; \
    fi

# try this if you want but faster-whisper is faster
#RUN bash src/anytran/pywhispercppcuda.sh
RUN pip install --group all

COPY Dockerfile .dockerignore entrypt.sh /app/
COPY tests /app/tests
COPY doc /app/doc
COPY src /app/src

RUN pip install -e .

# we have to do this in case the CPU version was installed
# make sure to ignore the unintall in case onnxruntime is not installed
RUN if [[ "$TARGETARCH" = "amd64" ]]; then \
        echo "Build for AMD64 with CUDA ..." && \
        pip uninstall onnxruntime onnxruntime-gpu -y && \
        pip install onnxruntime-gpu; \
    fi

ENTRYPOINT ["./entrypt.sh"]

CMD []
 
