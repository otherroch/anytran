# Use an official Python runtime as a parent image
# FROM python:3.12-slim
FROM nvidia/cuda:13.1.0-devel-ubuntu24.04
# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-dev python3-venv python3-pip portaudio19-dev build-essential \
     ffmpeg git cmake && \
    rm -rf /var/lib/apt/lists/*

# Install any needed packages specified in requirements.txt (if you had one)
RUN python3 -m venv venv
ENV PATH="venv/bin:$PATH"
#RUN bash src/anytran/pywhispercppcuda.sh
RUN pip install -e .[all]

ENTRYPOINT ["./entrypt.sh"]

CMD []
 
