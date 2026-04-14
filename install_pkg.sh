#!/bin/bash

apt-get update 

# should all be execute by root (or sudo)
apt-get install -y --no-install-recommends \
   python3-dev python3-venv python3-pip portaudio19-dev build-essential 

apt-get install -y --no-install-recommends   ffmpeg git cmake \
    libavdevice-dev libavformat-dev libavcodec-dev libavutil-dev libswscale-dev libavfilter-dev   

rm -rf /var/lib/apt/lists/*
