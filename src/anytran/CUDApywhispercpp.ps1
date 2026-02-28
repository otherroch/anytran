$env:GGML_CUDA=1
$env:NO_REPAIR=1
$env:CMAKE_BUILD_PARALLEL_LEVEL = "12"
cd src
git clone https://github.com/absadiki/pywhispercpp.git 
cd pywhispercpp 
git submodule update --init --recursive 
pip install -e .