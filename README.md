
# quick start process
conda create -n userrl python=3.12
pip install torch==2.7.0
pip install -e .
pip install vllm==0.9.1
pip install sglang[srt]==0.4.7
pip install flash-attn --no-build-isolation

# install all the gyms
bash install_gyms.sh