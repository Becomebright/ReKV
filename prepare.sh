conda create -n rekv python=3.11 -y
conda activate rekv

pip install -U torch torchvision torchaudio
pip install -U "git+https://github.com/huggingface/transformers.git@66bc4def9505fa7c7fe4aa7a248c34a026bb552b"
pip install -U psutil
pip install -e . --no-build-isolation

cd model/longva
pip install -e .
