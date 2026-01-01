export PATH=/usr/local/cuda-9.0/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-9.0/lib64:${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
conda activate tf1.5
CUDA_VISIBLE_DEVICES='2' OPENCV_IO_ENABLE_OPENEXR=1 python generate_train_dataset.py