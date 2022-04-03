# tf-sandbox
> learning tf

- [tf-sandbox](#tf-sandbox)
  - [setup](#setup)
  - [conda for dummies](#conda-for-dummies)
  - [monitor gpu usage](#monitor-gpu-usage)
  - [model conversion](#model-conversion)
## setup

```sh
# install miniconda3 for conda
yay -S miniconda3

# closed source nvidia drivers
sudo mhwd -a pci nonfree 0300

# conda environment (contains all the python deps)
# NOTE: setting constraints speeds up the conda dep resolver
# if it takes longer than 2 mins cancel it
conda create --name tf python=3.7 tensorflow=2.7.0 pillow keras

# activate the conda environment
conda activate tf

# validate that tf can see the gpu
# time python -c "import tensorflow as tf; tf.test.is_gpu_available()"

# run the first example
# the first run takes ~2 mins, next run then takes seconds
# https://github.com/tensorflow/tensorflow/issues/18652 might be related to this
python ./01-intro-to-tensorflow/main.py
```

## conda for dummies

```sh
conda list

conda search tensorflow-gpu
conda install tensorflow-gpu=2.2.0
conda uninstall python
conda install python=3.7

conda deactivate
conda env remove -n tf
```


## monitor gpu usage
```sh
watch nvidia-smi
```


## model conversion

```
pip install tensorflowjs

tensorflowjs_converter --input_format=keras ./humans-and-horses.h5 ./model
```
