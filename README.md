U-Net - PyTorch Implementation
============================== 

Implementation of the [U-Net](http://arxiv.org/abs/1505.04597) model, a popular image segmentation network.
This is quite stable and configurable, I've used it across multiple datasets and as a component in a couple of projects.
Update: Also supports segmentation of 3-D volumes based on the [3-D UNet architecture](https://arxiv.org/abs/1606.06650)



Installation
===========

You can install this package in your local python environment and import it as a module in your project.

* Clone this repository in a folder of your choice.
```
cd <UNET_FOLDER>
git clone https://github.com/kilgore92/PyTorch-UNet.git

```

* Install package dependencies as follows:
```
cd <UNET_FOLDER>
<PATH_TO_PYTHON_ENV>/bin/pip install -r requirements.txt
```


* Install this in your local python environment using the ```setup.py``` script.
```
cd <UNET_FOLDER>
conda activate <ENV_NAME>
python setup.py install
```
or

```
<PATH_TO_PYTHON_ENV>/bin/python setup.py install 
```


Example Usage
=============
```python
from unet.model import UNet
...

model = UNet(n_channels=1,
             num_classes=2,
             use_bn=True,
             mode='2D',
             use_pooling=False
             )
```

For more information about various instantiation arguments:
```python
from unet.model import UNet
print(UNet.__doc__)
```
This package is exclusively tested on Python 3.7.3 and PyTorch 1.1.0 

Note
====

This project has been set up using PyScaffold 3.2.1. For details and usage
information on PyScaffold see https://pyscaffold.org/.
