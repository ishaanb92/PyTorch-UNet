##U-Net - PyTorch Implementation 

Implementation of the [U-Net](http://arxiv.org/abs/1505.04597) model, a popular image segmentation network.
This is quite stable and configurable, I've used it across multiple datasets and as a component in a couple of projects. 



Installation
===========

You can install this package in your local python environment and import it as a module in your project.

* Clone this repository in a folder of your choice.
```
cd <UNET_FOLDER>
git clone https://github.com/kilgore92/PyTorch-UNet.git

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

* Install package dependencies as follows:
```
cd <UNET_FOLDER>
<PATH_TO_PYTHON_ENV>/bin/pip install -r requirements.txt
```

Example Usage
=============
```python
from unet.model import UNet
...

model = UNet(image_size=256,
             num_classes=2,
             use_bn=True,
             n_channels=6)
    
```

For more information about various instantiation arguments:
```python

print(UNet.__doc__)

```


Note
====

This project has been set up using PyScaffold 3.2.1. For details and usage
information on PyScaffold see https://pyscaffold.org/.
