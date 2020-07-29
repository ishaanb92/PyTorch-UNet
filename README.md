U-Net - PyTorch Implementation
============================== 

Implementation of the [U-Net](http://arxiv.org/abs/1505.04597) model, a popular image segmentation network.
This is quite stable and configurable, I've used it across multiple datasets and as a component in a couple of projects.

* Update: Also supports segmentation of 3-D volumes based on the [3-D UNet architecture](https://arxiv.org/abs/1606.06650)

* Update: All batch normalization operations have been replaced by instance normalization (to account for small batch sizes in medical images) and ReLU activation has been replaced by LeakyReLU due to its greater adoption in recent works. 



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
     PyTorch class definition for the U-Net architecture for image segmentation

     Parameters:
         n_channels (int) : Number of image channels
         base_filter_num (int) : Number of filters for the first convolution (doubled for every subsequent block)
         num_blocks (int) : Number of encoder/decoder blocks
         num_classes(int) : Number of classes that need to be segmented
         mode (str): 2D or 3D input
         use_pooling (bool): Set to 'True' to use MaxPool as downnsampling op.
                             If 'False', strided convolution would be used to downsample feature maps (http://arxiv.org/abs/1908.02182)

     Returns:
         out (torch.Tensor) : Prediction of the segmentation map

```
This package is exclusively tested on Python 3.7.3 and PyTorch 1.1.0 

Note
====

This project has been set up using PyScaffold 3.2.1. For details and usage
information on PyScaffold see https://pyscaffold.org/.
