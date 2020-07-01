"""
A PyTorch Implementation of a U-Net
http://arxiv.org/abs/1505.04597

Author: Ishaan Bhat
Email: ishaan@isi.uu.nl

"""
from .blocks import *
from math import pow


class UNet(nn.Module):
    """
     PyTorch class definition for the U-Net architecture for image segmentation

     Parameters:
         image_size (int) : Height or width of a square image (assumes image is square)
         n_channels (int) : Number of image channels (3 for RGB, 1 for grayscale)
         base_filter_num (int) : Number of filters for the first convolution (doubled for every subsequent block)
         num_blocks (int) : Number of encoder/decoder blocks
         num_classes(int) : Number of classes that need to be segmented
         mode (str): 2D or 3D
         dropout (bool) : Whether dropout should be added to central encoder and decoder blocks (eg: BayesianSegNet)
         dropout_rate (float) : Dropout probability

     Returns:
         out (torch.Tensor) : Prediction of the segmentation map

     """
    def __init__(self, n_channels=1, base_filter_num=64, num_blocks=4, num_classes=5, use_bn=True, mode='2D', dropout=False, dropout_rate=0.3):

        super(UNet, self).__init__()
        self.use_bn = use_bn
        self.contracting_path = nn.ModuleList()
        self.expanding_path = nn.ModuleList()

        self.num_blocks = num_blocks
        self.n_channels = int(n_channels)
        self.n_classes = int(num_classes)
        self.base_filter_num = int(base_filter_num)
        self.enc_layer_depths = []  # Keep track of the output depths of each encoder block
        self.mode = mode
        self.dropout = dropout
        self.dropout_rate = dropout_rate

        if mode == '2D':
            self.encoder = EncoderBlock
            self.decoder = DecoderBlock
            self.pool = nn.MaxPool2d
        elif mode == '3D':
            self.encoder = EncoderBlock3D
            self.decoder = DecoderBlock3D
            self.pool = nn.MaxPool3d
        else:
            print('{} mode is invalid'.format(mode))

        for block_id in range(num_blocks):
            enc_block_filter_num = int(pow(2, block_id)*self.base_filter_num)  # Output depth of current encoder stage (for 2D UNet)
            if block_id == 0:
                enc_in_channels = self.n_channels
            else:
                if self.mode == '2D':
                    enc_in_channels = enc_block_filter_num//2
                else:
                    enc_in_channels = enc_block_filter_num  # In the 3D UNet arch, the encoder features double in the 2nd convolution op

            if self.mode == '2D':
                self.enc_layer_depths.append(enc_block_filter_num)
            else:
                self.enc_layer_depths.append(enc_block_filter_num*2)

            # Dropout only applied to central encoder blocks -- See BayesianSegNet by Kendall et al.
            if self.dropout is True and block_id >= num_blocks/2:
                self.contracting_path.append(self.encoder(in_channels=enc_in_channels,
                                                          filter_num=enc_block_filter_num,
                                                          use_bn=self.use_bn,
                                                          dropout=True,
                                                          dropout_rate=self.dropout_rate))
            else:
                self.contracting_path.append(self.encoder(in_channels=enc_in_channels,
                                                          filter_num=enc_block_filter_num,
                                                          use_bn=self.use_bn,
                                                          dropout=False))

        # Bottleneck layer
        if self.mode == '2D':
            bottle_neck_filter_num = self.enc_layer_depths[-1]*2
            bottle_neck_in_channels = self.enc_layer_depths[-1]
            self.bottle_neck_layer = self.encoder(filter_num=bottle_neck_filter_num,
                                                  in_channels=bottle_neck_in_channels,
                                                  use_bn=self.use_bn)
        else:  # Modified for the 3D UNet architecture
            bottle_neck_in_channels = self.enc_layer_depths[-1]
            bottle_neck_filter_num = self.enc_layer_depths[-1]*2
            self.bottle_neck_layer =  nn.Sequential(nn.Conv3d(in_channels=bottle_neck_in_channels,
                                                              out_channels=bottle_neck_in_channels,
                                                              kernel_size=3,
                                                              padding=1),

                                                    nn.BatchNorm3d(num_features=bottle_neck_in_channels),

                                                    nn.ReLU(),

                                                    nn.Conv3d(in_channels=bottle_neck_in_channels,
                                                              out_channels=bottle_neck_filter_num,
                                                              kernel_size=3,
                                                              padding=1),

                                                    nn.BatchNorm3d(num_features=bottle_neck_filter_num),

                                                    nn.ReLU())

        # Decoder Path
        for block_id in range(num_blocks):
            dec_in_channels = int(bottle_neck_filter_num//pow(2, block_id))
            if self.dropout is True and block_id <= num_blocks/2:
                self.expanding_path.append(self.decoder(in_channels=dec_in_channels,
                                                        filter_num=self.enc_layer_depths[-1-block_id],
                                                        concat_layer_depth=self.enc_layer_depths[-1-block_id],
                                                        interpolate=False,
                                                        use_bn=self.use_bn,
                                                        dropout=True,
                                                        dropout_rate=self.dropout_rate))
            else:
                self.expanding_path.append(self.decoder(in_channels=dec_in_channels,
                                                        filter_num=self.enc_layer_depths[-1-block_id],
                                                        concat_layer_depth=self.enc_layer_depths[-1-block_id],
                                                        interpolate=False,
                                                        use_bn=self.use_bn,
                                                        dropout=False))

        # Output Layer
        if mode == '2D':
            self.output = nn.Conv2d(in_channels=int(self.enc_layer_depths[0]),
                                    out_channels=self.n_classes,
                                    kernel_size=1)
        else:
            self.output = nn.Conv3d(in_channels=int(self.enc_layer_depths[0]),
                                    out_channels=self.n_classes,
                                    kernel_size=1)

    def forward(self, x):

        if self.mode == '2D':
            h, w = x.shape[-2:]
        else:
            d, h, w = x.shape[-3:]

        # Encoder
        enc_outputs = []
        for stage, enc_op in enumerate(self.contracting_path):
            x = enc_op(x)
            enc_outputs.append(x)
            x = self.pool(kernel_size=2)(x)

        # Bottle-neck layer
        x = self.bottle_neck_layer(x)

        # Decoder
        for block_id, dec_op in enumerate(self.expanding_path):
            x = dec_op(x, enc_outputs[-1-block_id])

        # Output
        x = self.output(x)

        return x






