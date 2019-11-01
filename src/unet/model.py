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

     Returns:
         out (torch.Tensor) : Prediction of the segmentation map

     """
    def __init__(self, n_channels=1, base_filter_num=64, num_blocks=4, num_classes=5, use_bn=True):

        super(UNet, self).__init__()
        self.use_bn = use_bn
        self.contracting_path = nn.ModuleList()
        self.expanding_path = nn.ModuleList()

        self.num_blocks = num_blocks
        self.n_channels = int(n_channels)
        self.n_classes = int(num_classes)
        self.base_filter_num = int(base_filter_num)
        self.enc_layer_depths = []  # Keep track of the output depths of each encoder block

        for block_id in range(num_blocks):
            enc_block_filter_num = int(pow(2, block_id)*self.base_filter_num)  # Output depth of current encoder stage
            if block_id == 0:
                enc_in_channels = self.n_channels
            else:
                enc_in_channels = enc_block_filter_num//2
            self.enc_layer_depths.append(enc_block_filter_num)
            self.contracting_path.append(EncoderBlock(in_channels=enc_in_channels,
                                                      filter_num=enc_block_filter_num,
                                                      use_bn=self.use_bn))

        # Bottleneck layer
        bottle_neck_filter_num = self.enc_layer_depths[-1]*2
        bottle_neck_in_channels = self.enc_layer_depths[-1]
        self.bottle_neck_layer = EncoderBlock(filter_num=bottle_neck_filter_num,
                                              in_channels=bottle_neck_in_channels,
                                              use_bn=self.use_bn)

        # Decoder Path
        for block_id in range(num_blocks):
            dec_in_channels = int(bottle_neck_filter_num//pow(2, block_id))
            self.expanding_path.append(DecoderBlock(in_channels=dec_in_channels,
                                                    filter_num=self.enc_layer_depths[-1-block_id],
                                                    concat_layer_depth=self.enc_layer_depths[-1-block_id],
                                                    interpolate=True,
                                                    use_bn=self.use_bn))

        # Output Layer
        self.output = nn.Conv2d(in_channels=int(self.enc_layer_depths[0]),
                                out_channels=self.n_classes,
                                kernel_size=1)

    def forward(self, x):

        h, w = x.shape[-2:]

        # Encoder
        enc_outputs = []
        for enc_op in self.contracting_path:
            x = enc_op(x)
            enc_outputs.append(x)
            x = nn.MaxPool2d(kernel_size=2)(x)

        # Bottle-neck layer
        x = self.bottle_neck_layer(x)

        # Decoder
        for block_id, dec_op in enumerate(self.expanding_path):
            x = dec_op(x, enc_outputs[-1-block_id])

        # Output
        x = self.output(x)

        # Interpolate to match the size of seg-map
        out = F.interpolate(input=x,
                            size=(h, w),
                            mode='bilinear',
                            align_corners=True)

        out = F.relu(out)

        return out






