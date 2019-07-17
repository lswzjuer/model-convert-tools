
# lidar net 
from .network import Network
import tensorflow as tf
from config import cfg
import numpy as np
import cv2

class LIDAR(Network):

    def __init__(self):
        pass
    def _layer_map(self):
        pass

    def setup(self):
        '''calculate the bit_rs '''
        (self.feed('data')
             .conv(3, 3, 32, 2, 2, name='resnetv1pyr1_hybridsequential0_conv0_fwd',fl=7,rs=7) 
             .conv(3, 3, 32, 1, 1, name='resnetv1pyr1_hybridsequential1_conv0_fwd',fl=2,rs=7) 
             .conv(3, 3, 64, 1, 1, name='resnetv1pyr1_hybridsequential2_conv0_fwd',fl=3,rs=8)
             .max_pool(3, 3, 2, 2, name='pool0_fwd')
             .conv(1, 1, 32, 1, 1, name='resnetv1pyr1_stage_4bottleneckv10_hybridsequential0_conv0_fwd',fl=3,rs=8)
             .conv(3, 3, 32, 1, 1, name='resnetv1pyr1_stage_4bottleneckv10_hybridsequential1_conv0_fwd',fl=3,rs=9)
             .conv(1, 1, 256, 1, 1, relu=False, name='resnetv1pyr1_stage_4bottleneckv10_hybridsequential2_conv0_fwd',fl=3,rs=7))

        (self.feed('pool0_fwd')
             .conv(1, 1, 256, 1, 1, relu=False, name='resnetv1pyr1_stage_4_shortcuthybridsequential0_conv0_fwd',fl=3,rs=8))

        (self.feed('resnetv1pyr1_stage_4bottleneckv10_hybridsequential2_conv0_fwd', 
                   'resnetv1pyr1_stage_4_shortcuthybridsequential0_conv0_fwd')
             .add(name='resnetv1pyr1_stage_4bottleneckv10__plus0')
             .relu(name='resnetv1pyr1_stage_4bottleneckv10_relu2_fwd')
             .conv(1, 1, 32, 1, 1, name='resnetv1pyr1_stage_4bottleneckv11_hybridsequential0_conv0_fwd',fl=3,rs=9)
             .conv(3, 3, 32, 1, 1, name='resnetv1pyr1_stage_4bottleneckv11_hybridsequential1_conv0_fwd',fl=3,rs=9)
             .conv(1, 1, 256, 1, 1, relu=False, name='resnetv1pyr1_stage_4bottleneckv11_hybridsequential2_conv0_fwd',fl=3,rs=7))

        (self.feed('resnetv1pyr1_stage_4bottleneckv11_hybridsequential2_conv0_fwd', 
                   'resnetv1pyr1_stage_4bottleneckv10_relu2_fwd')
             .add(name='resnetv1pyr1_stage_4bottleneckv11__plus0')
             .relu(name='resnetv1pyr1_stage_4bottleneckv11_relu2_fwd')
             .conv(1, 1, 32, 1, 1, name='resnetv1pyr1_stage_4bottleneckv12_hybridsequential0_conv0_fwd',fl=3,rs=9)
             .conv(3, 3, 32, 1, 1, name='resnetv1pyr1_stage_4bottleneckv12_hybridsequential1_conv0_fwd',fl=3,rs=9)
             .conv(1, 1, 256, 1, 1, relu=False, name='resnetv1pyr1_stage_4bottleneckv12_hybridsequential2_conv0_fwd',fl=3,rs=7))

        (self.feed('resnetv1pyr1_stage_4bottleneckv12_hybridsequential2_conv0_fwd', 
                   'resnetv1pyr1_stage_4bottleneckv11_relu2_fwd')
             .add(name='resnetv1pyr1_stage_4bottleneckv12__plus0')
             .relu(name='resnetv1pyr1_stage_4bottleneckv12_relu2_fwd')
             .conv(1, 1, 128, 2, 2, name='resnetv1pyr1_stage_8bottleneckv10_hybridsequential0_conv0_fwd',fl=2,rs=8)
             .conv(3, 3, 128, 1, 1, name='resnetv1pyr1_stage_8bottleneckv10_hybridsequential1_conv0_fwd',fl=3,rs=10)
             .conv(1, 1, 512, 1, 1, relu=False, name='resnetv1pyr1_stage_8bottleneckv10_hybridsequential2_conv0_fwd',fl=3,rs=8))

        (self.feed('resnetv1pyr1_stage_4bottleneckv12_relu2_fwd')
             .avg_pool(2, 2, 2, 2, name='resnetv1pyr1_stage_8_shortcutpool0_fwd')
             .conv(1, 1, 512, 1, 1, relu=False, name='resnetv1pyr1_stage_8_shortcuthybridsequential0_conv0_fwd',fl=2,rs=7))

        (self.feed('resnetv1pyr1_stage_8bottleneckv10_hybridsequential2_conv0_fwd', 
                   'resnetv1pyr1_stage_8_shortcuthybridsequential0_conv0_fwd')
             .add(name='resnetv1pyr1_stage_8bottleneckv10__plus0')
             .relu(name='resnetv1pyr1_stage_8bottleneckv10_relu2_fwd')
             .conv(1, 1, 64, 1, 1, name='resnetv1pyr1_stage_8bottleneckv11_hybridsequential0_conv0_fwd',fl=3,rs=10)
             .conv(3, 3, 64, 1, 1, name='resnetv1pyr1_stage_8bottleneckv11_hybridsequential1_conv0_fwd',fl=3,rs=9)
             .conv(1, 1, 512, 1, 1, relu=False, name='resnetv1pyr1_stage_8bottleneckv11_hybridsequential2_conv0_fwd',fl=3,rs=7))

        (self.feed('resnetv1pyr1_stage_8bottleneckv11_hybridsequential2_conv0_fwd', 
                   'resnetv1pyr1_stage_8bottleneckv10_relu2_fwd')
             .add(name='resnetv1pyr1_stage_8bottleneckv11__plus0')
             .relu(name='resnetv1pyr1_stage_8bottleneckv11_relu2_fwd')
             .conv(1, 1, 64, 1, 1, name='resnetv1pyr1_stage_8bottleneckv12_hybridsequential0_conv0_fwd',fl=3,rs=10)
             .conv(3, 3, 64, 1, 1, name='resnetv1pyr1_stage_8bottleneckv12_hybridsequential1_conv0_fwd',fl=3,rs=9)
             .conv(1, 1, 512, 1, 1, relu=False, name='resnetv1pyr1_stage_8bottleneckv12_hybridsequential2_conv0_fwd',fl=3,rs=7))

        (self.feed('resnetv1pyr1_stage_8bottleneckv12_hybridsequential2_conv0_fwd', 
                   'resnetv1pyr1_stage_8bottleneckv11_relu2_fwd')
             .add(name='resnetv1pyr1_stage_8bottleneckv12__plus0')
             .relu(name='resnetv1pyr1_stage_8bottleneckv12_relu2_fwd')
             .conv(1, 1, 64, 1, 1, name='resnetv1pyr1_stage_8bottleneckv13_hybridsequential0_conv0_fwd',fl=3,rs=10)
             .conv(3, 3, 64, 1, 1, name='resnetv1pyr1_stage_8bottleneckv13_hybridsequential1_conv0_fwd',fl=3,rs=9)
             .conv(1, 1, 512, 1, 1, relu=False, name='resnetv1pyr1_stage_8bottleneckv13_hybridsequential2_conv0_fwd',fl=3,rs=7))

        (self.feed('resnetv1pyr1_stage_8bottleneckv13_hybridsequential2_conv0_fwd', 
                   'resnetv1pyr1_stage_8bottleneckv12_relu2_fwd')
             .add(name='resnetv1pyr1_stage_8bottleneckv13__plus0')
             .relu(name='resnetv1pyr1_stage_8bottleneckv13_relu2_fwd')
             .conv(1, 1, 128, 1, 1, relu=False, name='resnetv1pyr1_hybridsequential6_conv0_fwd',fl=2,rs=9))

        (self.feed('resnetv1pyr1_stage_8bottleneckv13_relu2_fwd')
             .conv(1, 1, 256, 2, 2, name='resnetv1pyr1_stage_16bottleneckv10_hybridsequential0_conv0_fwd',fl=2,rs=9)
             .conv(3, 3, 256, 1, 1, name='resnetv1pyr1_stage_16bottleneckv10_hybridsequential1_conv0_fwd',fl=3,rs=10)
             .conv(1, 1, 1024, 1, 1, relu=False, name='resnetv1pyr1_stage_16bottleneckv10_hybridsequential2_conv0_fwd',fl=4,rs=9))

        (self.feed('resnetv1pyr1_stage_8bottleneckv13_relu2_fwd')
             .avg_pool(2, 2, 2, 2, name='resnetv1pyr1_stage_16_shortcutpool0_fwd')
             .conv(1, 1, 1024, 1, 1, relu=False, name='resnetv1pyr1_stage_16_shortcuthybridsequential0_conv0_fwd',fl=2,rs=7))

        (self.feed('resnetv1pyr1_stage_16bottleneckv10_hybridsequential2_conv0_fwd', 
                   'resnetv1pyr1_stage_16_shortcuthybridsequential0_conv0_fwd')
             .add(name='resnetv1pyr1_stage_16bottleneckv10__plus0')
             .relu(name='resnetv1pyr1_stage_16bottleneckv10_relu2_fwd')
             .conv(1, 1, 128, 1, 1, name='resnetv1pyr1_stage_16bottleneckv11_hybridsequential0_conv0_fwd',fl=3,rs=10)
             .conv(3, 3, 128, 1, 1, name='resnetv1pyr1_stage_16bottleneckv11_hybridsequential1_conv0_fwd',fl=3,rs=9)
             .conv(1, 1, 1024, 1, 1, relu=False, name='resnetv1pyr1_stage_16bottleneckv11_hybridsequential2_conv0_fwd',fl=4,rs=9))

        (self.feed('resnetv1pyr1_stage_16bottleneckv11_hybridsequential2_conv0_fwd', 
                   'resnetv1pyr1_stage_16bottleneckv10_relu2_fwd')
             .add(name='resnetv1pyr1_stage_16bottleneckv11__plus0')
             .relu(name='resnetv1pyr1_stage_16bottleneckv11_relu2_fwd')
             .conv(1, 1, 128, 1, 1, name='resnetv1pyr1_stage_16bottleneckv12_hybridsequential0_conv0_fwd',fl=3,rs=10)
             .conv(3, 3, 128, 1, 1, name='resnetv1pyr1_stage_16bottleneckv12_hybridsequential1_conv0_fwd',fl=3,rs=10)
             .conv(1, 1, 1024, 1, 1, relu=False, name='resnetv1pyr1_stage_16bottleneckv12_hybridsequential2_conv0_fwd',fl=3,rs=8))

        (self.feed('resnetv1pyr1_stage_16bottleneckv12_hybridsequential2_conv0_fwd', 
                   'resnetv1pyr1_stage_16bottleneckv11_relu2_fwd')
             .add(name='resnetv1pyr1_stage_16bottleneckv12__plus0')
             .relu(name='resnetv1pyr1_stage_16bottleneckv12_relu2_fwd')
             .conv(1, 1, 256, 1, 1, relu=False, name='resnetv1pyr1_hybridsequential4_conv0_fwd',fl=3,rs=10))

        (self.feed('resnetv1pyr1_stage_16bottleneckv12_relu2_fwd')
             .conv(1, 1, 128, 1, 1, name='resnetv1pyr1_stage_16_dbottleneckv10_hybridsequential0_conv0_fwd',fl=3,rs=10)
             .conv(3, 3, 128, 1, 1, name='resnetv1pyr1_stage_16_dbottleneckv10_hybridsequential1_conv0_fwd',fl=3,rs=8)
             .conv(1, 1, 1024, 1, 1, relu=False, name='resnetv1pyr1_stage_16_dbottleneckv10_hybridsequential2_conv0_fwd',fl=4,rs=8))

        (self.feed('resnetv1pyr1_stage_16bottleneckv12_relu2_fwd')
             .conv(1, 1, 1024, 1, 1, relu=False, name='resnetv1pyr1_stage_16_d_shortcuthybridsequential0_conv0_fwd',fl=3,rs=9))

        (self.feed('resnetv1pyr1_stage_16_dbottleneckv10_hybridsequential2_conv0_fwd', 
                   'resnetv1pyr1_stage_16_d_shortcuthybridsequential0_conv0_fwd')
             .add(name='resnetv1pyr1_stage_16_dbottleneckv10__plus0')
             .relu(name='resnetv1pyr1_stage_16_dbottleneckv10_relu2_fwd')
             .conv(1, 1, 128, 1, 1, name='resnetv1pyr1_stage_16_dbottleneckv11_hybridsequential0_conv0_fwd',fl=4,rs=11)
             .conv(3, 3, 128, 1, 1, name='resnetv1pyr1_stage_16_dbottleneckv11_hybridsequential1_conv0_fwd',fl=3,rs=8)
             .conv(1, 1, 1024, 1, 1, relu=False, name='resnetv1pyr1_stage_16_dbottleneckv11_hybridsequential2_conv0_fwd',fl=4,rs=8))

        (self.feed('resnetv1pyr1_stage_16_dbottleneckv11_hybridsequential2_conv0_fwd', 
                   'resnetv1pyr1_stage_16_dbottleneckv10_relu2_fwd')
             .add(name='resnetv1pyr1_stage_16_dbottleneckv11__plus0')
             .relu(name='resnetv1pyr1_stage_16_dbottleneckv11_relu2_fwd')
             .conv(1, 1, 128, 1, 1, name='resnetv1pyr1_stage_16_dbottleneckv12_hybridsequential0_conv0_fwd',fl=4,rs=11)
             .conv(3, 3, 128, 1, 1,  name='resnetv1pyr1_stage_16_dbottleneckv12_hybridsequential1_conv0_fwd',fl=3,rs=9)
             .conv(1, 1, 1024, 1, 1, relu=False, name='resnetv1pyr1_stage_16_dbottleneckv12_hybridsequential2_conv0_fwd',fl=3,rs=7))

        (self.feed('resnetv1pyr1_stage_16_dbottleneckv12_hybridsequential2_conv0_fwd', 
                   'resnetv1pyr1_stage_16_dbottleneckv11_relu2_fwd')
             .add(name='resnetv1pyr1_stage_16_dbottleneckv12__plus0')
             .relu(name='resnetv1pyr1_stage_16_dbottleneckv12_relu2_fwd')
             .conv(1, 1, 256, 1, 1, relu=False, name='resnetv1pyr1_hybridsequential3_conv0_fwd',fl=3,rs=9))

        (self.feed('resnetv1pyr1_hybridsequential4_conv0_fwd', 
                   'resnetv1pyr1_hybridsequential3_conv0_fwd')
             .add(name='resnetv1pyr1__plus0')
             .conv(3, 3, 128, 1, 1, relu=False, name='resnetv1pyr1_hybridsequential5_conv0_fwd',fl=3,rs=12)
             .deconv(2, 2, 128, 2, 2, relu=False, name='resnetv1pyr1_upsampling0',fl=10,rs=10))         #TODO X=3,Y=3

        (self.feed('resnetv1pyr1_hybridsequential6_conv0_fwd', 
                   'resnetv1pyr1_upsampling0')
             .add(name='resnetv1pyr1__plus1')
             .conv(3, 3, 64, 1, 1, relu=False, name='resnetv1pyr1_hybridsequential7_conv0_fwd',fl=2,rs=12)
             .deconv(2, 2, 64, 2, 2, relu=False, name='resnetv1pyr1_upsampling1',fl=10,rs=10))         #TODO X=3,Y=3

        (self.feed('resnetv1pyr1_stage_4bottleneckv12_relu2_fwd')
             .conv(1, 1, 64, 1, 1, relu=False, name='resnetv1pyr1_hybridsequential8_conv0_fwd',fl=2,rs=9))

        (self.feed('resnetv1pyr1_upsampling1', 
                   'resnetv1pyr1_hybridsequential8_conv0_fwd')
             .add(name='resnetv1pyr1__plus2')
             .conv(3, 3, 64, 1, 1, relu=False, name='resnetv1pyr1_hybridsequential9_conv0_fwd',fl=3,rs=11)
             .conv(3, 3, 64, 1, 1, name='resnetv1pyr1_share_head_4hybridsequential0_conv0_fwd',fl=3,rs=9)
             .conv(3, 3, 64, 1, 1, name='resnetv1pyr1_cls_head_4hybridsequential0_conv0_fwd',fl=2,rs=8)
             .conv(3, 3, 64, 1, 1, name='resnetv1pyr1_cls_head_4hybridsequential1_conv0_fwd',fl=2,rs=8)
             .conv(1, 1, 4, 1, 1, relu=False, name='resnetv1pyr1_cls_head_4hybridsequential2_conv0_fwd',fl=2,rs=8))

        (self.feed('resnetv1pyr1_hybridsequential7_conv0_fwd')
             .conv(3, 3, 128, 1, 1, name='resnetv1pyr1_share_head_8hybridsequential0_conv0_fwd',fl=3,rs=8)
             .conv(3, 3, 128, 1, 1, name='resnetv1pyr1_cls_head_8hybridsequential0_conv0_fwd',fl=3,rs=8)
             .conv(3, 3, 64, 1, 1, name='resnetv1pyr1_cls_head_8hybridsequential1_conv0_fwd',fl=3,rs=8)
             .conv(1, 1, 4, 1, 1, relu=False, name='resnetv1pyr1_cls_head_8hybridsequential2_conv0_fwd',fl=3,rs=9))

        (self.feed('resnetv1pyr1_hybridsequential5_conv0_fwd')
             .conv(3, 3, 256, 1, 1, name='resnetv1pyr1_share_head_16hybridsequential0_conv0_fwd',fl=3,rs=11)
             .conv(3, 3, 128, 1, 1, name='resnetv1pyr1_cls_head_16hybridsequential0_conv0_fwd',fl=1,rs=9)
             .conv(3, 3, 64, 1, 1, name='resnetv1pyr1_cls_head_16hybridsequential1_conv0_fwd',fl=1,rs=9)
             .conv(1, 1, 4, 1, 1, relu=False, name='resnetv1pyr1_cls_head_16hybridsequential2_conv0_fwd',fl=1,rs=7))

        (self.feed('resnetv1pyr1_share_head_4hybridsequential0_conv0_fwd')
             .conv(3, 3, 64, 1, 1, name='resnetv1pyr1_reg_head_4hybridsequential0_conv0_fwd',fl=2,rs=7)
             .conv(3, 3, 64, 1, 1, name='resnetv1pyr1_reg_head_4hybridsequential1_conv0_fwd',fl=3,rs=6)
             .conv(1, 1, 28, 1, 1, relu=False, name='resnetv1pyr1_reg_head_4hybridsequential2_conv0_fwd',fl=5,rs=6))

        (self.feed('resnetv1pyr1_share_head_8hybridsequential0_conv0_fwd')
             .conv(3, 3, 128, 1, 1, name='resnetv1pyr1_reg_head_8hybridsequential0_conv0_fwd',fl=3,rs=8)
             .conv(3, 3, 128, 1, 1, name='resnetv1pyr1_reg_head_8hybridsequential1_conv0_fwd',fl=4,rs=9)
             .conv(1, 1, 28, 1, 1, relu=False, name='resnetv1pyr1_reg_head_8hybridsequential2_conv0_fwd',fl=4,rs=6))

        (self.feed('resnetv1pyr1_share_head_16hybridsequential0_conv0_fwd')
             .conv(3, 3, 128, 1, 1, name='resnetv1pyr1_reg_head_16hybridsequential0_conv0_fwd',fl=1,rs=10)
             .conv(3, 3, 128, 1, 1, name='resnetv1pyr1_reg_head_16hybridsequential1_conv0_fwd',fl=1,rs=10)
             .conv(1, 1, 28, 1, 1, relu=False, name='resnetv1pyr1_reg_head_16hybridsequential2_conv0_fwd',fl=3,rs=8))