import torch
import torch.nn as nn
from model.modules import *
from model.network import *

class MDEDNet(nn.Module):
    def __init__(self):
        super(MDEDNet, self).__init__()
        conv_dim = 32
        event_ch = 13
        blur_ch = 1
        enc_blk_nums=[2,2,2]
        dec_blk_nums=[2,2,2]          
        self.intro_blur = Conv(blur_ch,conv_dim*2**0,ksize=3)

        self.encoder_blur_1 = nn.Sequential(
                    *[ResBlock(conv_dim*2**0,conv_dim*2**0) for _ in range(enc_blk_nums[0])]
                )
        self.down_blur_1_2 = Downsample(conv_dim*2**0) 

        self.encoder_blur_2 = nn.Sequential(
                    *[ResBlock(conv_dim*2**1,conv_dim*2**1) for _ in range(enc_blk_nums[1])]
                )
        self.down_blur_2_3 = Downsample(conv_dim*2**1) 

        self.encoder_blur_3 = nn.Sequential(
                    *[ResBlock(conv_dim*2**2,conv_dim*2**2) for _ in range(enc_blk_nums[2])]
                )   

        self.decoder_blur_3 = nn.Sequential(
                    *[ResBlock(conv_dim*2**2,conv_dim*2**2) for _ in range(dec_blk_nums[2])]
                )   
        self.up_blur_3_2 = Upsample(conv_dim*2**2)

        self.decoder_blur_2 = nn.Sequential(
                    *[ResBlock(conv_dim*2**1,conv_dim*2**1) for _ in range(dec_blk_nums[1])]
                )   
        self.up_blur_2_1 = Upsample(conv_dim*2**1)

        self.decoder_blur_1 = nn.Sequential(
                    *[ResBlock(conv_dim*2**0,conv_dim*2**0) for _ in range(dec_blk_nums[0])]
                )   

        self.output_blur = nn.Conv2d(conv_dim*2**0, blur_ch, kernel_size=3, stride=1, padding=1)


        self.intro_event = Conv(event_ch,conv_dim,ksize=3)

        self.encoder_event_1 = nn.Sequential(
                    *[ResBlock(conv_dim*2**0,conv_dim*2**0) for _ in range(enc_blk_nums[0])]
                )
        self.down_event_1_2 = Downsample(conv_dim*2**0) 

        self.encoder_event_2 = nn.Sequential(
                    *[ResBlock(conv_dim*2**1,conv_dim*2**1) for _ in range(enc_blk_nums[1])]
                )
        self.down_event_2_3 = Downsample(conv_dim*2**1) 

        self.encoder_event_3 = nn.Sequential(
                    *[ResBlock(conv_dim*2**2,conv_dim*2**2) for _ in range(enc_blk_nums[2])]
                )   

        self.decoder_event_3 = nn.Sequential(
                    *[ResBlock(conv_dim*2**2,conv_dim*2**2) for _ in range(dec_blk_nums[2])]
                )   
        self.up_event_3_2 = Upsample(conv_dim*2**2)

        self.decoder_event_2 = nn.Sequential(
                    *[ResBlock(conv_dim*2**1,conv_dim*2**1) for _ in range(dec_blk_nums[1])]
                )   
        self.up_event_2_1 = Upsample(conv_dim*2**1)

        self.decoder_event_1 = nn.Sequential(
                    *[ResBlock(conv_dim*2**0,conv_dim*2**0) for _ in range(dec_blk_nums[0])]
                )  
        self.output_event = nn.Conv2d(conv_dim*2**0, event_ch, kernel_size=3, stride=1, padding=1) 

        self.sce1 = Spectral_Consistency_Enhancemnent(conv_dim*2**0)
        self.sce2 = Spectral_Consistency_Enhancemnent(conv_dim*2**1)
        self.sce3 = Spectral_Consistency_Enhancemnent(conv_dim*2**2) 
        self.cmi1 = Cross_Modal_Multi_Order_Interaction(conv_dim*2**0,1) 
        self.cmi2 = Cross_Modal_Multi_Order_Interaction(conv_dim*2**1,2) 
        self.cmi3 = Cross_Modal_Multi_Order_Interaction(conv_dim*2**2,4)
    def forward(self,blur,event):
        f_blur = self.intro_blur(blur)
        f_event = self.intro_event(event)
        
        fe_blur_1 = self.encoder_blur_1(f_blur)
        fe_blur_1_2 = self.down_blur_1_2(fe_blur_1)
        fe_blur_2 = self.encoder_blur_2(fe_blur_1_2)
        fe_blur_2_3 = self.down_blur_2_3(fe_blur_2)
        fe_blur_3 = self.encoder_blur_3(fe_blur_2_3)

        fe_event_1 = self.encoder_event_1(f_event)
        fe_event_1_2 = self.down_event_1_2(fe_event_1)
        fe_event_2 = self.encoder_event_2(fe_event_1_2)
        fe_event_2_3 = self.down_event_2_3(fe_event_2)
        fe_event_3 = self.encoder_event_3(fe_event_2_3)

        fe_blur_3_c, fe_event_3_c = self.sce3(fe_blur_3,fe_event_3)
        fe_event_res3,fe_blur_res3 = self.cmi3(fe_blur_3_c, fe_event_3_c)  
        fe_blur_3_en = fe_blur_3+fe_event_res3
        fe_event_3_en = fe_event_3+fe_blur_res3

        fe_blur_2_c, fe_event_2_c = self.sce2(fe_blur_2,fe_event_2)          
        fe_event_res2,fe_blur_res2 = self.cmi2(fe_blur_2_c, fe_event_2_c) 
        fe_blur_2_en = fe_blur_2+fe_event_res2
        fe_event_2_en = fe_event_2+fe_blur_res2

        fe_blur_1_c, fe_event_1_c = self.sce1(fe_blur_1,fe_event_1)          
        fe_event_res1,fe_blur_res1 = self.cmi1(fe_blur_1_c, fe_event_1_c)
        fe_blur_1_en = fe_blur_1+fe_event_res1
        fe_event_1_en = fe_event_1+fe_blur_res1
                                                                                          
        fd_blur_3 = self.decoder_blur_3(fe_blur_3_en)
        fd_blur_3_2 = self.up_blur_3_2(fd_blur_3)

        fd_blur_2 = self.decoder_blur_2(fd_blur_3_2+fe_blur_2_en)
        fd_blur_2_1 = self.up_blur_2_1(fd_blur_2)
        
        fd_blur_1 = self.decoder_blur_1(fd_blur_2_1+fe_blur_1_en)

        fd_event_3 = self.decoder_event_3(fe_event_3_en)
        fd_event_3_2 = self.up_event_3_2(fd_event_3)

        fd_event_2 = self.decoder_event_2(fd_event_3_2+fe_event_2_en)
        fd_event_2_1 = self.up_event_2_1(fd_event_2)

        fd_event_1 = self.decoder_event_1(fd_event_2_1+fe_event_1_en)

        deblur = self.output_blur(fd_blur_1)+blur
        deblur = torch.clamp(deblur, min=0, max=1)

        denoise = self.output_event(fd_event_1)+event
        return deblur, denoise



