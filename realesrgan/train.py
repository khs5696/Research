# flake8: noqa
import os.path as osp
from train_pipeline import train_pipeline
# HS ) pip install basicsr

# import realesrgan.archs
# import realesrgan.data
# import realesrgan.models

# wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.1/ESRGAN_SRx4_DF2KOST_official-ff704c30.pth -P experiments/pretrained_models
# python realesrgan/train.py -opt options/train_realesrnet_x4plus.yml --auto_resume
if __name__ == '__main__':
    # root_path = C:\Users\khs20\Desktop\Real-ESRGAN-master
    root_path = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir))
    train_pipeline(root_path)
