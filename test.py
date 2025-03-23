import os
import cv2
import torch
import numpy as np
from torchvision.utils import save_image
from tqdm import tqdm

from src.diffusers import AutoencoderVidTok


def from_pretrained(ckpt_path):
    sd = torch.load(ckpt_path, map_location="cpu")["state_dict"]
    return sd


if __name__ == "__main__":
    # load data
    sample_num = 118
    # height = 256
    # width = 256
    # height = 128
    # width = 160
    height = 1280
    width = 1600
    
    # load model
    config = {
        "is_causal": False,
        "in_channels": 3,
        "out_channels": 3,
        "ch": 128,
        "ch_mult": [1, 2, 4, 4, 4],
        "z_channels": 16,
        "double_z": True,
        "num_res_blocks": 2, 
        "temporal_compression_ratio": 4,
        "regularizer": 'kl',  # kl
        "codebook_size": 262144,
    }
    model = AutoencoderVidTok(**config).cuda()
    
    model.enable_slicing()
    model.enable_tiling()
    
    ckpt_path = '/hdd/dataset_anni/code/msra/checkpoints/vidtok_kl_noncausal_41616_16chn.ckpt'
    sd = from_pretrained(ckpt_path)
    missing, unexpected = model.load_state_dict(sd, strict=False)

    # load data
    datadir = "tmp/imgs"
    img_lst = [cv2.resize(cv2.imread(f'{datadir}/{i+1}.png')[:,:,::-1], (width, height)) for i in range(sample_num)]
    imgs = np.stack(img_lst, axis=0)
    imgs = torch.from_numpy(imgs).permute([3,0,1,2]).unsqueeze(0).float() / 255.  # (1,3,t,256,256), 0~1
    x = (imgs * 2 - 1).cuda()  # (1,3,t,256,256), -1~1

    # model forward
    with torch.no_grad():
        out = model(x, sample_posterior=True).sample
    assert x.shape == out.shape, f'x: {x.shape}, out: {out.shape}'

    # save image
    out = out.clamp(-1, 1)
    x, out = map(lambda x: (x + 1) / 2, (x, out))
    
    # save_fd = "output"
    save_fd = "output_tile"
    os.system(f'rm -rf tmp/input; rm -rf tmp/{save_fd}')
    os.makedirs('tmp/input', exist_ok=True)
    os.makedirs(f'tmp/{save_fd}', exist_ok=True)
    
    for idx in tqdm(range(out.shape[2])):
        save_image(x[0,:,idx,:,:], f'tmp/input/{idx}.png')
        save_image(out[0,:,idx,:,:], f'tmp/{save_fd}/{idx}.png')


