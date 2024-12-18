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
    # load model
    config = {
        "is_causal": True,
        "in_channels": 3,
        "out_channels": 3,
        "ch": 128,
        "ch_mult": [1, 2, 4, 4],
        "z_channels": 16,
        "double_z": True,
        "num_res_blocks": 2, 
        "temporal_compression_ratio": 4,
        "regularizer": 'kl',  # kl
        "codebook_size": 262144,
    }
    model = AutoencoderVidTok(**config).cuda()

    model.enable_tiling()
    model.enable_slicing()
    
    ckpt_path = '/home/v-tanganni/workspace/code/tmp/checkpoints/generative_codec/logs_1102/VidTok_new/kl_causal_488_16chn.ckpt'
    sd = from_pretrained(ckpt_path)
    missing, unexpected = model.load_state_dict(sd, strict=False)

    # load data
    sample_num = 18
    # sample_num = 16
    datadir = "/home/v-tanganni/workspace/code/tmp/test_results/tmp/rename/input/video3"
    img_lst = [cv2.imread(f'{datadir}/{i+1}.png')[:,:,::-1] for i in range(sample_num)]
    imgs = np.stack(img_lst, axis=0)
    imgs = torch.from_numpy(imgs).permute([3,0,1,2]).unsqueeze(0).float() / 255.  # (1,3,t,256,256), 0~1
    x = (imgs * 2 - 1).cuda()  # (1,3,t,256,256), -1~1

    # model forward
    with torch.no_grad():
        out = model(x, sample_posterior=True).sample
    assert x.shape == out.shape

    # save image
    out = out.clamp(-1, 1)
    x, out = map(lambda x: (x + 1) / 2, (x, out))
    
    for idx in tqdm(range(out.shape[2])):
        save_image(x[0,:,idx,:,:], f'tmp/input/{idx}.png')
        save_image(out[0,:,idx,:,:], f'tmp/output/{idx}.png')


