"""
TabSyn sampling (vendored from amazon-science/tabsyn `tabsyn/sample.py`).

TabQueryBench changes:
- `--num_samples` (upstream always sampled len(train) rows); `--steps` honoured (default 50).
- Sampling + VAE decoding in chunks (TABSYN_SAMPLE_BATCH_SIZE, auto from token count) on the GPU.
"""
import argparse
import os
import time
import warnings

import numpy as np
import torch

from model import MLPDiffusion, Model
from latent_utils import get_input_generate, recover_data, split_num_cat_target
from diffusion_utils import sample

warnings.filterwarnings('ignore')


def _auto_chunk(n_tokens: int) -> int:
    per_row = n_tokens * n_tokens + n_tokens * 128
    return int(max(1024, min(100000, 2e8 // max(per_row, 1))))


def main(args):
    device = args.device
    steps = args.steps or 50
    save_path = args.save_path

    train_z, _, _, ckpt_path, info, num_inverse, cat_inverse = get_input_generate(args)
    in_dim = train_z.shape[1]

    mean = train_z.mean(0).to(device)

    denoise_fn = MLPDiffusion(in_dim, 1024).to(device)
    model = Model(denoise_fn=denoise_fn, hid_dim=in_dim).to(device)
    model.load_state_dict(torch.load(f'{ckpt_path}/model.pt', map_location=device))
    model.eval()

    num_samples = args.num_samples if args.num_samples and args.num_samples > 0 else train_z.shape[0]
    n_tokens = in_dim // info['token_dim']
    raw_chunk = os.environ.get("TABSYN_SAMPLE_BATCH_SIZE", "").strip()
    chunk = int(raw_chunk) if raw_chunk else _auto_chunk(n_tokens)
    print(f"[TabSyn][Sample] num_samples={num_samples} steps={steps} chunk={chunk} device={device}", flush=True)

    start_time = time.time()
    nums, cats, tgts = [], [], []
    for s in range(0, num_samples, chunk):
        b = min(chunk, num_samples - s)
        x_next = sample(model.denoise_fn_D, b, in_dim, num_steps=steps, device=device)
        x_next = x_next * 2 + mean
        syn_num, syn_cat, syn_target = split_num_cat_target(x_next.float(), info, num_inverse, cat_inverse, device)
        nums.append(syn_num)
        cats.append(syn_cat)
        tgts.append(syn_target)

    syn_num = np.concatenate(nums, axis=0)
    syn_cat = np.concatenate(cats, axis=0)
    syn_target = np.concatenate(tgts, axis=0)

    syn_df = recover_data(syn_num, syn_cat, syn_target, info)

    idx_name_mapping = info['idx_name_mapping']
    idx_name_mapping = {int(key): value for key, value in idx_name_mapping.items()}

    syn_df.rename(columns=idx_name_mapping, inplace=True)
    syn_df.to_csv(save_path, index=False)

    print('Time:', time.time() - start_time)
    print('Saving sampled data to {}'.format(save_path))


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Generation')

    parser.add_argument('--dataname', type=str, default='adult', help='Name of dataset.')
    parser.add_argument('--gpu', type=int, default=0, help='GPU index.')
    parser.add_argument('--epoch', type=int, default=None, help='Epoch.')
    parser.add_argument('--steps', type=int, default=None, help='Number of function evaluations.')
    parser.add_argument('--num_samples', type=int, default=None, help='Rows to generate (default: len(train)).')
    parser.add_argument('--save_path', type=str, required=True, help='Output CSV path.')

    args = parser.parse_args()

    if args.gpu != -1 and torch.cuda.is_available():
        args.device = f'cuda:{args.gpu}'
    else:
        args.device = 'cpu'
    main(args)
