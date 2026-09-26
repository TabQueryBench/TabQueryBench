"""
TabSyn stage 1: VAE training (vendored from amazon-science/tabsyn `tabsyn/vae/main.py`).

TabQueryBench changes (model math unchanged):
- Env-driven epochs / batch size (TABSYN_VAE_EPOCHS, TABSYN_VAE_BATCH_SIZE); batch auto-shrinks for wide
  tables because attention memory grows with n_tokens^2.
- Tensor-index batching instead of a per-item DataLoader (much faster; same sampling semantics).
- Chunked validation / final encoding to bound GPU memory.
- Checkpoints go to $TABSYN_CKPT_BASE/vae/<dataname>/ when set.
"""
import argparse
import json
import os
import time
import warnings

import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau

from vae.model import Model_VAE, Encoder_model, Decoder_model
from utils_train import preprocess

warnings.filterwarnings('ignore')


LR = 1e-3
WD = 0
D_TOKEN = 4
TOKEN_BIAS = True

N_HEAD = 1
FACTOR = 32
NUM_LAYERS = 2

UPSTREAM_BATCH_SIZE = 4096
UPSTREAM_EPOCHS = 4000


def _env_int(name, default):
    raw = os.environ.get(name, "").strip()
    return int(raw) if raw else default


def _auto_batch_size(n_tokens: int) -> int:
    """Upstream 4096; shrink ~1/T^2 beyond 64 tokens so attention activations stay roughly constant."""
    if n_tokens <= 64:
        return UPSTREAM_BATCH_SIZE
    return max(256, int(UPSTREAM_BATCH_SIZE * (64.0 / n_tokens) ** 2))


def _place(t: torch.Tensor, device: str) -> torch.Tensor:
    budget = float(os.environ.get("TABSYN_GPU_DATA_MAX_GB", "4")) * (1024 ** 3)
    if str(device).startswith("cuda") and t.element_size() * t.nelement() <= budget:
        return t.to(device)
    return t


def compute_loss(X_num, X_cat, Recon_X_num, Recon_X_cat, mu_z, logvar_z):
    ce_loss_fn = nn.CrossEntropyLoss()
    mse_loss = (X_num - Recon_X_num).pow(2).mean()
    ce_loss = 0
    acc = 0
    total_num = 0
    n_cat_heads = 0

    for idx, x_cat in enumerate(Recon_X_cat):
        if x_cat is not None:
            ce_loss += ce_loss_fn(x_cat, X_cat[:, idx])
            x_hat = x_cat.argmax(dim=-1)
            acc += (x_hat == X_cat[:, idx]).float().sum()
            total_num += x_hat.shape[0]
            n_cat_heads += 1

    if n_cat_heads > 0:
        ce_loss /= n_cat_heads
        acc /= total_num
    else:
        ce_loss = torch.tensor(0.0, device=X_num.device)
        acc = torch.tensor(1.0, device=X_num.device)

    temp = 1 + logvar_z - mu_z.pow(2) - logvar_z.exp()

    loss_kld = -0.5 * torch.mean(temp.mean(-1).mean())
    return mse_loss, ce_loss, loss_kld, acc


def main(args):
    dataname = args.dataname
    data_dir = f'data/{dataname}'

    max_beta = args.max_beta
    min_beta = args.min_beta
    lambd = args.lambd

    device = args.device

    with open(f'data/{dataname}/info.json', 'r') as f:
        info = json.load(f)

    curr_dir = os.path.dirname(os.path.abspath(__file__))
    ckpt_root = os.environ.get('TABSYN_CKPT_BASE')
    ckpt_dir = f'{ckpt_root}/vae/{dataname}' if ckpt_root else f'{curr_dir}/ckpt/{dataname}'
    os.makedirs(ckpt_dir, exist_ok=True)

    model_save_path = f'{ckpt_dir}/model.pt'
    encoder_save_path = f'{ckpt_dir}/encoder.pt'
    decoder_save_path = f'{ckpt_dir}/decoder.pt'

    X_num, X_cat, categories, d_numerical = preprocess(data_dir, task_type=info['task_type'])

    X_train_num, X_test_num = X_num
    X_train_cat, X_test_cat = X_cat

    X_train_num = torch.tensor(X_train_num, dtype=torch.float32)
    X_test_num = torch.tensor(X_test_num, dtype=torch.float32)
    X_train_cat = torch.tensor(X_train_cat, dtype=torch.long)
    X_test_cat = torch.tensor(X_test_cat, dtype=torch.long)

    n_tokens = 1 + d_numerical + len(categories)
    batch_size = _env_int("TABSYN_VAE_BATCH_SIZE", 0) or _auto_batch_size(n_tokens)
    eval_chunk = _env_int("TABSYN_VAE_EVAL_BATCH_SIZE", 0) or _env_int("TABSYN_VAE_INFER_BATCH_SIZE", 0) or batch_size
    encode_chunk = _env_int("TABSYN_VAE_ENCODE_BATCH_SIZE", 0) or _env_int("TABSYN_VAE_INFER_BATCH_SIZE", 0) or batch_size
    num_epochs = _env_int("TABSYN_VAE_EPOCHS", UPSTREAM_EPOCHS)
    log_every = _env_int("TABSYN_LOG_EVERY", max(1, num_epochs // 50))

    n_train = X_train_num.shape[0]
    print(
        f"[TabSyn][VAE] rows={n_train} val_rows={X_test_num.shape[0]} d_numerical={d_numerical} "
        f"n_cat={len(categories)} tokens={n_tokens} epochs={num_epochs} batch_size={batch_size} "
        f"steps/epoch={(n_train + batch_size - 1) // batch_size} device={device}",
        flush=True,
    )

    X_train_num = _place(X_train_num, device)
    X_train_cat = _place(X_train_cat, device)
    X_test_num = X_test_num.to(device)
    X_test_cat = X_test_cat.to(device)

    model = Model_VAE(NUM_LAYERS, d_numerical, categories, D_TOKEN, n_head=N_HEAD, factor=FACTOR, bias=True)
    model = model.to(device)

    if os.environ.get("TABSYN_RESUME", "0") not in ("", "0", "false", "False"):
        if os.path.exists(model_save_path):
            try:
                model.load_state_dict(torch.load(model_save_path, map_location=device))
                print(f"[TabSyn][VAE] Resume from {model_save_path}")
            except Exception as e:
                print(f"[TabSyn][VAE] Resume skipped ({e})")

    pre_encoder = Encoder_model(NUM_LAYERS, d_numerical, categories, D_TOKEN, n_head=N_HEAD, factor=FACTOR).to(device)
    pre_decoder = Decoder_model(NUM_LAYERS, d_numerical, categories, D_TOKEN, n_head=N_HEAD, factor=FACTOR).to(device)

    pre_encoder.eval()
    pre_decoder.eval()

    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.95, patience=10)

    best_train_loss = float('inf')
    current_lr = optimizer.param_groups[0]['lr']
    patience = 0

    beta = max_beta
    start_time = time.time()
    for epoch in range(num_epochs):
        curr_loss_multi = 0.0
        curr_loss_gauss = 0.0
        curr_loss_kl = 0.0
        curr_count = 0

        model.train()
        perm = torch.randperm(n_train, device=X_train_num.device)
        for s in range(0, n_train, batch_size):
            idx = perm[s:s + batch_size]
            batch_num = X_train_num[idx].to(device, non_blocking=True)
            batch_cat = X_train_cat[idx.to(X_train_cat.device)].to(device, non_blocking=True)

            optimizer.zero_grad()
            Recon_X_num, Recon_X_cat, mu_z, std_z = model(batch_num, batch_cat)
            loss_mse, loss_ce, loss_kld, train_acc = compute_loss(batch_num, batch_cat, Recon_X_num, Recon_X_cat, mu_z, std_z)

            loss = loss_mse + loss_ce + beta * loss_kld
            loss.backward()
            optimizer.step()

            batch_length = batch_num.shape[0]
            curr_count += batch_length
            curr_loss_multi += loss_ce.item() * batch_length
            curr_loss_gauss += loss_mse.item() * batch_length
            curr_loss_kl += loss_kld.item() * batch_length

        num_loss = curr_loss_gauss / curr_count
        cat_loss = curr_loss_multi / curr_count
        kl_loss = curr_loss_kl / curr_count

        # Evaluation (chunked)
        model.eval()
        with torch.no_grad():
            n_te = X_test_num.shape[0]
            sum_mse = sum_ce = sum_acc = 0.0
            for s in range(0, n_te, eval_chunk):
                e = min(s + eval_chunk, n_te)
                bn = X_test_num[s:e]
                bc = X_test_cat[s:e]
                Recon_X_num, Recon_X_cat, mu_z, std_z = model(bn, bc)
                vm, vc, _, va = compute_loss(bn, bc, Recon_X_num, Recon_X_cat, mu_z, std_z)
                w = float(e - s)
                sum_mse += vm.item() * w
                sum_ce += float(vc) * w
                sum_acc += float(va) * w
            val_mse_loss = sum_mse / n_te
            val_ce_loss = sum_ce / n_te
            val_acc = sum_acc / n_te
            val_loss = val_ce_loss  # upstream: val_mse * 0 + val_ce

            scheduler.step(val_loss)
            new_lr = optimizer.param_groups[0]['lr']
            if new_lr != current_lr:
                current_lr = new_lr

            if val_loss < best_train_loss:
                best_train_loss = val_loss
                patience = 0
                torch.save(model.state_dict(), model_save_path)
            else:
                patience += 1
                if patience == 10:
                    if beta > min_beta:
                        beta = beta * lambd

        if epoch % log_every == 0 or epoch == num_epochs - 1:
            print(
                'epoch: {}, beta = {:.6f}, lr = {:.2e}, Train MSE: {:.6f}, Train CE:{:.6f}, Train KL:{:.6f}, '
                'Val MSE:{:.6f}, Val CE:{:.6f}, Train ACC:{:6f}, Val ACC:{:6f}, elapsed {:.1f}s'.format(
                    epoch, beta, current_lr, num_loss, cat_loss, kl_loss, val_mse_loss, val_ce_loss,
                    float(train_acc), val_acc, time.time() - start_time),
                flush=True,
            )

    end_time = time.time()
    print('Training time: {:.4f} mins'.format((end_time - start_time) / 60))

    # Saving latent embeddings (upstream uses the final model weights here)
    with torch.no_grad():
        pre_encoder.load_weights(model)
        pre_decoder.load_weights(model)

        torch.save(pre_encoder.state_dict(), encoder_save_path)
        torch.save(pre_decoder.state_dict(), decoder_save_path)

        print('Successfully load and save the model!')

        z_parts = []
        for s in range(0, n_train, encode_chunk):
            e = min(s + encode_chunk, n_train)
            bn = X_train_num[s:e].to(device)
            bc = X_train_cat[s:e].to(device)
            z_parts.append(pre_encoder(bn, bc).detach().cpu().numpy())
        train_z = np.concatenate(z_parts, axis=0)

        np.save(f'{ckpt_dir}/train_z.npy', train_z)

        print('Successfully save pretrained embeddings in disk!')


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Variational Autoencoder')

    parser.add_argument('--dataname', type=str, default='adult', help='Name of dataset.')
    parser.add_argument('--gpu', type=int, default=0, help='GPU index.')
    parser.add_argument('--max_beta', type=float, default=1e-2, help='Initial Beta.')
    parser.add_argument('--min_beta', type=float, default=1e-5, help='Minimum Beta.')
    parser.add_argument('--lambd', type=float, default=0.7, help='Decay of Beta.')

    args = parser.parse_args()

    if args.gpu != -1 and torch.cuda.is_available():
        args.device = 'cuda:{}'.format(args.gpu)
    else:
        args.device = 'cpu'
    main(args)
