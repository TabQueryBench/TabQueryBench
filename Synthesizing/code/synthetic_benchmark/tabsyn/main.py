"""
TabSyn stage 2: latent diffusion training (vendored from amazon-science/tabsyn `tabsyn/main.py`).

TabQueryBench changes (model math unchanged):
- Env-driven epochs / batch / patience: TABSYN_DIFFUSION_EPOCHS (legacy alias TABSYN_DIFFUSION_MAX_EPOCHS),
  TABSYN_DIFFUSION_BATCH_SIZE (4096), TABSYN_DIFFUSION_PATIENCE (500).
- Tensor-index batching instead of a per-item DataLoader.
- Periodic model_{epoch}.pt snapshots off by default (TABSYN_DIFFUSION_SAVE_EVERY>0 to enable).
"""
import argparse
import os
import time
import warnings

import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau

from model import MLPDiffusion, Model
from latent_utils import get_input_train

warnings.filterwarnings('ignore')

UPSTREAM_EPOCHS = 10000 + 1


def _env_int(name, default):
    raw = os.environ.get(name, "").strip()
    return int(raw) if raw else default


def main(args):
    device = args.device

    train_z, _, _, ckpt_path, _ = get_input_train(args)
    os.makedirs(ckpt_path, exist_ok=True)

    in_dim = train_z.shape[1]

    mean = train_z.mean(0)
    train_z = (train_z - mean) / 2

    batch_size = _env_int("TABSYN_DIFFUSION_BATCH_SIZE", 4096)
    num_epochs = _env_int("TABSYN_DIFFUSION_EPOCHS", _env_int("TABSYN_DIFFUSION_MAX_EPOCHS", UPSTREAM_EPOCHS))
    patience_limit = _env_int("TABSYN_DIFFUSION_PATIENCE", 500)
    save_every = _env_int("TABSYN_DIFFUSION_SAVE_EVERY", 0)
    log_every = _env_int("TABSYN_LOG_EVERY", max(1, num_epochs // 50))

    n = train_z.shape[0]
    budget = float(os.environ.get("TABSYN_GPU_DATA_MAX_GB", "4")) * (1024 ** 3)
    if str(device).startswith("cuda") and train_z.element_size() * train_z.nelement() <= budget:
        train_z = train_z.to(device)

    print(
        f"[TabSyn][Diffusion] rows={n} in_dim={in_dim} epochs={num_epochs} batch_size={batch_size} "
        f"steps/epoch={(n + batch_size - 1) // batch_size} patience={patience_limit} ckpt={ckpt_path}",
        flush=True,
    )

    denoise_fn = MLPDiffusion(in_dim, 1024).to(device)
    num_params = sum(p.numel() for p in denoise_fn.parameters())
    print("the number of parameters", num_params)

    model = Model(denoise_fn=denoise_fn, hid_dim=in_dim).to(device)
    model_path = f'{ckpt_path}/model.pt'
    if os.environ.get("TABSYN_RESUME", "0") not in ("", "0", "false", "False"):
        if os.path.exists(model_path):
            try:
                model.load_state_dict(torch.load(model_path, map_location=device))
                print(f"[TabSyn][Diffusion] Resume from {model_path}")
            except Exception as e:
                print(f"[TabSyn][Diffusion] Resume skipped ({e})")

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=0)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.9, patience=20)

    model.train()

    best_loss = float('inf')
    patience = 0
    start_time = time.time()
    for epoch in range(num_epochs):
        batch_loss = 0.0
        len_input = 0
        perm = torch.randperm(n, device=train_z.device)
        for s in range(0, n, batch_size):
            inputs = train_z[perm[s:s + batch_size]].float().to(device, non_blocking=True)
            loss = model(inputs)
            loss = loss.mean()

            batch_loss += loss.item() * len(inputs)
            len_input += len(inputs)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        curr_loss = batch_loss / len_input
        scheduler.step(curr_loss)

        if curr_loss < best_loss:
            best_loss = curr_loss
            patience = 0
            torch.save(model.state_dict(), model_path)
        else:
            patience += 1
            if patience == patience_limit:
                print(f'Early stopping at epoch {epoch} (best loss {best_loss:.6f})')
                break

        if save_every > 0 and epoch % save_every == 0:
            torch.save(model.state_dict(), f'{ckpt_path}/model_{epoch}.pt')

        if epoch % log_every == 0 or epoch == num_epochs - 1:
            print(f"epoch {epoch}: loss {curr_loss:.6f} best {best_loss:.6f} "
                  f"lr {optimizer.param_groups[0]['lr']:.2e} elapsed {time.time() - start_time:.1f}s", flush=True)

    end_time = time.time()
    print('Time: ', end_time - start_time)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Training of TabSyn')

    parser.add_argument('--dataname', type=str, default='adult', help='Name of dataset.')
    parser.add_argument('--gpu', type=int, default=0, help='GPU index.')

    args = parser.parse_args()

    if args.gpu != -1 and torch.cuda.is_available():
        args.device = f'cuda:{args.gpu}'
    else:
        args.device = 'cpu'
    main(args)
