import torch.nn.functional as F
import torch
import numpy as np
import os
from torchdiffeq import odeint_adjoint as odeint


class ExpVFM(torch.nn.Module):
    def __init__(
            self,
            num_classes: np.array,
            num_numerical_features: int,
            vf_fn,
            device=torch.device('cpu'),
            **kwargs
        ):

        super(ExpVFM, self).__init__()

        self.num_numerical_features = num_numerical_features
        self.num_classes = num_classes # it as a vector [K1, K2, ..., Km]
        self.num_classes_expanded = torch.from_numpy(
            np.concatenate([num_classes[i].repeat(num_classes[i]) for i in range(len(num_classes))])
        ).to(device) if len(num_classes)>0 else torch.tensor([]).to(device).int()
        self.neg_infinity = -1000000.0 

        offsets = np.cumsum(self.num_classes)
        offsets = np.append([0], offsets)
        self.slices_for_classes = []
        for i in range(1, len(offsets)):
            self.slices_for_classes.append(np.arange(offsets[i - 1], offsets[i]))
        self.offsets = torch.from_numpy(offsets).to(device)
        
        offsets = np.cumsum(self.num_classes) + np.arange(1, len(self.num_classes)+1)
        offsets = np.append([0], offsets)

        self._vf_fn = vf_fn
        self.device = device
    

    def mixed_loss(self, x):
        b = x.shape[0]
        dev = x.device

        x_num = x[:, :self.num_numerical_features]
        x_cat = x[:, self.num_numerical_features:].long()

        t = torch.rand(b, device=dev, dtype=x_num.dtype)
        t = t[:, None]
            
        # Continuous interpolation
        x_num_t = x_num
        if x_num.shape[1] > 0:
            noise = torch.randn_like(x_num)
            x_num_t = t * x_num + (1 - t) * noise # + noise * sigma_num
        
        # Discrete interpolation
        x_cat_oh = self.to_one_hot(x_cat).float()
        x_cat_t = x_cat_oh
        if x_cat.shape[1] > 0:
            x_cat_t = t * x_cat_oh + (1 - t) * torch.randn_like(x_cat_oh)

        # Predict orignal data (distribution)
        model_out_num, model_out_cat = self._vf_fn(x_num_t, x_cat_t, t.squeeze())

        d_loss = torch.zeros((1,)).float()
        c_loss = torch.zeros((1,)).float()

        # Compute the loss
        if x_num.shape[1] > 0:
            c_loss = self._mvgloss(model_out_num, x_num, t)

        if x_cat.shape[1] > 0:
            d_loss = self._absorbed_closs(model_out_cat, x_cat, self._vf_fn.categories)
            
        return d_loss.mean(), c_loss.mean()
    
    def _mvgloss(self, mu_t, x_num_t, t):
        n, k = mu_t.shape
        dev = mu_t.device
        dt = mu_t.dtype

        identity = torch.eye(k, device=dev, dtype=dt).unsqueeze(0).expand(n, -1, -1)
        scale = 1 - (1 - 0.01) * t.unsqueeze(1) ** 2
        sigma = scale * identity
        dist = torch.distributions.MultivariateNormal(mu_t, sigma)
        return -dist.log_prob(x_num_t).mean()

    @torch.no_grad()
    def sample(self, num_samples):
        dev = self.device
        dt = torch.float32
        d_in = self.num_numerical_features + sum(self.num_classes)
        d_out = self.num_numerical_features + len(self.num_classes)

        x0 = torch.randn(num_samples, d_in, device=dev)
        t = torch.tensor([0.0, 0.999]).to(dev)
        vf = Velocity(self._vf_fn)
        solver = os.environ.get("EFVFM_ODE_SOLVER", "dopri5").strip() or "dopri5"
        rtol = float(os.environ.get("EFVFM_ODE_RTOL", "1e-5"))
        atol = float(os.environ.get("EFVFM_ODE_ATOL", "1e-5"))
        fallback_enabled = os.environ.get("EFVFM_ODE_FALLBACK", "1").strip().lower() not in ("0", "false", "no")
        fallback_steps = max(3, int(os.environ.get("EFVFM_RK4_STEPS", "32")))
        fixed_step = solver in ("rk4", "euler", "midpoint", "explicit_adams", "implicit_adams")
        if fixed_step:
            # fixed-step solvers only step on the requested grid; [0, 0.999] would be a single step
            t = torch.linspace(0.0, 0.999, fallback_steps, device=dev)
        # cap adaptive-solver steps so a stiff / barely-trained field fails fast into the rk4 fallback
        max_steps = int(os.environ.get("EFVFM_ODE_MAX_STEPS", "2000"))
        options = None if fixed_step or max_steps <= 0 else {"max_num_steps": max_steps}
        try:
            trajectory = odeint(vf, x0, t, method=solver, rtol=rtol, atol=atol, options=options)
        except Exception as e:
            if (not fallback_enabled) or fixed_step:
                raise
            print(f"[EFVFM] ODE solver {solver} failed ({type(e).__name__}: {str(e)[:200]}); "
                  f"falling back to rk4 with {fallback_steps} grid points", flush=True)
            t_fb = torch.linspace(0.0, 0.999, fallback_steps, device=dev)
            trajectory = odeint(vf, x0, t_fb, method="rk4")
        out = trajectory[-1]  # state at t=0.999 (index 1 is only correct for a 2-point grid)

        sample = torch.zeros(num_samples, d_out, device=dev, dtype=dt)
        sample[:, :self.num_numerical_features] = out[:, :self.num_numerical_features].to(torch.float32)
        if sum(self.num_classes) != 0:
            idx = self.num_numerical_features
            for i, val in enumerate(self.num_classes):
                col = self.num_numerical_features + i
                sample[:, col] = torch.argmax(out[:, idx:idx + val], dim=1)
                idx += val
                assert val >= sample[:, col].max() >= 0, f"Sampled value {sample[:, col].max()} is out of range for categorical feature {i} with {val} classes."

        return sample.cpu()
    
    def sample_all(self, num_samples, batch_size, keep_nan_samples=False):        
        b = batch_size

        all_samples = []
        num_generated = 0
        while num_generated < num_samples:
            print(f"Samples left to generate: {num_samples-num_generated}")
            sample = self.sample(b)
            mask_nan = torch.any(sample.isnan(), dim=1)
            if keep_nan_samples:
                # If the sample instances that contains Nan are decided to be kept, the row with Nan will be foreced to all zeros
                sample = sample * (~mask_nan)[:, None]
            else:
                # Otherwise the instances with Nan will be eliminated
                sample = sample[~mask_nan]

            all_samples.append(sample)
            num_generated += sample.shape[0]

        x_gen = torch.cat(all_samples, dim=0)[:num_samples]

        return x_gen

    def to_one_hot(self, x_cat):
        if len(self.num_classes) == 0:
            return torch.zeros(x_cat.shape[0], 0, device=x_cat.device, dtype=torch.long)
        x_cat_oh = torch.cat(
            [F.one_hot(x_cat[:, i], num_classes=self.num_classes[i]) for i in range(len(self.num_classes))],
            dim=-1
        )
        return x_cat_oh
    
    def _absorbed_closs(self, model_output, x0, cats): #, sigma, dsigma):
        """
            alpha: (bs,)
        """
        cum_sum =0
        losses = torch.zeros(len(cats), device=model_output.device)
        for i, val in enumerate(cats):
            dist = torch.distributions.Categorical(logits=model_output[:, cum_sum:cum_sum+val])
            losses[i] = -dist.log_prob(x0[:, i]).mean()
            cum_sum += val

        loss = losses.sum()
        return loss
    

class Velocity(torch.nn.Module):
    def __init__(self, model):
        super(Velocity, self).__init__()
        self.model = model

    def forward(self, t, x):
        t = t * torch.ones(x.shape[0]).to(x.device)

        x_num = x[:, :self.model.d_numerical]
        x_cat = x[:, self.model.d_numerical:]
        mu, logits = self.model(x_num, x_cat, t)

        # Numerical velocity
        if self.model.d_numerical > 0:
            v_num = (mu - (1 - 0.01) * x_num) / (1 - (1 - 0.01) * t.unsqueeze(1))
        else:
            v_num = torch.zeros_like(x_num)

        # Categorical velocity: normalize logits into probability space before computing velocity
        if len(self.model.categories) > 0:
            v_cat_parts = []
            logit_idx = 0
            oh_idx = 0
            for k in self.model.categories:
                probs_k = F.softmax(logits[:, logit_idx:logit_idx + k], dim=-1)
                x_k = x_cat[:, oh_idx:oh_idx + k]
                v_k = (probs_k - (1 - 0.01) * x_k) / (1 - (1 - 0.01) * t.unsqueeze(1))
                v_cat_parts.append(v_k)
                logit_idx += k
                oh_idx += k
            v_cat = torch.cat(v_cat_parts, dim=1)
        else:
            v_cat = torch.zeros_like(x_cat)

        v_t = torch.cat([v_num, v_cat], dim=1)
        return v_t
