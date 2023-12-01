"""
Extremely Minimalistic Implementation of DDPM

https://arxiv.org/abs/2006.11239

Everything is self contained. (Except for pytorch and torchvision... of course)

run it with `python superminddpm.py`
"""

from typing import Dict, Tuple
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from torchvision.datasets import CIFAR10
from torchvision import transforms
from torchvision.utils import save_image, make_grid
import transformers
transformers.logging.set_verbosity_error()
from transformers import CLIPTextModel, AutoTokenizer

from unet import NaiveUnet
import os


def ddpm_schedules(beta1: float, beta2: float, T: int) -> Dict[str, torch.Tensor]:
    """
    Returns pre-computed schedules for DDPM sampling, training process.
    """
    assert beta1 < beta2 < 1.0, "beta1 and beta2 must be in (0, 1)"

    beta_t = (beta2 - beta1) * torch.arange(0, T + 1, dtype=torch.float32) / T + beta1
    sqrt_beta_t = torch.sqrt(beta_t)
    alpha_t = 1 - beta_t
    log_alpha_t = torch.log(alpha_t)
    alphabar_t = torch.cumsum(log_alpha_t, dim=0).exp()

    sqrtab = torch.sqrt(alphabar_t)
    oneover_sqrta = 1 / torch.sqrt(alpha_t)

    sqrtmab = torch.sqrt(1 - alphabar_t)
    mab_over_sqrtmab_inv = (1 - alpha_t) / sqrtmab

    return {
        "alpha_t": alpha_t,  # \alpha_t
        "oneover_sqrta": oneover_sqrta,  # 1/\sqrt{\alpha_t}
        "sqrt_beta_t": sqrt_beta_t,  # \sqrt{\beta_t}
        "alphabar_t": alphabar_t,  # \bar{\alpha_t}
        "sqrtab": sqrtab,  # \sqrt{\bar{\alpha_t}}
        "sqrtmab": sqrtmab,  # \sqrt{1-\bar{\alpha_t}}
        "mab_over_sqrtmab": mab_over_sqrtmab_inv,  # (1-\alpha_t)/\sqrt{1-\bar{\alpha_t}}
    }

class DDPM(nn.Module):
    def __init__(
        self,
        eps_model: nn.Module,
        betas: Tuple[float, float],
        n_T: int,
    ) -> None:
        super(DDPM, self).__init__()
        self.eps_model = eps_model

        # register_buffer allows us to freely access these tensors by name. It helps device placement.
        for k, v in ddpm_schedules(betas[0], betas[1], n_T).items():
            self.register_buffer(k, v)

        self.n_T = n_T

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """
        Makes forward diffusion x_t, and tries to guess epsilon value from x_t using eps_model.
        This implements Algorithm 1 in the paper.
        """

        _ts = torch.randint(1, self.n_T, (x.shape[0],)).to(
            x.device
        )  # t ~ Uniform(0, n_T)
        eps = torch.randn_like(x)  # eps ~ N(0, 1)

        x_t = (
            self.sqrtab[_ts, None, None, None] * x
            + self.sqrtmab[_ts, None, None, None] * eps
        )  # This is the x_t, which is sqrt(alphabar) x_0 + sqrt(1-alphabar) * eps
        # We should predict the "error term" from this x_t. Loss is what we return.

        return (eps - self.eps_model(x_t, _ts / self.n_T, context)).square().mean()

    def ddim_sample(self, n_sample: int, context, size, device, n_steps=None, eta = 0) -> torch.Tensor:
        if n_steps is None:
            n_steps = self.n_T
        assert n_sample%n_steps == 0
        step_size = n_sample//n_steps
        x_i = torch.randn(n_sample, *size).to(device)  # x_T ~ N(0, 1)

        for i in range(self.n_T, 1, -step_size):
            z = torch.randn(n_sample, *size).to(device) if i > 1 else 0
            eps = self.eps_model(x_i, torch.tensor(i / self.n_T).to(device).repeat(n_sample, 1), context)
            x0_t = (x_i - eps * (1 - self.alphabar_t[i]).sqrt()) / self.alphabar_t[i].sqrt()
            c1 = eta * ((1 - self.alphabar_t[i] / self.alphabar_t[i - 1]) * (1 - self.alphabar_t[i - 1]) / (
                    1 - self.alphabar_t[i])).sqrt()
            c2 = ((1 - self.alphabar_t[i - 1]) - c1 ** 2).sqrt()
            x_i = self.alphabar_t[i - 1].sqrt() * x0_t + c1 * z + c2 * eps

        return x_i

def label_to_text_prompt(label):
    cifar10_classes = [
        "airplane",
        "automobile",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
    ]
    return "a photo of a " + cifar10_classes[label]

def make_dataloader():
    def _make_dataloader():
        tf = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5,), (1.0))]
        )

        dataset = CIFAR10(
            "./data",
            train=True,
            download=True,
            transform=tf,
        )
        return DataLoader(dataset, batch_size=128, shuffle=True, num_workers=1)

    while True:
        dataloader = _make_dataloader()
        for x, y in dataloader:
            yield x, [label_to_text_prompt(label) for label in y.tolist()]

@torch.no_grad()
def clip_encode(text, clip_tokenizer, clip_text_model):
    clip_inputs = clip_tokenizer(text=text, return_tensors="pt", padding="max_length", max_length=77, truncation=True)
    for k, v in clip_inputs.items():
        clip_inputs[k] = v.to(clip_text_model.device)
    return clip_text_model(**clip_inputs, output_hidden_states=True).hidden_states[-2].float()

def train(
    n_iterations=10_000,
    device="cuda",
    n_T=1024,
    betas=(1e-4, 0.02),
    in_channels=3,
    n_feat=256,
    lr=2e-4,
):
    eps_model = NaiveUnet(in_channels=in_channels, out_channels=in_channels, n_feat=n_feat)
    ddpm = DDPM(eps_model=eps_model, betas=betas, n_T=n_T).to(device)
    print(f"model has {sum(p.numel() for p in ddpm.parameters())} parameters")
    optim = torch.optim.Adam(ddpm.parameters(), lr=lr)
    dataloader = make_dataloader()
    clip_text_model = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32").to(device).half()
    clip_tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    for i in tqdm(range(n_iterations), disable=True):
        loss_ema = None
        optim.zero_grad()
        img, text_strings = next(dataloader)
        img = img.to(device)
        import time
        s = time.time()
        context = clip_encode(text_strings, clip_tokenizer, clip_text_model)
        torch.cuda.synchronize()
        print('clip', time.time() -s)
        s = time.time()
        loss = ddpm(img, context)
        loss.backward()
        torch.cuda.synchronize()
        print('ddpm', time.time() -s)
        if loss_ema is None:
            loss_ema = loss.item()
        else:
            loss_ema = 0.9 * loss_ema + 0.1 * loss.item()
        if i % 100 == 0:
            tqdm.write(f"loss: {loss_ema:.4f}")
        optim.step()

        if (i+1)%500 == 0:
            ddpm.eval()
            with torch.no_grad():
                xh = ddpm.sample(16, (1, 32, 32), device)
                grid = make_grid(xh, nrow=4)
                os.path.makedirs("./samples/train/", exist_ok=True)
                save_image(grid, f"./samples/train/iter_{i}.png")

                # save model
                os.path.makedirs("./ckpt/", exist_ok=True)
                torch.save(ddpm, f"./ckpt/ddpm_iter_{i}.pt")
            ddpm.train()

def sample(checkpoint_path, device="cuda", n_steps=256, n_per_class=4, eta=0.0):
    with open(checkpoint_path, "rb") as f:
        ddpm = torch.load(f)
    ddpm.eval()
    
    clip_text_model = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")
    clip_tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")

    n_examples = n_per_class * 10
    classes = [x // n_per_class for x in range(n_examples)]
    text = [label_to_text_prompt(label) for label in classes]
    context = clip_encode(text, clip_tokenizer, clip_text_model)
    with torch.no_grad():
        xh = ddpm.ddim_sample(n_examples, (1, 32, 32), context, device, n_steps=n_steps, eta=eta)
        grid = make_grid(xh, nrow=n_examples)
        os.path.makedirs("./samples", exist_ok=True)
        i = len(os.listdir("./samples"))
        save_image(grid, f"./samples/ddpm_sample_{i}.png")

def main(mode, **kwargs):
    if mode == "train":
        train(**kwargs)
    elif mode == "sample":
        assert "checkpoint_path" in kwargs, "must pass in checkpoint_path"
        sample(**kwargs)
    else:
        raise ValueError("mode must be either 'train' or 'sample'")

if __name__ == "__main__":
    import fire
    fire.Fire(main)