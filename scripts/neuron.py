"""InceptionV1 (Lucid weights via Lucent) with a pre-ReLU hook on mixed4e channel 55.

Mixed4e concatenates four branches; channels 0-255 are the 1x1 branch, so
channel 55 of mixed4e is channel 55 of the 1x1 conv, read here before its ReLU.
"""
import numpy as np
import torch
from PIL import Image
from lucent.modelzoo import inceptionv1

CHANNEL = 55
HOOK_LAYER = "mixed4e_1x1_pre_relu_conv"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class Neuron:
    def __init__(self):
        self.model = inceptionv1(pretrained=True, redirected_ReLU=False).to(DEVICE).eval()
        self._act = None
        getattr(self.model, HOOK_LAYER).register_forward_hook(self._hook)

    def _hook(self, module, inp, out):
        self._act = out[:, CHANNEL].detach()

    @staticmethod
    def preprocess(img: Image.Image) -> torch.Tensor:
        """Resize the short side to 224, center-crop, scale to [0, 1].

        The Lucid input transform (x * 255 - 117) is applied in maps(), matching
        lucent.optvis.transform.preprocess_inceptionv1.
        """
        img = img.convert("RGB")
        w, h = img.size
        s = 224 / min(w, h)
        img = img.resize((max(224, round(w * s)), max(224, round(h * s))), Image.BICUBIC)
        w, h = img.size
        l, t = (w - 224) // 2, (h - 224) // 2
        img = img.crop((l, t, l + 224, t + 224))
        return torch.from_numpy(np.asarray(img, dtype=np.float32) / 255).permute(2, 0, 1)

    @torch.no_grad()
    def maps(self, batch: torch.Tensor) -> torch.Tensor:
        """Pre-ReLU spatial maps, shape (N, H, W)."""
        self.model(batch.to(DEVICE) * 255 - 117)
        return self._act.cpu()

    def measure(self, images):
        """Return (max, mean, spatial map) per PIL image."""
        x = torch.stack([self.preprocess(im) for im in images])
        m = self.maps(x)
        return [(float(a.max()), float(a.mean()), a.numpy()) for a in m]
