import torch

from ldm_repro.autoencoder import AutoencoderKL, AutoencoderVQ, DiagonalGaussian, VectorQuantizer
from ldm_repro.conditioning import ClassEmbedder, TextEncoder
from ldm_repro.diffusion import Schedule, ddim_sample, guided_eps
from ldm_repro.unet import UNet


def test_autoencoder_downsampling_factor():
    ae = AutoencoderKL(8, (1, 2, 2), 4, 1)
    rec, post = ae(torch.rand(2, 3, 32, 32))
    assert post.mean.shape == (2, 4, 8, 8) and rec.shape == (2, 3, 32, 32) and ae.f == 4
    vq = AutoencoderVQ(8, (1, 2), 3, 16, 1)
    r, l = vq(torch.rand(1, 3, 16, 16))
    assert r.shape == (1, 3, 16, 16) and l > 0


def test_kl_zero_for_standard_normal():
    p = DiagonalGaussian(torch.zeros(2, 8, 4, 4))
    assert torch.allclose(p.kl(), torch.zeros(2))


def test_vq_straight_through():
    vq = VectorQuantizer(8, 4)
    z = torch.randn(1, 4, 3, 3, requires_grad=True)
    q, _, idx = vq(z)
    q.sum().backward()
    assert torch.allclose(z.grad, torch.ones_like(z))
    assert torch.allclose(q, vq.codebook(idx).view(1, 3, 3, 4).permute(0, 3, 1, 2), atol=1e-5)


def test_schedule_endpoints():
    s = Schedule()
    assert abs(s.betas[0].item() - 0.00085) < 1e-9 and abs(s.betas[-1].item() - 0.012) < 1e-9
    assert s.alphas_cumprod[-1] < 0.01 and s.alphas_cumprod[0] > 0.99


def test_unet_cross_attention_uses_context():
    torch.manual_seed(0)
    u = UNet(4, 16, (1, 2), (1,), 1, 2, d_ctx=8).eval()
    z, t = torch.randn(2, 4, 8, 8), torch.tensor([10, 500])
    a = u(z, t, context=torch.randn(2, 3, 8))
    b = u(z, t, context=torch.randn(2, 3, 8))
    assert a.shape == z.shape and not torch.allclose(a, b)


def test_ddim_recovers_x0_with_oracle():
    """With an oracle eps-predictor for a fixed x0, deterministic DDIM (eta=0) returns x0."""
    s = Schedule()
    x0 = torch.randn(1, 2, 3, 3)
    ac = s.alphas_cumprod.float()
    oracle = lambda z, t: (z - ac[t].view(-1, 1, 1, 1).sqrt() * x0) / (1 - ac[t]).view(-1, 1, 1, 1).sqrt()
    out = ddim_sample(oracle, s, x0.shape, steps=20, eta=0.0, g=torch.Generator().manual_seed(0))
    assert torch.allclose(out, x0, atol=1e-3)


def test_classifier_free_guidance_formula():
    m = lambda z, t, c: z * c
    z = torch.ones(1)
    assert torch.allclose(guided_eps(m, z, 0, {"c": 2.0}, {"c": 1.0}, 3.0), torch.tensor([4.0]))


def test_conditioners():
    e = ClassEmbedder(10, 16)
    assert e(torch.tensor([0, e.null])).shape == (2, 1, 16)
    assert TextEncoder(100, 32, 1, 4)(torch.randint(0, 100, (2, 7))).shape == (2, 7, 32)
