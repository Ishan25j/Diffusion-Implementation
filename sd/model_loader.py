import numpy as np

from clip import CLIP
from encoder import VAE_Encoder
from decoder import VAE_Decoder
from diffusion import Diffusion
import model_converter

def preload_models_from_standard_weights(ckpt_path, device):
    state_dict = model_converter.load_from_standard_weights(ckpt_path, device)

    encoder = VAE_Encoder().to(device)
    encoder.load_state_dict(state_dict['encoder'], strict=True)

    decoder = VAE_Decoder().to(device)
    decoder.load_state_dict(state_dict['decoder'], strict=True)

    diffusion = Diffusion().to(device)
    diffusion.load_state_dict(state_dict['diffusion'], strict=True)

    clip = CLIP().to(device)
    clip.load_state_dict(state_dict['clip'], strict=True)

    return {
        'clip': clip,
        'encoder': encoder,
        'decoder': decoder,
        'diffusion': diffusion,
    }

def total_model_parameters():

    encoder = VAE_Encoder()

    decoder = VAE_Decoder()
    diffusion = Diffusion()

    clip = CLIP()

    parameters_encoder = filter(lambda p: p.requires_grad, encoder.parameters())
    parameters_encoder = sum([np.prod(p.size()) for p in parameters_encoder]) / 1_000_000
    
    parameters_decoder = filter(lambda p: p.requires_grad, decoder.parameters())
    parameters_decoder = sum([np.prod(p.size()) for p in parameters_decoder]) / 1_000_000

    parameters_diffusion = filter(lambda p: p.requires_grad, diffusion.parameters())
    parameters_diffusion = sum([np.prod(p.size()) for p in parameters_diffusion]) / 1_000_000

    parameters_clip = filter(lambda p: p.requires_grad, clip.parameters())
    parameters_clip = sum([np.prod(p.size()) for p in parameters_clip]) / 1_000_000

    parameters = parameters_encoder + parameters_decoder + parameters_diffusion + parameters_clip

    return 'Total Trainable Parameters: %.3fM' % parameters