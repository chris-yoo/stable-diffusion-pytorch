import torch
import numpy as np
from tqdm import tqdm
from ddpm import DDPMSampler


WIDTH = 512
HEIGHT = 512
LATENTS_WIDTH = WIDTH // 8
LATENTS_HEIGHT = HEIGHT // 8

# output = w * (output_conditioned - output_unconditioned) + output_unconditioned
# 여기서 w 가 cfg_scale

# prompt: str,   생성할 이미지에 대한 프롬프트 (사용자의 입력)
# uncond_prompt: str,   Unconditional Prompt (CFG 사용 시 필요) 보통 empty string
# input_image=None,   이미지 기반 생성 시 초기 이미지 (없을 수도 있음)
# strength=0.8,   기존 이미지와 생성된 이미지의 영향도 (0~1 범위)
# do_cfg=True,   Classifier-Free Guidance (CFG)를 사용할지 여부
# cfg_scale=7.5,   CFG 강도 (높을수록 프롬프트 반영 강함)
# sampler_name="ddpm",   샘플링 방법 (DDPM, DDIM 등)
# n_inference_steps=50,   Diffusion 과정에서 사용할 스텝 수
# models={},   사전 학습된 모델들 (예: UNet, VAE 등)
# seed=None,   랜덤 시드 (같은 값이면 동일한 이미지 생성)
# device=None,   실행할 장치 (GPU/CPU)
# idle_device=None,   사용하지 않는 경우 데이터를 이동할 장치
# tokenizer=None,   텍스트 프롬프트 토크나이저 (예: CLIP Tokenizer)


def generate(
    prompt: str,
    uncond_prompt: str,
    input_image=None,
    strength=0.8,
    do_cfg=True,
    cfg_scale=7.5,
    sampler_name="ddpm",
    n_inference_steps=50,
    models={},
    seed=None,
    device=None,
    idle_device=None,
    tokenizer=None,
):

    with torch.no_grad():  # inference 과정에서 사용 역전파사용안함

        if not (0 < strength <= 1):
            raise ValueError("Strength must be between 0 and 1")

        # 사용하지 않는 모델을 특정 디바이스(idle_divice) 로 이동하는 함수
        # idle: 활성화 되지 않거나 사용되지 않는 상태
        if idle_device:
            to_idle = lambda x: x.to(idle_device)
        else:
            to_idle = lambda x: x

        generator = torch.Generator(
            device=device
        )  # 랜덤시드 생성기 device=device 해당 device 에서 사용할 난수를 생성한다.
        if seed is None:
            generator.seed()  # 주어진 시드가 없으면 그냥 랜덤으로 시드를 생성
        else:
            generator.manual_seed(seed)

        clip = models["clip"]
        clip.to(device)

        if do_cfg:
            # Conver the prompt into tokens using the tokennizer
            # padding 을 통해서 seq_len 를 강제로 77에 맞춰준다.
            cond_tokens = tokenizer.batch_encode_plus(
                [prompt], padding="max_length", max_length=77
            )
            # (batch, seq_len) tensor 로 바꿔준다., type 설정 및, device 를 설정해준다.
            cond_tokens = torch.tensor(
                cond_tokens, dtype=torch.long, device=device
            ).input_ids
            # (batch, seq_len) -> (batch, seq_len, dim)
            cond_context = clip(cond_tokens)
            # Convert into a list of length Seq_Len=77
            uncond_tokens = tokenizer.batch_encode_plus(
                [uncond_prompt], dtype=torch.long, device=device
            ).input_ids
            # (Batch_Size, Seq_Len)
            uncond_tokens = torch.tensor(uncond_tokens, dtype=torch.long, device=device)
            # (Batch_Size, Seq_Len) -> (Batch_Size, Seq_Len, Dim)
            uncond_context = clip(uncond_tokens)
            # (Batch_Size, Seq_Len, Dim) + (Batch_Size, Seq_Len, Dim) -> (2 * Batch_Size, Seq_Len, Dim)
            # 나중에 inference 단계에서 latent 를 2배로 만들어준다.
            context = torch.cat([cond_context, uncond_context])

        else:
            # CFG 는 프롬프트 반영강도를 조절할 수 있지만 이경우는 조절할 수 없음
            # convert it into a list of tokens
            # output = output_conditioned

            tokens = tokenizer.batch_encode_plus(
                [prompt], padding="max_length", max_length=77
            )
            tokens = torch.tensor(tokens, dtype=torch.long, device=device)
            # (Batch_Size, Seq_Len, Dim)
            context = clip(context)

        to_idle(clip)

        # sampler 방식을 설정, step 수를 설정한다.
        if sampler_name == "ddpm":
            sampler = DDPMSampler(generator)
            sampler.set_inference_timesteps(n_inference_steps)
        else:
            raise ValueError(f"Unknown sampler {sampler_name}")

        latents_shape = (1, 4, LATENTS_HEIGHT, LATENTS_WIDTH)

        # 이미지가 있을때 이미지를 가지고 새로운 이미지를 만드는 task
        if input_image:
            encoder = models["encoder"]
            encoder.to(device)

            # PIL(Python Imaging Library) 객체에서 제공하는 이미지 크기 조절 함수이다. 이미지 데이터가 (512, 512) 로 변경됨
            input_image_tensor = input_image.resize((WIDTH, HEIGHT))
            input_image_tensor = np.array(input_image_tensor)
            # (height, width, channel)
            input_image_tensor = torch.tensor(
                input_image_tensor, dtype=torch.float32, device=device
            )
            input_image_tensor = rescale(input_image_tensor, (0, 255), (-1, 1))
            # (height, width, channels) -> (batch, height, width, channels)
            input_image_tensor = input_image_tensor.unsqueeze(0)
            # (batch, height, width, channels) ->  (batch, channels, height, width)
            input_image_tensor = input_image_tensor.permute(0, 3, 1, 2)

            # torch.Generator 의 객체 -> generator 만약 seed 가 None 이면 항상 다른 노이즈가 생성이 되고 seed 가 고정이 되면 deterministic 한 결과를 볼 수 있다.
            encoder_noise = torch.randn(
                latents_shape, generator=generator, device=device
            )

            # run the images through the encoder of the VAE
            # z 를 noisy 하게 바꾸는 과정에서 노이즈를 많이 추가하면 원래의 이미지랑 멀어진 이미지가 생성이 되고 더 다양하게 이미지가 나온다.
            # 노이즈를 적게 추가하면 원본 이미지와 비슷한 이미지가 생성이 된다.
            latents = encoder(input_image_tensor, encoder_noise)
            # VAE 의 encoder 에 노이즈를 넣어주는 이유: latent vector z 는 마지막에 정규분포에서 추출된 노이즈에 std 를 곱하고 mean 을 더해주는 방식으로 도출되게 된다. 이를 위해서 정규분포에서 생성된 노이즈를 같이 넣어주는 것이다.
            sampler.set_strength(strength=strength)
            # encoder → z_0 → 노이즈 추가 → z_T -> denoising → decoder 바로 encoder 의 출력을 가지고 denosing 하는 것이 아니라 노이즈를 추가해줌
            # 여기서 만약 T = 1000 이면 sampler.timesteps[0] -> 1000 이 된다.
            latents = sampler.add_noise(latents, sampler.timesteps[0])

            to_idle(encoder)

        else:
            # 이미지가 주어지지 않았을때 prompt 만 가지고 이미지를 생성 처음에 random z 를 생성한다.
            # If we are doing text-to_image, start with random noise N(0,I)
            # decoder 는 노이즈가 제거된 z0 를 input 으로 받는다.
            # encoder 의 output 은 z0 이다. 그리고 T step 만큼의 노이즈를 추가하여 zT 를 만든다.
            latents = torch.randn(latents_shape, generator=generator, device=device)

        # step 수가 50 이라는 것의 의미: 1000 -> 980 -> 960 -> ... -> 0 DDIM
        diffusion = models["diffusion"]
        diffusion.to(device)

        # sampler.timesteps = [1000, 980, ..., 0]
        timesteps = tqdm(sampler.timesteps)  # 진행상태를 알 수 있는 라이브래리
        for i, timestep in enumerate(timesteps):  # 0, 1000 / 1, 980
            # (1,320) 원래 스칼리 값을 해당 벡터로 변환한다.
            time_embedding = get_time_embedding(timestep).to(device)

            # (Batch_Size, 4, Latents_Height, Latents_Width)
            model_input = latents

            if do_cfg:
                # (Batch_Size, 4, Latents_Height, Latents_Width) -> (2*Batch_Size, 4, Latents_Height, Latents_Width) 데이터를 두배로 만든다.
                model_input = model_input.repeat(2, 1, 1, 1)

            # model_output is the predicted noise by the UNET
            model_output = diffusion(model_input, context, time_embedding)

            if do_cfg:
                # 0번째 차원에 대해서 두개의 텐서로 나눈다.
                output_cond, output_uncond = model_output.chunk(2)
                model_output = cfg_scale * (output_cond - output_uncond) + output_uncond

            # Remove noise predicted by the UNET , timestep, z, 노이즈를 입력으로 하고 덜 노이지한 입력을 반환 그리고 반복한다.
            latents = sampler.step(timestep, latents, model_output)

        to_idle(diffusion)

        decoder = models["decoder"]
        decoder.to(device)

        # (Batch_Size, 4, Latents_Height, Latents_Width) -> (Batch_Size, 3, Height, Width)
        images = decoder(latents)
        to_idle(decoder)

        images = rescale(images, (-1, 1), (0, 255), clamp=True)
        # (Batch_Size, Channel, Height, Width) -> (Batch_Size, Height, Width, Channel)
        images = images.permute(0, 2, 3, 1)
        images = images.to("cpu", torch.uint8).numpy()

        # 첫번째 이미지만 출력함.
        return images[0]


def rescale(z, old_range, new_range, clamp=False):
    old_min, old_max = old_range
    new_min, new_max = new_range
    x -= old_min
    x *= (new_max - new_min) / (old_max - old_min)
    x += new_min
    if clamp:
        x = x.clamp(new_min, new_max)
    return x


# time embedding 을 형성
def get_time_embedding(timestep):
    # (160,)
    # torch.arange(start=0, end=160, dtype=torch.float32) -> tensor([0, 1, 2, ..., 159])
    # tensor([0.0000, -0.00625, -0.0125, ..., -0.99375])
    # torch.pow(base, exponent) 여기서 10000을 각 값으로 거듭제곱한다.
    freqs = torch.pow(10000, -torch.arange(start=0, end=160, dtype=torch.float32) / 160)
    # (1,160)
    # timestep = 1000 → tensor([1000.])
    # [:, None] -> 기존 텐서는 (1,) 크기이지만, 이를 (1, 1) 크기로 확장
    # freqs: [None]을 추가하면 (1, 160) 크기로 확장됨
    x = torch.tensor([timestep], dtype=torch.float32)[:, None] * freqs[None]
    # (1,320)
    # cos 과 sin 을 적용해주고 concatenate
    return torch.cat([torch.cos(x), torch.sin(x)], dim=-1)
