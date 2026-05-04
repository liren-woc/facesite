from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image


def _resolve_dtype(name: str):
    import torch

    normalized = name.strip().lower()
    if normalized == "float16":
        return torch.float16
    if normalized == "float32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")


def _positive_seed(seed: int):
    if seed >= 0:
        return seed

    import torch

    return int(torch.seed() % (2**31 - 1))


def _prepare_repo_imports(repo_dir: Path) -> None:
    os.chdir(repo_dir)
    sys.path.insert(0, str(repo_dir))


class LocalStableHair:
    def __init__(
        self,
        *,
        config_path: str,
        device: str,
        weight_dtype,
        pretrained_model_path: str | None = None,
    ) -> None:
        from omegaconf import OmegaConf

        self.config = OmegaConf.load(config_path)
        if pretrained_model_path:
            self.config.pretrained_model_path = pretrained_model_path
        self.device = device
        self.weight_dtype = weight_dtype

    def _maybe_enable_attention_slicing(self, pipeline) -> None:
        if hasattr(pipeline, "enable_attention_slicing"):
            pipeline.enable_attention_slicing("max")

    def _build_remove_hair_pipeline(self):
        import torch
        from diffusers import UniPCMultistepScheduler
        from diffusers.models import UNet2DConditionModel
        from ref_encoder.latent_controlnet import ControlNetModel
        from utils.pipeline_cn import StableDiffusionControlNetPipeline

        unet = UNet2DConditionModel.from_pretrained(
            self.config.pretrained_model_path,
            subfolder="unet",
            torch_dtype=self.weight_dtype,
        )
        bald_converter = ControlNetModel.from_unet(unet)
        bald_state = torch.load(self.config.bald_converter_path, map_location="cpu")
        bald_converter.load_state_dict(bald_state, strict=False)
        bald_converter.to(dtype=self.weight_dtype)
        del bald_state
        del unet

        pipeline = StableDiffusionControlNetPipeline.from_pretrained(
            self.config.pretrained_model_path,
            controlnet=bald_converter,
            safety_checker=None,
            torch_dtype=self.weight_dtype,
        )
        pipeline.scheduler = UniPCMultistepScheduler.from_config(pipeline.scheduler.config)
        self._maybe_enable_attention_slicing(pipeline)
        pipeline = pipeline.to(self.device)
        return pipeline

    def _build_transfer_components(self):
        import torch
        from diffusers import UniPCMultistepScheduler
        from diffusers.models import UNet2DConditionModel
        from ref_encoder.adapter import adapter_injection
        from ref_encoder.latent_controlnet import ControlNetModel
        from ref_encoder.reference_unet import ref_unet
        from utils.pipeline import StableHairPipeline

        unet = UNet2DConditionModel.from_pretrained(
            self.config.pretrained_model_path,
            subfolder="unet",
            torch_dtype=self.weight_dtype,
        )
        controlnet = ControlNetModel.from_unet(unet)
        controlnet_state = torch.load(
            os.path.join(self.config.pretrained_folder, self.config.controlnet_path),
            map_location="cpu",
        )
        controlnet.load_state_dict(controlnet_state, strict=False)
        controlnet.to(dtype=self.weight_dtype)
        del controlnet_state

        pipeline = StableHairPipeline.from_pretrained(
            self.config.pretrained_model_path,
            controlnet=controlnet,
            safety_checker=None,
            torch_dtype=self.weight_dtype,
        )
        pipeline.scheduler = UniPCMultistepScheduler.from_config(pipeline.scheduler.config)
        self._maybe_enable_attention_slicing(pipeline)
        pipeline = pipeline.to(self.device)

        hair_encoder = ref_unet.from_pretrained(
            self.config.pretrained_model_path,
            subfolder="unet",
        )
        encoder_state = torch.load(
            os.path.join(self.config.pretrained_folder, self.config.encoder_path),
            map_location="cpu",
        )
        hair_encoder.load_state_dict(encoder_state, strict=False)
        hair_encoder = hair_encoder.to(self.device, dtype=self.weight_dtype)
        del encoder_state

        hair_adapter = adapter_injection(
            pipeline.unet,
            device=self.device,
            dtype=self.weight_dtype,
            use_resampler=False,
        )
        adapter_state = torch.load(
            os.path.join(self.config.pretrained_folder, self.config.adapter_path),
            map_location="cpu",
        )
        hair_adapter.load_state_dict(adapter_state, strict=False)
        hair_adapter = hair_adapter.to(self.device, dtype=self.weight_dtype)
        del adapter_state
        del unet

        return pipeline, hair_encoder, hair_adapter

    def get_bald(self, image: Image.Image, *, scale: float) -> Image.Image:
        import torch

        width, height = image.size
        remove_hair_pipeline = self._build_remove_hair_pipeline()
        result = remove_hair_pipeline(
            prompt="",
            negative_prompt="",
            num_inference_steps=30,
            guidance_scale=1.5,
            width=width,
            height=height,
            image=image,
            controlnet_conditioning_scale=float(scale),
            generator=None,
        ).images[0]
        del remove_hair_pipeline
        if self.device == "cuda":
            torch.cuda.empty_cache()
        return result

    def transfer(
        self,
        *,
        source_image_path: str,
        reference_image_path: str,
        random_seed: int,
        step: int,
        guidance_scale: float,
        scale: float,
        controlnet_conditioning_scale: float,
        size: int,
    ) -> np.ndarray:
        import torch
        from ref_encoder.adapter import set_scale

        source_image = Image.open(source_image_path).convert("RGB").resize((size, size))
        reference_image = np.array(Image.open(reference_image_path).convert("RGB").resize((size, size)))
        source_bald = np.array(self.get_bald(source_image, scale=0.9))
        height, width, _ = source_bald.shape

        pipeline, hair_encoder, _hair_adapter = self._build_transfer_components()

        set_scale(pipeline.unet, float(scale))
        generator = torch.Generator(device=self.device)
        generator.manual_seed(int(random_seed))
        result = pipeline(
            "",
            negative_prompt="",
            num_inference_steps=int(step),
            guidance_scale=float(guidance_scale),
            width=width,
            height=height,
            controlnet_condition=source_bald,
            controlnet_conditioning_scale=float(controlnet_conditioning_scale),
            generator=generator,
            reference_encoder=hair_encoder,
            ref_image=reference_image,
        ).samples
        del pipeline
        del hair_encoder
        if self.device == "cuda":
            torch.cuda.empty_cache()
        return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a single Stable-Hair inference request.")
    parser.add_argument("--repo", required=True, help="Path to the Stable-Hair repository.")
    parser.add_argument("--face-path", required=True, help="Source face image path.")
    parser.add_argument("--shape-path", required=True, help="Reference hairstyle image path.")
    parser.add_argument("--result-path", required=True, help="Output image path.")
    parser.add_argument("--config", default="./configs/hair_transfer.yaml")
    parser.add_argument("--pretrained-model-path", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="float16", choices=["float16", "float32"])
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--guidance-scale", type=float, default=1.5)
    parser.add_argument("--hair-encoder-scale", type=float, default=1.0)
    parser.add_argument("--controlnet-conditioning-scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=-1)
    args = parser.parse_args()

    repo_dir = Path(args.repo).resolve()
    face_path = Path(args.face_path).resolve()
    shape_path = Path(args.shape_path).resolve()
    result_path = Path(args.result_path).resolve()

    if not repo_dir.exists():
        raise FileNotFoundError(f"Stable-Hair repo not found: {repo_dir}")
    if not face_path.exists():
        raise FileNotFoundError(f"Source face image not found: {face_path}")
    if not shape_path.exists():
        raise FileNotFoundError(f"Reference hairstyle image not found: {shape_path}")

    _prepare_repo_imports(repo_dir)

    import torch

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    dtype = _resolve_dtype(args.dtype)
    if device == "cpu" and dtype == torch.float16:
        dtype = torch.float32

    model = LocalStableHair(
        config_path=args.config,
        device=device,
        weight_dtype=dtype,
        pretrained_model_path=args.pretrained_model_path,
    )
    generated = model.transfer(
        source_image_path=str(face_path),
        reference_image_path=str(shape_path),
        random_seed=_positive_seed(args.seed),
        step=args.steps,
        guidance_scale=args.guidance_scale,
        scale=args.hair_encoder_scale,
        controlnet_conditioning_scale=args.controlnet_conditioning_scale,
        size=args.size,
    )

    result_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.clip(generated * 255.0, 0, 255).astype(np.uint8)).save(result_path)
    print(result_path)


if __name__ == "__main__":
    main()
