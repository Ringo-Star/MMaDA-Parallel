import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import time
import math
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoTokenizer
from model import LLaDAForMultiModalGeneration
from utils.generation_utils import setup_seed
from utils.image_utils import (
    preprocess_image, decode_vq_to_image, calculate_vq_params,
    generate_crop_size_list, var_center_crop, add_break_line, encode_img_with_breaks,
    encode_img_with_paint
)
# Reuse the exact sampling helpers from the original generator so behaviour matches.
from generators.parallel_generator import (
    add_gumbel_noise, mask_by_random_topk, get_num_transfer_tokens
)
from utils.prompt_utils import generate_text_image_to_text_image_prompt

SPECIAL_TOKENS = {
    "mask_token": 126336,
    "newline_token": 126084,
    "image_token_offset": 126356,
    "answer_start": 126354,
    "answer_end": 126355,
    "boi": 126349,
    "eoi": 126350,
    "uncondition": 126351
}
SYSTEM_PROMPT = (
    "Generate an image applying the following editing instruction based on the original image."
)

MASK_TOKEN = SPECIAL_TOKENS["mask_token"]
NEW_LINE = SPECIAL_TOKENS["newline_token"]


def cosine_schedule(t):
    return torch.cos(t * math.pi / 2)


def _extract_text(combined_input_ids, text_start, text_end, tokenizer):
    """Decode the currently-unmasked text tokens (masked ones are dropped)."""
    toks = combined_input_ids[0, text_start:text_end].cpu().tolist()
    n_masked = sum(1 for t in toks if t == MASK_TOKEN)
    toks = [t for t in toks if t != MASK_TOKEN]
    if tokenizer is not None:
        text = tokenizer.decode(toks, skip_special_tokens=True)
    else:
        text = str(toks)
    return text, n_masked


def _extract_image_vq(combined_input_ids, image_position_mapping,
                      text_vocab_size, codebook_size, fill_value=0):
    """Build a full-length VQ-code list for the current state.

    Masked image positions are filled with `fill_value` so the partial image can
    still be decoded; the count of masked positions is returned for logging.
    """
    vq_tokens = []
    n_masked = 0
    for pos in image_position_mapping:
        token = combined_input_ids[0, pos].item()
        if token == MASK_TOKEN:
            vq_tokens.append(fill_value)
            n_masked += 1
        else:
            v = token - text_vocab_size
            v = max(0, min(v, codebook_size - 1))
            vq_tokens.append(v)
    return vq_tokens, n_masked


def generate_ti2ti_process(
    model,
    input_ids,
    text_start,
    text_end,
    image_start,
    seq_len,
    newline_every,
    save_dir,
    vqvae,
    vae_ckpt,
    output_image_height,
    output_image_width,
    input_image=None,
    save_interval=1,
    save_partial_image=True,
    make_gif=True,
    text_steps=100,
    text_gen_length=256,
    text_block_length=64,
    timesteps=100,
    temperature=1.0,
    text_temperature=0.7,
    cfg_scale=0.0,
    cfg_img=4.0,
    uncon_text=None,
    uncon_image=None,
    tokenizer=None,
    remasking='low_confidence',
    noise_schedule=cosine_schedule,
    generator=None,
    text_vocab_size=126356,
    codebook_size=8192,
):
    """Interleaved TI2TI generation that dumps the intermediate text + image at
    every (sub-sampled) step so the thinking/denoising process can be probed.

    This mirrors `generators.parallel_generator.generate_ti2ti` exactly; the only
    additions are the per-step extraction and saving blocks.
    """

    device = input_ids.device

    steps_dir = os.path.join(save_dir, "steps")
    os.makedirs(steps_dir, exist_ok=True)
    log_path = os.path.join(save_dir, "thinking_log.txt")
    log_f = open(log_path, "w", encoding="utf-8")

    combined_input_ids = input_ids.clone()

    num_vq_tokens = seq_len
    total_image_len = seq_len + seq_len // newline_every
    image_end = image_start + total_image_len

    print(f"Interleaved generation: {text_steps} total steps")
    print(f"  - Text generation range: [{text_start}, {text_end})")
    print(f"  - Image generation range: [{image_start}, {image_end}) (total {total_image_len} including newlines)")
    print(f"  - VQ tokens: {num_vq_tokens}")
    print(f"  - Saving intermediate states to: {steps_dir}")

    text_masked_indices = combined_input_ids[:, text_start:text_end] == MASK_TOKEN
    num_transfer_tokens = get_num_transfer_tokens(text_masked_indices, text_steps)

    image_generation_step_indices = torch.linspace(
        text_steps // 4, text_steps - 1, timesteps
    ).round().int().tolist()
    image_gen_set = set(image_generation_step_indices)

    print(f"  - Image generation at steps: {image_generation_step_indices[:5]}...{image_generation_step_indices[-5:]}")

    image_position_mapping = []
    for i in range(image_start, image_end):
        if combined_input_ids[0, i] != NEW_LINE:
            image_position_mapping.append(i)

    assert len(image_position_mapping) == num_vq_tokens, \
        f"Expected {num_vq_tokens} VQ tokens, got {len(image_position_mapping)}"

    def _save_step(step, image_changed):
        """Persist current text (always) and image (when allowed) for this step."""
        text, n_text_masked = _extract_text(combined_input_ids, text_start, text_end, tokenizer)

        with open(os.path.join(steps_dir, f"step_{step:04d}.txt"), "w", encoding="utf-8") as tf:
            tf.write(text + "\n")

        img_frame = None
        n_img_masked = num_vq_tokens
        if save_partial_image:
            vq_tokens, n_img_masked = _extract_image_vq(
                combined_input_ids, image_position_mapping,
                text_vocab_size, codebook_size, fill_value=0
            )
            vq_tensor = torch.tensor(vq_tokens, dtype=torch.long, device=device).unsqueeze(0)
            img_path = os.path.join(steps_dir, f"step_{step:04d}.png")
            img_frame = decode_vq_to_image(
                vq_tensor,
                save_path=img_path,
                vae_ckpt=vae_ckpt,
                image_height=output_image_height,
                image_width=output_image_width,
                vqvae=vqvae,
            )
            if input_image is not None:
                w1, h1 = input_image.size
                w2, h2 = img_frame.size
                canvas = Image.new("RGB", (w1 + w2, max(h1, h2)), "white")
                canvas.paste(input_image, (0, 0))
                canvas.paste(img_frame, (w1, 0))
                canvas.save(os.path.join(steps_dir, f"step_{step:04d}_concat.png"))

        log_line = (
            f"[step {step:04d}] img_step={int(image_changed)} "
            f"text_masked={n_text_masked} img_masked={n_img_masked}/{num_vq_tokens} | "
            f"text: {text}"
        )
        log_f.write(log_line + "\n")
        log_f.flush()
        return img_frame

    gif_frames = []
    batch_size = combined_input_ids.shape[0]

    # ========== Interleaved Generation Loop ==========
    for step in tqdm(range(text_steps), desc="Interleaved generation (probe)"):

        with torch.no_grad():
            cond_logits = model(combined_input_ids, infer=True, use_cache=False).logits

        # ===== Text Generation Step =====
        text_masked_indices = combined_input_ids[:, text_start:text_end] == MASK_TOKEN

        if text_masked_indices.sum() > 0:
            text_logits = cond_logits[:, text_start:text_end, :]

            logits_with_noise = add_gumbel_noise(text_logits, temperature=text_temperature, generator=generator)
            x0 = torch.argmax(logits_with_noise, dim=-1)

            if remasking == 'low_confidence':
                p = F.softmax(text_logits.to(torch.float64), dim=-1)
                x0_p = torch.squeeze(torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1)
            elif remasking == 'random':
                if generator is not None:
                    x0_p = torch.rand(x0.shape, dtype=x0.dtype, device=x0.device, generator=generator)
                else:
                    x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
            else:
                raise NotImplementedError(remasking)

            x0 = torch.where(text_masked_indices, x0, combined_input_ids[:, text_start:text_end])
            confidence = torch.where(text_masked_indices, x0_p, -np.inf)

            transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
            for j in range(confidence.shape[0]):
                k = num_transfer_tokens[j, step].item()
                if k > 0:
                    _, select_index = torch.topk(confidence[j], k=k)
                    transfer_index[j, select_index] = True

            combined_input_ids[:, text_start:text_end][transfer_index] = x0[transfer_index]

        # ===== Image Generation Step (scheduled) =====
        image_changed = step in image_gen_set
        if image_changed:
            vq_tokens_list = []
            for pos in image_position_mapping:
                token = combined_input_ids[0, pos].item()
                if token == MASK_TOKEN:
                    vq_tokens_list.append(-1)
                else:
                    vq_token = token - text_vocab_size
                    vq_token = max(0, min(vq_token, codebook_size - 1))
                    vq_tokens_list.append(vq_token)

            vq_tokens_tensor = torch.tensor(vq_tokens_list, device=device).unsqueeze(0)
            unknown_map = vq_tokens_tensor == -1

            cond_image_logits_list = []
            for pos in image_position_mapping:
                cond_image_logits_list.append(cond_logits[:, pos:pos+1, text_vocab_size:text_vocab_size+codebook_size])
            cond_vq_logits = torch.cat(cond_image_logits_list, dim=1)

            if (cfg_scale > 0.0 and uncon_text is not None) or (cfg_img > 0.0 and uncon_image is not None):
                if uncon_text is None:
                    combined_uncond_text = combined_input_ids.clone()
                else:
                    combined_uncond_text = combined_input_ids.clone()
                    prefix_len = uncon_text.shape[1]
                    combined_uncond_text[:, :prefix_len] = uncon_text.to(device)

                if uncon_image is None:
                    combined_uncond_img = combined_input_ids.clone()
                else:
                    combined_uncond_img = combined_input_ids.clone()
                    prefix_len_img = uncon_image.shape[1]
                    combined_uncond_img[:, :prefix_len_img] = uncon_image.to(device)

                with torch.no_grad():
                    uncond_text_logits_full = model(combined_uncond_text, infer=True, use_cache=False).logits
                    uncond_img_logits_full = model(combined_uncond_img, infer=True, use_cache=False).logits

                uncond_text_vq_list = []
                uncond_img_vq_list = []
                for pos in image_position_mapping:
                    uncond_text_vq_list.append(uncond_text_logits_full[:, pos:pos+1, text_vocab_size:text_vocab_size+codebook_size])
                    uncond_img_vq_list.append(uncond_img_logits_full[:, pos:pos+1, text_vocab_size:text_vocab_size+codebook_size])

                uncond_text_vq_logits = torch.cat(uncond_text_vq_list, dim=1)
                uncond_img_vq_logits = torch.cat(uncond_img_vq_list, dim=1)
            else:
                uncond_text_vq_logits = torch.zeros_like(cond_vq_logits)
                uncond_img_vq_logits = torch.zeros_like(cond_vq_logits)

            if cfg_scale == 0.0 and cfg_img == 0.0:
                image_logits = cond_vq_logits
            else:
                image_logits = cond_vq_logits
                if cfg_scale != 0.0:
                    image_logits = image_logits + cfg_scale * (cond_vq_logits - uncond_text_vq_logits)
                if cfg_img != 0.0:
                    image_logits = image_logits + cfg_img * (cond_vq_logits - uncond_img_vq_logits)

            probs = F.softmax(image_logits, dim=-1)

            if temperature == 0:
                sampled_ids = probs.argmax(dim=-1)
            else:
                sampled = probs.reshape(-1, image_logits.size(-1))
                if generator is not None:
                    sampled_ids = torch.multinomial(sampled, 1, generator=generator)[:, 0].view(*image_logits.shape[:-1])
                else:
                    sampled_ids = torch.multinomial(sampled, 1)[:, 0].view(*image_logits.shape[:-1])

            sampled_ids = torch.where(unknown_map, sampled_ids, vq_tokens_tensor)
            sampled_ids = torch.clamp(sampled_ids, 0, codebook_size - 1)

            selected_probs = torch.gather(probs, -1, sampled_ids.long()[..., None]).squeeze(-1)

            high_val = torch.finfo(selected_probs.dtype).max
            selected_probs = torch.where(unknown_map, selected_probs, high_val)

            ratio = 1.0 * (step + 1) / text_steps
            mask_ratio = noise_schedule(torch.tensor(ratio, device=device))
            unknown_counts = unknown_map.sum(dim=-1, keepdim=True)
            mask_len = (num_vq_tokens * mask_ratio).floor().unsqueeze(0).to(device)
            mask_len = torch.max(torch.tensor([1], device=device), torch.min(unknown_counts - 1, mask_len.to(device).long()))
            if mask_len.ndim == 1:
                mask_len = mask_len.unsqueeze(1)

            img_temp = temperature * (1.0 - ratio)

            masking = mask_by_random_topk(mask_len, selected_probs, img_temp, generator=generator)

            final_vq_tokens = torch.where(masking, torch.tensor(-1, device=device), sampled_ids)

            for idx, pos in enumerate(image_position_mapping):
                v = final_vq_tokens[0, idx].item()
                if v == -1:
                    combined_input_ids[0, pos] = MASK_TOKEN
                else:
                    combined_input_ids[0, pos] = int(v + text_vocab_size)

        # ===== Per-step intermediate dump =====
        is_save_step = (step % save_interval == 0) or (step == text_steps - 1) or image_changed
        if is_save_step:
            frame = _save_step(step, image_changed)
            if make_gif and frame is not None:
                gif_frames.append(frame.copy())

    # ===== Extract final results (identical to original generator) =====
    text_tokens = combined_input_ids[0, text_start:text_end].cpu().tolist()
    text_tokens = [t for t in text_tokens if t != MASK_TOKEN]
    generated_text = tokenizer.decode(text_tokens, skip_special_tokens=True) if tokenizer is not None else text_tokens

    image_tokens = []
    for pos in image_position_mapping:
        token = combined_input_ids[0, pos].item()
        if token != MASK_TOKEN:
            vq_token = token - text_vocab_size
            vq_token = max(0, min(vq_token, codebook_size - 1))
            image_tokens.append(vq_token)
        else:
            image_tokens.append(int(torch.randint(0, codebook_size, (1,)).item()))

    log_f.close()

    if make_gif and len(gif_frames) > 1:
        gif_path = os.path.join(save_dir, "process.gif")
        gif_frames[0].save(
            gif_path, save_all=True, append_images=gif_frames[1:],
            duration=200, loop=0
        )
        print(f"  - Process GIF saved to: {gif_path}")

    print(f"Interleaved generation complete.")
    print(f"  - Generated text: {len(text_tokens)} tokens")
    print(f"  - Generated image: {len(image_tokens)} VQ tokens (range [0, {codebook_size}))")
    print(f"  - Per-step dumps in: {steps_dir}")
    print(f"  - Step log: {log_path}")

    return image_tokens, generated_text


def main():
    parser = argparse.ArgumentParser(description="Probe intermediate TI2TI thinking/denoising process")
    parser.add_argument("--checkpoint", type=str, required=True, help="Fine-tuned checkpoint path")
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt for editing")
    parser.add_argument("--image_path", type=str, required=True, help="Input image path")
    parser.add_argument("--height", type=int, default=512, help="Output image height")
    parser.add_argument("--width", type=int, default=512, help="Output image width")
    parser.add_argument("--timesteps", type=int, default=64, help="Number of diffusion timesteps")
    parser.add_argument("--text_steps", type=int, default=256, help="Number of text generation steps")
    parser.add_argument("--text_gen_length", type=int, default=256, help="Maximum text generation length")
    parser.add_argument("--text_block_length", type=int, default=32, help="Text generation block length")
    parser.add_argument("--cfg_scale", type=float, default=2.5, help="CFG scale for text")
    parser.add_argument("--cfg_img", type=float, default=4.0, help="CFG scale for image")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    parser.add_argument("--text_temperature", type=float, default=0.7, help="Text generation temperature")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--vae_ckpt", type=str, required=True, help="VAE checkpoint path")
    parser.add_argument("--output_dir", type=str, default="results_process", help="Output directory")
    parser.add_argument("--remasking", type=str, default="low_confidence",
                        choices=["low_confidence", "random"],
                        help="Remasking strategy")
    parser.add_argument("--painting_mode", type=str, default=None, help="If set, use painting-mode encoding")
    parser.add_argument("--mask_h_ratio", type=float, default=0.5, help="mask height ratio for painting mode")
    parser.add_argument("--mask_w_ratio", type=float, default=0.5, help="mask width ratio for painting mode")
    parser.add_argument("--debug_tokens", action="store_true", help="Print token debug info to verify sequence layout")
    # Probe-specific options
    parser.add_argument("--save_interval", type=int, default=1,
                        help="Save an intermediate snapshot every N text steps (image steps are always saved)")
    parser.add_argument("--no_partial_image", action="store_true",
                        help="Skip decoding/saving the partial image at each step (text only, much faster)")
    parser.add_argument("--no_gif", action="store_true", help="Do not assemble a process.gif from the frames")
    args = parser.parse_args()

    MASK = SPECIAL_TOKENS["mask_token"]
    NEW_LINE_LOCAL = SPECIAL_TOKENS["newline_token"]
    BOA = SPECIAL_TOKENS["answer_start"]
    EOA = SPECIAL_TOKENS["answer_end"]
    BOI = SPECIAL_TOKENS["boi"]
    EOI = SPECIAL_TOKENS["eoi"]

    if args.seed != 0:
        setup_seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Loading model from {args.checkpoint}...")
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, trust_remote_code=True)
    model = LLaDAForMultiModalGeneration.from_pretrained(
        args.checkpoint, torch_dtype=torch.bfloat16, device_map="auto",
    )

    config = model.config
    text_vocab_size = getattr(config, 'text_vocab_size', 126356)
    codebook_size = getattr(config, 'codebook_size', 8192)

    print(f"Vocabulary config: text_vocab_size={text_vocab_size}, codebook_size={codebook_size}")

    print(f"Loading VQ-VAE from {args.vae_ckpt}...")
    from diffusers import VQModel
    vqvae = VQModel.from_pretrained(args.vae_ckpt, subfolder="vqvae").to(device)
    vae_scale = 2 ** (len(vqvae.config.block_out_channels) - 1)

    prompt_text = args.prompt
    input_image_path = args.image_path

    print(f"\n{'='*80}")
    print(f"TI2TI Process Probe")
    print(f"{'='*80}")
    print(f"Input image: {input_image_path}")
    print(f"Prompt: {prompt_text}")
    print(f"Output size: {args.height}x{args.width}")
    print(f"{'='*80}\n")

    input_prompt, uncon_text = generate_text_image_to_text_image_prompt(
        prompt_text, SYSTEM_PROMPT
    )

    print("Conditioning prompt:\n", input_prompt)
    if args.debug_tokens:
        print("Unconditional text prompt (first 200 chars):", uncon_text[:200])

    prompt_ids = tokenizer(input_prompt)["input_ids"]
    uncon_text_ids = tokenizer(uncon_text)["input_ids"]

    img = Image.open(input_image_path).convert("RGB")
    crop_size_list = generate_crop_size_list((512 // 32) ** 2, 32)
    img = var_center_crop(img, crop_size_list=crop_size_list)

    input_image_width, input_image_height = img.size

    print("Encoding input image for conditioning...")
    input_img_token = encode_img_with_breaks(img, vqvae)

    con_input_list = prompt_ids[:-1] + input_img_token + prompt_ids[-1:]
    uncon_input_text = uncon_text_ids[:-1] + input_img_token + uncon_text_ids[-1:]
    uncon_input_image = prompt_ids

    output_image_height = args.height
    output_image_width = args.width
    seq_len, newline_every, token_grid_height, token_grid_width = calculate_vq_params(
        output_image_height, output_image_width, vae_scale
    )

    text_mask_tokens = [MASK] * args.text_gen_length

    if args.painting_mode:
        img_mask_token, img_vis = encode_img_with_paint(
            img, vqvae=vqvae, mask_h_ratio=args.mask_h_ratio, mask_w_ratio=args.mask_w_ratio, mask_mode=args.painting_mode
        )
    else:
        img_mask_token = add_break_line([MASK] * seq_len, token_grid_height, token_grid_width, new_number=NEW_LINE_LOCAL)

    end_token_ids = tokenizer("</answer>", add_special_tokens=False).input_ids

    pred_token = [BOA] + [BOI] + img_mask_token + [EOI] + text_mask_tokens + end_token_ids

    code_start = len(con_input_list)
    image_start = len(con_input_list) + 2
    image_end = image_start + len(img_mask_token)
    text_start = image_end + 1
    text_end = text_start + args.text_gen_length

    full_input_ids = con_input_list + pred_token
    con_input = torch.tensor(full_input_ids, device=device).unsqueeze(0)
    uncon_input_text = torch.tensor(uncon_input_text, device=device).unsqueeze(0)
    uncon_input_image = torch.tensor(uncon_input_image, device=device).unsqueeze(0)

    # Build a per-run output subfolder so all step dumps stay together.
    words = (prompt_text or "").split()
    filename_words = words[:10] if len(words) > 10 else words
    base_name = "_".join(filename_words)
    base_name = "".join(c for c in base_name if c.isalnum() or c in ('_', '-'))
    run_name = f"{base_name}_{output_image_height}x{output_image_width}_t{args.timesteps}_cfg{args.cfg_scale}_ti2ti"
    save_dir = os.path.join(args.output_dir, run_name)
    os.makedirs(save_dir, exist_ok=True)

    start_time = time.time()

    if args.seed != 0:
        generator = torch.Generator(device=device).manual_seed(args.seed)
    else:
        generator = None

    output_tokens, generated_text = generate_ti2ti_process(
        model=model,
        input_ids=con_input,
        text_start=text_start,
        text_end=text_end,
        image_start=image_start,
        seq_len=seq_len,
        newline_every=newline_every,
        save_dir=save_dir,
        vqvae=vqvae,
        vae_ckpt=args.vae_ckpt,
        output_image_height=output_image_height,
        output_image_width=output_image_width,
        input_image=img,
        save_interval=args.save_interval,
        save_partial_image=not args.no_partial_image,
        make_gif=not args.no_gif,
        text_steps=args.text_steps,
        text_gen_length=args.text_gen_length,
        text_block_length=args.text_block_length,
        timesteps=args.timesteps,
        temperature=args.temperature,
        text_temperature=args.text_temperature,
        cfg_scale=args.cfg_scale,
        cfg_img=args.cfg_img,
        uncon_text=uncon_input_text,
        uncon_image=uncon_input_image,
        tokenizer=tokenizer,
        remasking=args.remasking,
        noise_schedule=cosine_schedule,
        generator=generator,
        text_vocab_size=text_vocab_size,
        codebook_size=codebook_size,
    )

    end_time = time.time()
    elapsed_time = end_time - start_time

    print(f"\n{'='*80}")
    print(f"Final thinking/text output:")
    print(f"{'='*80}")
    print(generated_text)
    print(f"{'='*80}\n")

    print(f"Converting {len(output_tokens)} VQ tokens to tensor...")
    output_tokens_tensor = torch.tensor(output_tokens, dtype=torch.long, device=device).unsqueeze(0)
    print(f"VQ tokens range: [{min(output_tokens)}, {max(output_tokens)}]")

    save_path = os.path.join(save_dir, f"{run_name}.png")

    print("Decoding final image...")
    out_img = decode_vq_to_image(
        output_tokens_tensor,
        save_path,
        vae_ckpt=args.vae_ckpt,
        image_height=output_image_height,
        image_width=output_image_width,
        vqvae=vqvae
    )

    w1, h1 = img.size
    w2, h2 = out_img.size
    canvas = Image.new("RGB", (w1 + w2, max(h1, h2)), "white")
    canvas.paste(img, (0, 0))
    canvas.paste(out_img, (w1, 0))
    concat_path = save_path.replace(".png", "_concat.png")
    canvas.save(concat_path)

    text_path = save_path.replace(".png", "_thinking.txt")
    with open(text_path, "w", encoding="utf-8") as f:
        f.write(f"{generated_text}\n")

    print(f"\n[✓] Final image saved to: {concat_path}")
    print(f"[✓] Final text saved to: {text_path}")
    print(f"[✓] Per-step dumps in: {os.path.join(save_dir, 'steps')}")
    print(f"[✓] Total time: {elapsed_time:.2f}s")


if __name__ == '__main__':
    main()
