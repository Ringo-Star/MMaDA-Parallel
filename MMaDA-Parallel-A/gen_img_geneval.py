import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import math
import time
from typing import Callable, List, Optional

import torch
from PIL import Image
from transformers import AutoTokenizer

from model import LLaDAForMultiModalGeneration
from utils.generation_utils import setup_seed
from utils.image_utils import (
    add_break_line, calculate_vq_params, decode_vq_to_image,
)
from utils.prompt_utils import generate_text_to_image_prompt
from generators.image_generation_generator import generate_image
from generators.parallel_generator import generate_ti2ti

# Special tokens shared with inference.py (TI2TI). For text-to-image we only
# need the image-answer delimiters.
SPECIAL_TOKENS = {
    "mask_token": 126336,
    "newline_token": 126084,
    "answer_start": 126354,   # BOA <answer>
    "answer_end": 126355,     # EOA </answer>
    "boi": 126349,            # begin-of-image
    "eoi": 126350,            # end-of-image
}


def cosine_schedule(t):
    return torch.cos(t * math.pi / 2)


# --------------------------------------------------------------------------- #
# Core text-to-image generation (single trajectory)
# --------------------------------------------------------------------------- #
def build_t2i_inputs(prompt_text, tokenizer, seq_len, token_grid_height,
                     token_grid_width, device):
    """Build the conditional input ids and unconditional ids for one prompt.

    Sequence layout (pure text-to-image):
        [ conditioning prompt tokens ] [BOA][BOI] [img mask tokens] [EOI][EOA]
    ``generate_image`` fills the masked image region.  ``code_start`` points to
    the first image token; the trailing two tokens ([EOI],[EOA]) are stripped
    by the generator when it extracts the VQ ids.
    """
    MASK = SPECIAL_TOKENS["mask_token"]
    NEW_LINE = SPECIAL_TOKENS["newline_token"]
    BOA = SPECIAL_TOKENS["answer_start"]
    EOA = SPECIAL_TOKENS["answer_end"]
    BOI = SPECIAL_TOKENS["boi"]
    EOI = SPECIAL_TOKENS["eoi"]

    input_prompt, uncon_prompt = generate_text_to_image_prompt(prompt_text)
    con_prompt_ids = tokenizer(input_prompt)["input_ids"]
    uncon_prompt_ids = tokenizer(uncon_prompt)["input_ids"]

    img_mask_token = add_break_line(
        [MASK] * seq_len, token_grid_height, token_grid_width, new_number=NEW_LINE
    )

    pred_token = [BOA, BOI] + img_mask_token + [EOI, EOA]
    full_input_ids = con_prompt_ids + pred_token

    # image tokens start 2 positions (BOA, BOI) after the conditioning prompt
    code_start = len(con_prompt_ids) + 2

    con_input = torch.tensor(full_input_ids, device=device).unsqueeze(0)
    uncon_ids = torch.tensor(uncon_prompt_ids, device=device).unsqueeze(0)
    return con_input, uncon_ids, code_start


def build_t2i_thinking_inputs(prompt_text, tokenizer, seq_len, token_grid_height,
                              token_grid_width, text_gen_length, device):
    """Build inputs for joint parallel (thinking-aware) text-to-image generation.

    Unlike :func:`build_t2i_inputs`, this lays out a TI2TI-style answer block with
    both an image region and a masked text ("thinking") region, matching
    ``inference.py``::

        [ conditioning prompt ] [BOA][BOI] [img mask] [EOI] [text mask] </answer>

    ``generate_ti2ti`` denoises the text and image regions jointly so the image
    attends to the partially-formed thinking text.  Returns the conditional input,
    the unconditional *text* ids (for CFG), and the region indices the generator
    needs (``image_start``, ``text_start``, ``text_end``).
    """
    MASK = SPECIAL_TOKENS["mask_token"]
    NEW_LINE = SPECIAL_TOKENS["newline_token"]
    BOA = SPECIAL_TOKENS["answer_start"]
    BOI = SPECIAL_TOKENS["boi"]
    EOI = SPECIAL_TOKENS["eoi"]

    input_prompt, uncon_prompt = generate_text_to_image_prompt(prompt_text)
    con_prompt_ids = tokenizer(input_prompt)["input_ids"]
    uncon_prompt_ids = tokenizer(uncon_prompt)["input_ids"]

    img_mask_token = add_break_line(
        [MASK] * seq_len, token_grid_height, token_grid_width, new_number=NEW_LINE
    )
    text_mask_tokens = [MASK] * text_gen_length
    end_token_ids = tokenizer("</answer>", add_special_tokens=False).input_ids

    pred_token = [BOA, BOI] + img_mask_token + [EOI] + text_mask_tokens + end_token_ids
    full_input_ids = con_prompt_ids + pred_token

    image_start = len(con_prompt_ids) + 2               # after BOA, BOI
    image_end = image_start + len(img_mask_token)
    text_start = image_end + 1                           # after EOI
    text_end = text_start + text_gen_length

    con_input = torch.tensor(full_input_ids, device=device).unsqueeze(0)
    uncon_text_ids = torch.tensor(uncon_prompt_ids, device=device).unsqueeze(0)
    return con_input, uncon_text_ids, image_start, text_start, text_end


@torch.no_grad()
def generate_single_image(model, tokenizer, vqvae, prompt_text, args, device,
                          generator=None):
    """Generate one image (PIL) for a text prompt."""
    vae_scale = 2 ** (len(vqvae.config.block_out_channels) - 1)
    seq_len, newline_every, token_grid_height, token_grid_width = calculate_vq_params(
        args.height, args.width, vae_scale
    )

    con_input, uncon_ids, code_start = build_t2i_inputs(
        prompt_text, tokenizer, seq_len, token_grid_height, token_grid_width, device
    )

    vq_ids = generate_image(
        model=model,
        prompt=con_input,
        seq_len=seq_len,
        newline_every=newline_every,
        timesteps=args.steps,
        mask_token_id=SPECIAL_TOKENS["mask_token"],
        newline_id=SPECIAL_TOKENS["newline_token"],
        temperature=args.temperature,
        cfg_scale=args.cfg_scale,
        uncon_ids=uncon_ids,
        code_start=code_start,
        codebook_size=args.codebook_size,
        noise_schedule=cosine_schedule,
        text_vocab_size=args.text_vocab_size,
        generator=generator,
        use_cache=args.use_cache,
        debug=False,
    )

    # generate_image returns ids in full vocab space (offset by text_vocab_size);
    # the VQ-VAE decoder expects codebook indices in [0, codebook_size).
    vq_codes = (vq_ids - args.text_vocab_size).clamp_(0, args.codebook_size - 1).long()

    out_img = decode_vq_to_image(
        vq_codes,
        save_path=None,
        image_height=args.height,
        image_width=args.width,
        vqvae=vqvae,
    )
    return out_img


@torch.no_grad()
def generate_single_image_with_thinking(model, tokenizer, vqvae, prompt_text, args,
                                        device, generator=None):
    """Generate one image (PIL) plus its "thinking" text via joint parallel decoding.

    Reuses ``generate_ti2ti`` so the text and image regions are denoised together
    (MMaDA-Parallel behaviour).  There is no input image in pure T2I, so image CFG
    (``cfg_img``) defaults to 0 and ``uncon_image`` is left ``None`` — guidance is
    driven by ``cfg_scale`` against the unconditional text prompt.
    """
    vae_scale = 2 ** (len(vqvae.config.block_out_channels) - 1)
    seq_len, newline_every, token_grid_height, token_grid_width = calculate_vq_params(
        args.height, args.width, vae_scale
    )

    con_input, uncon_text_ids, image_start, text_start, text_end = build_t2i_thinking_inputs(
        prompt_text, tokenizer, seq_len, token_grid_height, token_grid_width,
        args.text_gen_length, device,
    )

    image_tokens, generated_text = generate_ti2ti(
        model=model,
        input_ids=con_input,
        text_start=text_start,
        text_end=text_end,
        image_start=image_start,
        seq_len=seq_len,
        newline_every=newline_every,
        text_steps=args.text_steps,
        text_gen_length=args.text_gen_length,
        text_block_length=args.text_block_length,
        timesteps=args.steps,
        temperature=args.temperature,
        text_temperature=args.text_temperature,
        cfg_scale=args.cfg_scale,
        cfg_img=args.cfg_img,
        uncon_text=uncon_text_ids,
        uncon_image=None,
        tokenizer=tokenizer,
        remasking="low_confidence",
        noise_schedule=cosine_schedule,
        generator=generator,
        text_vocab_size=args.text_vocab_size,
        codebook_size=args.codebook_size,
    )

    # generate_ti2ti already returns ids in codebook space (it subtracts
    # text_vocab_size internally), so unlike generate_single_image we do NOT
    # offset again here.
    vq_codes = torch.tensor(image_tokens, dtype=torch.long, device=device).unsqueeze(0)
    vq_codes = vq_codes.clamp_(0, args.codebook_size - 1)

    out_img = decode_vq_to_image(
        vq_codes,
        save_path=None,
        image_height=args.height,
        image_width=args.width,
        vqvae=vqvae,
    )
    return out_img, generated_text


# --------------------------------------------------------------------------- #
# Test-time search / selection over multiple trajectories
# --------------------------------------------------------------------------- #
def score_candidates(candidates: List[Image.Image], prompt_text: str,
                     scorer: Optional[Callable] = None) -> List[float]:
    """Score a list of candidate images for a prompt.

    Hook for test-time selection (best-of-N, particle search, etc.).  By default
    no verifier is available so all candidates score equally.  Plug a reward
    model / CLIP / detector-based scorer here and return one float per image.
    """
    if scorer is not None:
        return [scorer(img, prompt_text) for img in candidates]
    return [0.0 for _ in candidates]


def _generate_one(model, tokenizer, vqvae, prompt_text, args, device, generator):
    """Generate a single ``(image, thinking)`` trajectory.

    ``thinking`` is the generated reasoning text in thinking-aware mode, or
    ``None`` when ``--thinking`` is disabled.
    """
    if args.thinking:
        return generate_single_image_with_thinking(
            model, tokenizer, vqvae, prompt_text, args, device, generator=generator
        )
    img = generate_single_image(
        model, tokenizer, vqvae, prompt_text, args, device, generator=generator
    )
    return img, None


@torch.no_grad()
def generate_with_search(model, tokenizer, vqvae, prompt_text, args, device,
                         base_seed, scorer=None):
    """Generate one final ``(image, thinking)`` using the configured search algorithm.

    - ``none``: a single trajectory.
    - ``best_of_n``: sample ``num_trajectories`` candidates and keep the
      highest-scoring one (requires a ``scorer`` to be meaningful).

    ``thinking`` is ``None`` unless ``--thinking`` is enabled.
    """
    if args.search_algorithm == "none" or args.num_trajectories <= 1:
        gen = torch.Generator(device=device).manual_seed(base_seed) if base_seed else None
        return _generate_one(model, tokenizer, vqvae, prompt_text, args, device, gen)

    if args.search_algorithm == "best_of_n":
        candidates = []  # list of (img, thinking)
        for k in range(args.num_trajectories):
            seed_k = base_seed + k if base_seed else None
            gen = torch.Generator(device=device).manual_seed(seed_k) if seed_k else None
            candidates.append(
                _generate_one(model, tokenizer, vqvae, prompt_text, args, device, gen)
            )
        images = [img for img, _ in candidates]
        scores = score_candidates(images, prompt_text, scorer=scorer)
        best_idx = int(max(range(len(scores)), key=lambda i: scores[i]))
        return candidates[best_idx]

    raise ValueError(f"Unknown search_algorithm: {args.search_algorithm}")


# --------------------------------------------------------------------------- #
# Main: iterate over GenEval metadata and write GenEval-compatible output tree
# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(
        description="Generate images for the GenEval benchmark (text-to-image)."
    )
    # Model / VAE
    parser.add_argument("--checkpoint", type=str, required=True, help="Fine-tuned checkpoint path")
    parser.add_argument("--vae_ckpt", type=str, required=True, help="VQ-VAE checkpoint path")

    # GenEval I/O
    parser.add_argument("--metadata_file", type=str,
                        default="../eval/geneval/prompts/evaluation_metadata.jsonl",
                        help="GenEval metadata jsonl (one prompt spec per line)")
    parser.add_argument("--outdir", type=str, required=True,
                        help="Output dir; GenEval tree outdir/<NNNNN>/samples/<NNNN>.png")
    parser.add_argument("--n_samples", type=int, default=4,
                        help="Images generated per prompt (GenEval default is 4)")

    # Image size
    parser.add_argument("--height", type=int, default=512, help="Output image height")
    parser.add_argument("--width", type=int, default=512, help="Output image width")

    # ---- Hyperparameters left open for tuning ----
    parser.add_argument("--steps", type=int, default=18,
                        help="Number of diffusion (MaskGit) timesteps")
    parser.add_argument("--cfg_scale", type=float, default=3.0, help="Classifier-free guidance scale")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed (0 disables seeding)")
    parser.add_argument("--use_cache", action="store_true", help="Enable model feature caching")

    # ---- Thinking-aware (joint parallel text+image) generation ----
    parser.add_argument("--thinking", action="store_true",
                        help="Enable joint parallel generation of a 'thinking' text "
                             "alongside the image (MMaDA-Parallel / generate_ti2ti)")
    parser.add_argument("--text_steps", type=int, default=256,
                        help="Number of text denoising steps (thinking mode)")
    parser.add_argument("--text_gen_length", type=int, default=256,
                        help="Length of the masked thinking-text region (thinking mode)")
    parser.add_argument("--text_block_length", type=int, default=32,
                        help="Text generation block length (thinking mode)")
    parser.add_argument("--text_temperature", type=float, default=0.7,
                        help="Sampling temperature for the thinking text (thinking mode)")
    parser.add_argument("--cfg_img", type=float, default=0.0,
                        help="Image CFG scale (thinking mode); 0 for pure T2I (no input image)")

    # ---- Test-time search / selection ----
    parser.add_argument("--search_algorithm", type=str, default="none",
                        choices=["none", "best_of_n"],
                        help="Trajectory search/selection strategy")
    parser.add_argument("--num_trajectories", type=int, default=1,
                        help="Candidate trajectories per sample when searching")

    # Vocab (auto-detected from config when omitted)
    parser.add_argument("--text_vocab_size", type=int, default=None,
                        help="Override text vocab size (else read from config)")
    parser.add_argument("--codebook_size", type=int, default=None,
                        help="Override VQ codebook size (else read from config)")

    # Sharding / resume
    parser.add_argument("--start_index", type=int, default=0,
                        help="First prompt index (inclusive) to process")
    parser.add_argument("--end_index", type=int, default=-1,
                        help="Last prompt index (exclusive); -1 = all")
    parser.add_argument("--skip_existing", action="store_true",
                        help="Skip samples whose output png already exists")
    args = parser.parse_args()

    if args.seed != 0:
        setup_seed(args.seed)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"Loading model from {args.checkpoint}...")
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, trust_remote_code=True)
    model = LLaDAForMultiModalGeneration.from_pretrained(
        args.checkpoint, torch_dtype=torch.bfloat16, device_map="auto",
    )
    model.eval()

    config = model.config
    if args.text_vocab_size is None:
        args.text_vocab_size = getattr(config, 'text_vocab_size', 126356)
    if args.codebook_size is None:
        args.codebook_size = getattr(config, 'codebook_size', 8192)
    print(f"Vocabulary config: text_vocab_size={args.text_vocab_size}, "
          f"codebook_size={args.codebook_size}")

    print(f"Loading VQ-VAE from {args.vae_ckpt}...")
    from diffusers import VQModel
    vqvae = VQModel.from_pretrained(args.vae_ckpt, subfolder="vqvae").to(device)

    # Load prompts
    with open(args.metadata_file, "r") as f:
        metadatas = [json.loads(line) for line in f if line.strip()]

    end_index = len(metadatas) if args.end_index < 0 else min(args.end_index, len(metadatas))
    os.makedirs(args.outdir, exist_ok=True)

    print(f"\n{'='*80}")
    print(f"GenEval generation: prompts [{args.start_index}, {end_index}) of {len(metadatas)}")
    print(f"steps={args.steps} cfg={args.cfg_scale} temp={args.temperature} "
          f"n_samples={args.n_samples} search={args.search_algorithm} "
          f"trajectories={args.num_trajectories}")
    if args.thinking:
        print(f"thinking=ON text_steps={args.text_steps} "
              f"text_gen_length={args.text_gen_length} cfg_img={args.cfg_img}")
    print(f"{'='*80}\n")

    t0 = time.time()
    for index in range(args.start_index, end_index):
        metadata = metadatas[index]
        prompt_text = metadata["prompt"]

        outpath = os.path.join(args.outdir, f"{index:05d}")
        sample_path = os.path.join(outpath, "samples")
        os.makedirs(sample_path, exist_ok=True)

        # GenEval reads outdir/<NNNNN>/metadata.jsonl per evaluate_images.py
        with open(os.path.join(outpath, "metadata.jsonl"), "w") as f:
            json.dump(metadata, f)

        for sample_idx in range(args.n_samples):
            img_file = os.path.join(sample_path, f"{sample_idx:04d}.png")
            if args.skip_existing and os.path.exists(img_file):
                continue

            # distinct seed per (prompt, sample) so samples differ
            base_seed = (args.seed + index * args.n_samples + sample_idx) if args.seed else 0
            img, thinking = generate_with_search(
                model, tokenizer, vqvae, prompt_text, args, device, base_seed,
                scorer=None,  # plug a verifier/reward model here for search
            )
            img.save(img_file)

            # Save the thinking text to a sibling subdir so it never gets picked
            # up by GenEval's samples/*.png glob.
            if thinking is not None:
                thinking_path = os.path.join(outpath, "thinking")
                os.makedirs(thinking_path, exist_ok=True)
                with open(os.path.join(thinking_path, f"{sample_idx:04d}.txt"),
                          "w", encoding="utf-8") as tf:
                    tf.write(f"{thinking}\n")

        print(f"[{index:05d}] '{prompt_text}' -> {args.n_samples} sample(s)")

    print(f"\n[✓] Done. {end_index - args.start_index} prompts in {time.time() - t0:.1f}s")
    print(f"[✓] Output: {args.outdir}")
    print("Next: run eval/geneval/evaluation/evaluate_images.py on this directory.")


if __name__ == '__main__':
    main()
