#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import json
import math
import os
import re
from io import BytesIO
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import numpy as np
import requests
from dotenv import load_dotenv
from PIL import Image


DEFAULT_OLLAMA_URL = "http://127.0.0.1:11434/api/chat"
DEFAULT_PROMPT_MODEL = "llava:latest"
SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
OUTPUT_WIDTH = 480
OUTPUT_HEIGHT = 832
OUTPUT_FPS = 16
OUTPUT_FRAMES = 81


@dataclass(frozen=True)
class ShotSpec:
    index: int
    slug: str
    camera_move: str
    prompt_hint: str


SHOT_SPECS = [
    ShotSpec(
        index=1,
        slug="orbital_left",
        camera_move="Orbital Left",
        prompt_hint=(
            "simulate a subtle left orbital move around the focal point while "
            "keeping walls, windows, cabinetry, and other architectural features stable"
        ),
    ),
    ShotSpec(
        index=2,
        slug="dolly_in_pan_right",
        camera_move="Dolly In + Pan Right",
        prompt_hint=(
            "simulate a gentle dolly-in with a slight pan right that reveals more "
            "of the room without inventing furniture, windows, or fixtures"
        ),
    ),
    ShotSpec(
        index=3,
        slug="truck_right",
        camera_move="Truck Right",
        prompt_hint=(
            "simulate a smooth truck-right move that preserves perspective and "
            "maintains accurate room geometry"
        ),
    ),
    ShotSpec(
        index=4,
        slug="pedestal_up",
        camera_move="Pedestal Up",
        prompt_hint=(
            "simulate a slow pedestal-up move that emphasizes ceiling height, "
            "light, and vertical composition without hallucinating details"
        ),
    ),
    ShotSpec(
        index=5,
        slug="dolly_out_pan_left",
        camera_move="Dolly Out + Pan Left",
        prompt_hint=(
            "simulate a gentle dolly-out with a slight pan left that broadens the "
            "scene while keeping all architecture believable"
        ),
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate five real-estate assessment videos from listing photos."
    )
    parser.add_argument(
        "--images-dir",
        default="input_images",
        help="Directory containing listing photos downloaded from the chosen listing.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs",
        help="Directory where prompts, manifest, and videos will be written.",
    )
    parser.add_argument(
        "--listing-context",
        default="listing_context.txt",
        help="Text file with property details to condition prompt generation.",
    )
    parser.add_argument(
        "--listing-title",
        default="421 Hudson Street #213",
        help="Human-readable listing title included in prompts and metadata.",
    )
    parser.add_argument(
        "--listing-url",
        default="https://streeteasy.com/building/the-printing-house/213?featured=1",
        help="Source listing URL.",
    )
    parser.add_argument(
        "--prompt-model",
        default=os.getenv("OLLAMA_MODEL", DEFAULT_PROMPT_MODEL),
        help="Ollama vision model used to automatically write prompts.",
    )
    parser.add_argument(
        "--prompts-only",
        action="store_true",
        help="Generate prompt JSON files only and skip hosted video generation.",
    )
    return parser.parse_args()


def discover_images(images_dir: Path) -> list[Path]:
    if not images_dir.exists():
        raise FileNotFoundError(
            f"Image directory '{images_dir}' does not exist. Save listing photos there first."
        )

    images = sorted(
        [
            path
            for path in images_dir.iterdir()
            if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
        ]
    )

    if not images:
        raise FileNotFoundError(
            f"No listing photos were found in '{images_dir}'. Add at least one JPG/PNG/WebP file."
        )

    return images


def read_listing_context(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(
            f"Listing context file '{path}' was not found. Create it or use --listing-context."
        )

    content = path.read_text(encoding="utf-8").strip()
    if not content:
        raise ValueError(f"Listing context file '{path}' is empty.")
    return content


def encode_image_to_base64(path: Path) -> str:
    # Ollama vision models can fail on some source formats like WebP.
    # Normalize every input image to PNG bytes before sending it.
    with Image.open(path) as image:
        if image.mode not in {"RGB", "RGBA"}:
            image = image.convert("RGB")

        buffer = BytesIO()
        image.save(buffer, format="PNG")

    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def extract_json(text: str) -> dict[str, Any]:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            raise ValueError(f"Prompt model did not return JSON. Raw response:\n{text}")
        return json.loads(match.group(0))


def build_prompt_request(
    image_path: Path,
    shot: ShotSpec,
    listing_title: str,
    listing_url: str,
    listing_context: str,
    prompt_model: str,
) -> dict[str, str]:
    system_prompt = """
You are a real-estate video prompt writer.

Return exactly one JSON object with these keys:
- prompt
- negative_prompt
- focus_notes

Your job:
- inspect the input listing photo
- preserve the original architecture and layout
- write a prompt for a 5-second vertical 9:16 image-to-video generation
- follow the requested camera move
- keep the motion subtle, premium, and believable
- never ask the model to add fictional windows, doors, cabinets, fixtures, rooms, or decor
- keep the scene photorealistic and faithful to the source image

The positive prompt should mention:
- the requested camera move
- vertical framing suitable for short-form real-estate video
- natural light and premium interior cinematography when supported by the image
- preservation of walls, ceilings, floors, windows, and built-ins

The negative prompt should strongly discourage:
- hallucinated architecture
- warped geometry
- extra furniture
- duplicate objects
- text overlays
- flicker
- morphing
- people
- watermarks

Keep the prompt concise but production-ready.
""".strip()

    user_prompt = f"""
Listing title: {listing_title}
Listing URL: {listing_url}

Listing context:
{listing_context}

Requested camera move: {shot.camera_move}
Movement guidance: {shot.prompt_hint}

Write the JSON now for this single shot.
""".strip()

    payload = {
        "model": prompt_model,
        "stream": False,
        "format": "json",
        "messages": [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": user_prompt,
                "images": [encode_image_to_base64(image_path)],
            },
        ],
        "options": {"temperature": 0.2},
    }

    ollama_url = os.getenv("OLLAMA_URL", DEFAULT_OLLAMA_URL)
    response = requests.post(ollama_url, json=payload, timeout=600)
    response.raise_for_status()
    content = response.json()["message"]["content"]
    prompt_data = extract_json(content)

    return {
        "prompt": prompt_data["prompt"].strip(),
        "negative_prompt": prompt_data.get("negative_prompt", "").strip(),
        "focus_notes": prompt_data.get("focus_notes", "").strip(),
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def ease_in_out(progress: float) -> float:
    return progress * progress * (3 - 2 * progress)


def lerp(start: float, end: float, progress: float) -> float:
    return start + (end - start) * progress


def crop_dimensions_for_vertical_frame(width: int, height: int) -> tuple[float, float]:
    target_aspect = OUTPUT_WIDTH / OUTPUT_HEIGHT
    image_aspect = width / height

    if image_aspect > target_aspect:
        crop_height = height * 0.94
        crop_width = crop_height * target_aspect
    else:
        crop_width = width * 0.94
        crop_height = crop_width / target_aspect

    return crop_width, crop_height


def shot_motion_profile(shot: ShotSpec) -> dict[str, float]:
    profiles = {
        "orbital_left": {
            "x_start": 0.58,
            "x_end": 0.42,
            "y_start": 0.50,
            "y_end": 0.48,
            "zoom_start": 0.94,
            "zoom_end": 0.86,
        },
        "dolly_in_pan_right": {
            "x_start": 0.46,
            "x_end": 0.60,
            "y_start": 0.52,
            "y_end": 0.48,
            "zoom_start": 1.00,
            "zoom_end": 0.82,
        },
        "truck_right": {
            "x_start": 0.34,
            "x_end": 0.66,
            "y_start": 0.50,
            "y_end": 0.50,
            "zoom_start": 0.92,
            "zoom_end": 0.92,
        },
        "pedestal_up": {
            "x_start": 0.50,
            "x_end": 0.50,
            "y_start": 0.60,
            "y_end": 0.40,
            "zoom_start": 0.92,
            "zoom_end": 0.86,
        },
        "dolly_out_pan_left": {
            "x_start": 0.60,
            "x_end": 0.42,
            "y_start": 0.50,
            "y_end": 0.50,
            "zoom_start": 0.84,
            "zoom_end": 1.00,
        },
    }
    return profiles[shot.slug]


def clamp_center(center: float, crop_size: float, image_size: int) -> float:
    half = crop_size / 2
    return min(max(center, half), image_size - half)


def render_local_fallback_video(image_path: Path, output_path: Path, shot: ShotSpec) -> str:
    image = Image.open(image_path).convert("RGB")
    width, height = image.size
    base_crop_width, base_crop_height = crop_dimensions_for_vertical_frame(width, height)
    motion = shot_motion_profile(shot)

    writer = imageio.get_writer(
        output_path,
        fps=OUTPUT_FPS,
        codec="libx264",
        format="FFMPEG",
        macro_block_size=1,
    )

    try:
        for frame_idx in range(OUTPUT_FRAMES):
            raw_progress = frame_idx / max(OUTPUT_FRAMES - 1, 1)
            progress = ease_in_out(raw_progress)
            pulse = math.sin(progress * math.pi) * 0.01

            zoom = lerp(motion["zoom_start"], motion["zoom_end"], progress) - pulse
            crop_width = min(base_crop_width * zoom, width)
            crop_height = min(base_crop_height * zoom, height)

            center_x = lerp(motion["x_start"], motion["x_end"], progress) * width
            center_y = lerp(motion["y_start"], motion["y_end"], progress) * height
            center_x = clamp_center(center_x, crop_width, width)
            center_y = clamp_center(center_y, crop_height, height)

            left = int(round(center_x - crop_width / 2))
            top = int(round(center_y - crop_height / 2))
            right = int(round(left + crop_width))
            bottom = int(round(top + crop_height))

            frame = image.crop((left, top, right, bottom)).resize(
                (OUTPUT_WIDTH, OUTPUT_HEIGHT),
                Image.Resampling.LANCZOS,
            )
            writer.append_data(np.asarray(frame))
    finally:
        writer.close()

    return f"file://{output_path.resolve()}"


def select_image(images: list[Path], shot: ShotSpec) -> Path:
    return images[(shot.index - 1) % len(images)]


def main() -> None:
    load_dotenv()
    args = parse_args()

    images_dir = Path(args.images_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    images = discover_images(images_dir)
    listing_context = read_listing_context(Path(args.listing_context))

    manifest: list[dict[str, Any]] = []

    for shot in SHOT_SPECS:
        image_path = select_image(images, shot)
        prompt_bundle = build_prompt_request(
            image_path=image_path,
            shot=shot,
            listing_title=args.listing_title,
            listing_url=args.listing_url,
            listing_context=listing_context,
            prompt_model=args.prompt_model,
        )

        prompt_file = output_dir / f"{shot.index:02d}_{shot.slug}.json"
        write_json(
            prompt_file,
            {
                "shot": asdict(shot),
                "source_image": str(image_path),
                "listing_title": args.listing_title,
                "listing_url": args.listing_url,
                **prompt_bundle,
            },
        )

        manifest_entry = {
            "shot": asdict(shot),
            "source_image": str(image_path),
            "prompt_file": str(prompt_file),
            **prompt_bundle,
        }

        if args.prompts_only:
            manifest_entry["generation_mode"] = "prompts_only"
            manifest.append(manifest_entry)
            print(f"[ok] Shot {shot.index}: {prompt_file}")
            continue

        video_file = output_dir / f"{shot.index:02d}_{shot.slug}.mp4"
        video_url = render_local_fallback_video(
            image_path=image_path,
            output_path=video_file,
            shot=shot,
        )
        manifest_entry["generation_mode"] = "local_fallback_video"
        manifest_entry["video_file"] = str(video_file)
        manifest_entry["video_url"] = video_url
        manifest.append(manifest_entry)
        print(f"[ok] Shot {shot.index}: {video_file}")

    write_json(output_dir / "submission_manifest.json", {"shots": manifest})
    if args.prompts_only:
        print(f"[done] Wrote {len(manifest)} prompt files to {output_dir}")
    else:
        print(f"[done] Wrote {len(manifest)} local fallback videos to {output_dir}")


if __name__ == "__main__":
    main()
