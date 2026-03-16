
# Real Estate Assessment Pipeline

This project turns listing photos into five assessment-ready, 5-second vertical videos with automated prompt generation and local rendering.

## Stack

- Prompt writing model: local Ollama vision model `llava:latest`
- Video rendering: local pan/zoom camera-motion renderer
- Output format: 5-second `9:16` videos matching the requested camera moves

## What The Script Does

`run_assessment.py` will:

1. Read listing photos from `input_images/`
2. Read listing details from `listing_context.txt`
3. Use `llava` to automatically write one prompt per shot
4. Render a local vertical video using the requested camera move
5. Save five `.mp4` files plus a JSON manifest in `outputs/`

The five shots are:

1. Orbital Left
2. Dolly In + Pan Right
3. Truck Right
4. Pedestal Up
5. Dolly Out + Pan Left

## Important Note About Listing Images

StreetEasy blocks simple bot scraping, so the practical workflow is:

1. Open the listing in a browser:
   `https://streeteasy.com/building/the-printing-house/213?featured=1`
2. Save at least 5 listing photos into `input_images/`
3. Run the pipeline

This still preserves the assessment's core automation requirement on prompt writing because the prompts are generated automatically by a model without manual editing.

## Setup

```bash
cd /home/dhivakar/AI_project
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
```

If you want, copy the sample env file for local Ollama settings:

```bash
cp .env.example .env
```

Make sure Ollama is running and `llava` is available:

```bash
ollama pull llava:latest
ollama serve
```

## Workflow

Use this end-to-end workflow each time you want to generate a submission:

1. Open the listing page in your browser and save at least 5 property images into `input_images/`.
2. Update `listing_context.txt` with the property summary and any constraints you want the prompt writer to respect.
3. Start Ollama and confirm the `llava:latest` model is available.
4. Run a dry pass with `--prompts-only` to verify prompt generation and metadata output.
5. Review the JSON files in `outputs/` to confirm the prompts match the listing and camera moves.
6. Run the full pipeline to render the five vertical `.mp4` videos.
7. Check `outputs/submission_manifest.json` plus the generated videos before packaging the submission.

## Run

Generate prompts and local videos:

```bash
python3 run_assessment.py \
  --images-dir input_images \
  --listing-context listing_context.txt \
  --listing-title "421 Hudson Street #213" \
  --listing-url "https://streeteasy.com/building/the-printing-house/213?featured=1"
```

Generate prompts only, with no paid video API usage:

```bash
python3 run_assessment.py --prompts-only
```

Suggested validation flow:

```bash
python3 run_assessment.py --prompts-only
python3 run_assessment.py
```

## Output

The script writes:

- `outputs/01_orbital_left.mp4`
- `outputs/02_dolly_in_pan_right.mp4`
- `outputs/03_truck_right.mp4`
- `outputs/04_pedestal_up.mp4`
- `outputs/05_dolly_out_pan_left.mp4`
- `outputs/submission_manifest.json`

Each shot also gets a JSON file containing the exact auto-generated prompt and negative prompt used for generation.

In `--prompts-only` mode, the script skips `.mp4` generation and writes:

- `outputs/01_orbital_left.json`
- `outputs/02_dolly_in_pan_right.json`
- `outputs/03_truck_right.json`
- `outputs/04_pedestal_up.json`
- `outputs/05_dolly_out_pan_left.json`
- `outputs/submission_manifest.json`

By default, the script creates local MP4s using smooth pan/zoom camera motion paths derived from the requested shot types. This workflow does not use any hosted video API or API key.

## Assessment Checklist

- Automated prompt writing: yes, via `llava`
- Local zero-key rendering: yes
- Five 5-second videos: yes
- Vertical aspect ratio: yes, `9:16`
- Hallucination prevention: enforced in prompt template and negative prompt

## Original Assessment Brief

SystemStar asks for five, 5-second real-estate videos built from one listing, with these required moves:

1. Orbital
2. Dolly In + Pan
3. Truck Right or Left
4. Your choice
5. Your choice

The generated videos must remain vertical, preserve key architectural features, and avoid hallucinating fixtures like cabinets or windows.