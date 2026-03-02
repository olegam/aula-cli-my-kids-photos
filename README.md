# aula-cli-my-kids-photos

Sync gallery images from Aula and keep only photos that contain your own kids using local face recognition.

This tool is built on top of [`aula-cli`](https://github.com/olegam/aula-cli): `aula-cli` handles Aula authentication and API access, while this tool uses local facial recognition to keep photos that are likely to contain your kids based on your reference photos.

## Typical use case

Use this when you want a local folder of relevant Aula gallery photos without manually reviewing everything:

1. Login once with `aula-cli login`.
2. Add reference photos per child in `data/reference/<child-name>/`.
3. Run `aula-cli-my-kids-photos sync --latest-albums=10` regularly to fetch recent media and keep only matching photos.

## Facial recognition stack

Face matching is done with:

- [`insightface`](https://github.com/deepinsight/insightface) (Python face analysis library)
- [`buffalo_l`](https://github.com/deepinsight/insightface/tree/master/python-package#model-zoo) model pack (face detection + recognition)
- [`onnxruntime`](https://onnxruntime.ai/) (inference runtime used by InsightFace)

Matching is approximate, not exact. Think of results as "probably contains one of my kids" and tune with `--tolerance` if needed.

Provider behavior:

- Apple Silicon prefers `CoreMLExecutionProvider` and falls back to CPU automatically.
- Other hardware tries CUDA/ROCm/OpenVINO/DML when available, then falls back to CPU.
- Override provider order with `--insightface-providers=<csv>` when needed.

On first run, InsightFace downloads the `buffalo_l` model pack automatically.

## Quick Start

1. Install deps:

   ```bash
   bun install
   python3 -m pip install -r requirements.txt
   ```

2. Link the CLI command locally:

   ```bash
   bun link
   ```

3. Ensure Aula session is ready:

   ```bash
   aula-cli login
   ```

4. Initialize folders and auto-create child reference folders from your Aula profile:

   ```bash
   aula-cli-my-kids-photos init
   ```

5. Place reference photos:

   - Put 3-10 clear photos per kid in each folder under `data/reference/`.
   - Example: `data/reference/emma/`, `data/reference/oliver/`.
   - Use photos with only one person visible (the target child).
   - Make sure the face is large, sharp, and fully visible (not far away, blurred, or heavily occluded).
   - Include variety across different days, lighting, angles, and clothing.
   - Avoid near-duplicate shots from the same moment.

6. Sync + filter:

   ```bash
   aula-cli-my-kids-photos sync --latest-albums=10
   ```

Everything runs locally on your machine.

## What `init` does

- Creates:
  - `data/reference/`
  - `data/photos/`
  - `data/cache/`
- Calls `aula-cli me` and creates one child folder per kid under `data/reference/`.
- Prints where to place reference photos before running sync.

## What `sync` does

1. Uses `aula-cli` to list albums and media.
2. Downloads only new images (incremental sync).
3. Runs local face recognition against your reference photos.
4. Keeps photos with likely matches and deletes photos that do not match any of your kids.

## CLI usage

```bash
aula-cli-my-kids-photos init
aula-cli-my-kids-photos sync --latest-albums=10
aula-cli-my-kids-photos benchmark --images-dir=data/photos --limit=500
```

Optional flags:

- `--dry-run` (download/scan simulation)
- `--keep-unmatched` (do not delete non-matching photos)
- `--latest-albums=10` (recommended default: only newest albums)
- `--tolerance=0.48` (higher is stricter)
- `--min-face-px=80` (ignore very small faces)
- `--min-votes=2` (minimum matching references before accept)
- `--distance-margin=0.06` (winner vs runner-up confidence gap)
- `--insightface-model=buffalo_l` (model pack override)
- `--insightface-det-size=640,640` (detector input size; lower can be faster)
- `--insightface-providers=CoreMLExecutionProvider,CPUExecutionProvider` (manual provider order)
- `--aula-cli=<command>`
- `--session=<path>`
- `--base-url=<url>`
- `--python=<command>`

To process all albums instead of only recent ones:

```bash
aula-cli-my-kids-photos sync
```

## Data files

- `data/cache/download-state.json`: tracks already processed media
- `data/cache/last-report.json`: latest analysis report
