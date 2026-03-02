# aula-cli-my-kids-photos

Sync gallery images from Aula and keep only photos that contain your own kids using local face recognition.

## Quick Start

1. Install deps:

   ```bash
   brew install cmake
   bun install
   python3 -m pip install -r requirements.txt
   ```

   If Python fails building `dlib`, install Apple command line tools and retry:

   ```bash
   xcode-select --install
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

   - Put 3-10 clear, front-facing photos per kid in each folder under `data/reference/`.
   - Example: `data/reference/emma/`, `data/reference/oliver/`.

6. Sync + filter:

   ```bash
   aula-cli-my-kids-photos sync
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
4. Deletes photos that do not match any of your kids.

## CLI usage

```bash
aula-cli-my-kids-photos init
aula-cli-my-kids-photos sync
```

Optional flags:

- `--dry-run` (download/scan simulation)
- `--keep-unmatched` (do not delete non-matching photos)
- `--tolerance=0.47` (lower is stricter)
- `--aula-cli=<command>`
- `--session=<path>`
- `--base-url=<url>`
- `--python=<command>`

## Data files

- `data/cache/download-state.json`: tracks already processed media
- `data/cache/last-report.json`: latest analysis report
