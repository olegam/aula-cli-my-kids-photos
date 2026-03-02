#!/usr/bin/env bun

import { mkdir, readFile, writeFile } from "node:fs/promises";
import { basename, dirname, extname, join, resolve } from "node:path";

type JsonObject = Record<string, unknown>;

type RuntimeOptions = {
  aulaCliCommand: string;
  baseUrl: string;
  sessionPath: string;
  pageSize: number;
  maxAlbumPages: number;
  maxMediaPages: number;
  outputDir: string;
  cacheDir: string;
  referenceDir: string;
  pythonBin: string;
  pythonScript: string;
  tolerance: number;
  deleteUnmatched: boolean;
  dryRun: boolean;
};

type DownloadState = {
  processedMediaKeys: string[];
};

type NewDownload = {
  mediaKey: string;
  albumId: number;
  mediaId: string;
  outputPath: string;
  sourceUrl: string;
};

type FaceReport = {
  total: number;
  kept: number;
  removed: number;
  errors: number;
  keptByKid: Record<string, number>;
};

type ChildSummary = {
  name: string;
  shortName?: string;
};

const DEFAULTS: RuntimeOptions = {
  aulaCliCommand: "aula-cli",
  baseUrl: "https://www.aula.dk",
  sessionPath: "~/.aula-cli/latest-storage-state.json",
  pageSize: 50,
  maxAlbumPages: 50,
  maxMediaPages: 250,
  outputDir: "data/photos",
  cacheDir: "data/cache",
  referenceDir: "data/reference",
  pythonBin: "python3",
  pythonScript: "scripts/filter_kids.py",
  tolerance: 0.47,
  deleteUnmatched: true,
  dryRun: false
};

const parseCliFlag = (args: string[], name: string): string | undefined => {
  const prefix = `--${name}=`;
  const raw = args.find((part) => part.startsWith(prefix));
  return raw ? raw.slice(prefix.length) : undefined;
};

const hasFlag = (args: string[], name: string): boolean => args.includes(`--${name}`);

const parseNumberFlag = (args: string[], name: string, fallback: number): number => {
  const raw = parseCliFlag(args, name);
  if (!raw) {
    return fallback;
  }
  const parsed = Number(raw);
  return Number.isFinite(parsed) ? parsed : fallback;
};

const expandHome = (value: string): string => {
  if (!value.startsWith("~/")) {
    return value;
  }
  const home = process.env.HOME;
  if (!home) {
    throw new Error("Could not resolve HOME while expanding a ~/ path.");
  }
  return join(home, value.slice(2));
};

const runtimeOptionsFromArgs = (args: string[]): RuntimeOptions => {
  const aulaCliCommand = parseCliFlag(args, "aula-cli") ?? DEFAULTS.aulaCliCommand;
  const baseUrl = parseCliFlag(args, "base-url") ?? DEFAULTS.baseUrl;
  const sessionPath = parseCliFlag(args, "session") ?? DEFAULTS.sessionPath;
  const pythonBin = parseCliFlag(args, "python") ?? DEFAULTS.pythonBin;
  const tolerance = parseNumberFlag(args, "tolerance", DEFAULTS.tolerance);
  const pageSize = parseNumberFlag(args, "page-size", DEFAULTS.pageSize);
  const dryRun = hasFlag(args, "dry-run");
  const deleteUnmatched = !hasFlag(args, "keep-unmatched");

  return {
    ...DEFAULTS,
    aulaCliCommand,
    baseUrl,
    sessionPath,
    pythonBin,
    tolerance,
    pageSize,
    dryRun,
    deleteUnmatched
  };
};

const runCommand = async (cmd: string[], cwd: string): Promise<string> => {
  const proc = Bun.spawn(cmd, {
    cwd,
    stdout: "pipe",
    stderr: "pipe"
  });
  const stdout = await new Response(proc.stdout).text();
  const stderr = await new Response(proc.stderr).text();
  const exitCode = await proc.exited;
  if (exitCode !== 0) {
    throw new Error(`Command failed (${cmd.join(" ")}): ${stderr || stdout}`);
  }
  return stdout.trim();
};

const readJsonFile = async <T>(path: string): Promise<T> => {
  const raw = await readFile(path, "utf8");
  return JSON.parse(raw) as T;
};

const ensureDir = async (path: string): Promise<void> => {
  await mkdir(path, { recursive: true });
};

const isObject = (value: unknown): value is JsonObject => typeof value === "object" && value !== null && !Array.isArray(value);

const collectObjectArrays = (value: unknown, out: JsonObject[] = []): JsonObject[] => {
  if (Array.isArray(value)) {
    const objectItems = value.filter((item): item is JsonObject => isObject(item));
    if (objectItems.length > 0) {
      out.push(...objectItems);
    }
    for (const item of value) {
      collectObjectArrays(item, out);
    }
    return out;
  }
  if (isObject(value)) {
    for (const nested of Object.values(value)) {
      collectObjectArrays(nested, out);
    }
  }
  return out;
};

const unwrapApiData = (value: unknown): unknown => {
  if (!isObject(value)) {
    return value;
  }
  if ("data" in value) {
    return value.data;
  }
  return value;
};

const extractEntityList = (value: unknown): JsonObject[] => {
  const data = unwrapApiData(value);
  if (Array.isArray(data)) {
    return data.filter((item): item is JsonObject => isObject(item));
  }

  const arrays = collectObjectArrays(data);
  if (arrays.length > 0) {
    return arrays;
  }
  return [];
};

const getNum = (value: unknown): number | undefined => {
  if (typeof value === "number" && Number.isFinite(value)) {
    return value;
  }
  if (typeof value === "string") {
    const parsed = Number.parseInt(value, 10);
    if (Number.isFinite(parsed)) {
      return parsed;
    }
  }
  return undefined;
};

const getStr = (value: unknown): string | undefined => {
  if (typeof value === "string" && value.trim().length > 0) {
    return value;
  }
  return undefined;
};

const walkStringLeaves = (value: unknown, pathPrefix = ""): Array<{ path: string; value: string }> => {
  if (typeof value === "string") {
    return [{ path: pathPrefix, value }];
  }
  if (Array.isArray(value)) {
    return value.flatMap((item, index) => walkStringLeaves(item, `${pathPrefix}[${index}]`));
  }
  if (isObject(value)) {
    return Object.entries(value).flatMap(([key, nested]) => {
      const nextPath = pathPrefix ? `${pathPrefix}.${key}` : key;
      return walkStringLeaves(nested, nextPath);
    });
  }
  return [];
};

const looksLikeImageUrl = (candidate: string): boolean => {
  const lower = candidate.toLowerCase();
  return (
    lower.startsWith("http://") ||
    lower.startsWith("https://") ||
    lower.startsWith("/") ||
    lower.includes(".jpg") ||
    lower.includes(".jpeg") ||
    lower.includes(".png") ||
    lower.includes(".webp") ||
    lower.includes(".heic")
  );
};

const scoreUrlCandidate = (path: string, value: string): number => {
  const p = path.toLowerCase();
  const v = value.toLowerCase();
  let score = 0;
  if (p.includes("download")) score += 60;
  if (p.includes("original")) score += 40;
  if (p.includes("image") || p.includes("media")) score += 25;
  if (p.endsWith("url") || p.includes(".url")) score += 20;
  if (v.startsWith("https://") || v.startsWith("http://")) score += 20;
  if (v.includes(".jpg") || v.includes(".jpeg") || v.includes(".png") || v.includes(".webp") || v.includes(".heic")) score += 15;
  if (v.includes("/api/") || v.includes("/gallery/")) score += 10;
  if (v.includes("thumb")) score -= 30;
  return score;
};

const pickMediaUrl = (media: JsonObject): string | undefined => {
  const allStrings = walkStringLeaves(media)
    .filter((entry) => looksLikeImageUrl(entry.value))
    .sort((a, b) => scoreUrlCandidate(b.path, b.value) - scoreUrlCandidate(a.path, a.value));
  return allStrings[0]?.value;
};

const toAbsoluteUrl = (baseUrl: string, value: string): string => {
  if (value.startsWith("http://") || value.startsWith("https://")) {
    return value;
  }
  return new URL(value, baseUrl).toString();
};

const imageExtFromUrl = (url: string): string => {
  const clean = url.split("?")[0] ?? "";
  const ext = extname(clean).toLowerCase();
  if (ext === ".jpg" || ext === ".jpeg" || ext === ".png" || ext === ".webp" || ext === ".heic") {
    return ext;
  }
  return ".jpg";
};

const cookieHeaderFromSession = async (sessionPath: string): Promise<string | undefined> => {
  const json = await readJsonFile<{ cookies?: Array<{ name?: string; value?: string }> }>(sessionPath);
  const cookies = (json.cookies ?? [])
    .map((cookie) => {
      if (!cookie.name || !cookie.value) {
        return undefined;
      }
      return `${cookie.name}=${cookie.value}`;
    })
    .filter((cookie): cookie is string => Boolean(cookie));
  return cookies.length > 0 ? cookies.join("; ") : undefined;
};

const runAulaCliJson = async (options: RuntimeOptions, cwd: string, args: string[]): Promise<unknown> => {
  const cliArgs = [options.aulaCliCommand, ...args, "--output=json"];
  const stdout = await runCommand(cliArgs, cwd);
  try {
    return JSON.parse(stdout);
  } catch (error) {
    throw new Error(`Could not parse JSON output from ${cliArgs.join(" ")}: ${String(error)}`);
  }
};

const loadState = async (statePath: string): Promise<DownloadState> => {
  try {
    return await readJsonFile<DownloadState>(statePath);
  } catch {
    return { processedMediaKeys: [] };
  }
};

const saveState = async (statePath: string, state: DownloadState): Promise<void> => {
  await ensureDir(dirname(statePath));
  await writeFile(statePath, JSON.stringify(state, null, 2), "utf8");
};

const listAlbums = async (options: RuntimeOptions, cwd: string): Promise<number[]> => {
  const ids = new Set<number>();
  for (let page = 0; page < options.maxAlbumPages; page++) {
    const payload = await runAulaCliJson(options, cwd, ["gallery", "albums", `--index=${page}`, `--limit=${options.pageSize}`]);
    const albums = extractEntityList(payload);
    if (albums.length === 0) {
      break;
    }

    for (const album of albums) {
      const id = getNum(album.id);
      if (id) {
        ids.add(id);
      }
    }

    if (albums.length < options.pageSize) {
      break;
    }
  }
  return [...ids];
};

const listMediaForAlbum = async (options: RuntimeOptions, cwd: string, albumId: number): Promise<JsonObject[]> => {
  const media: JsonObject[] = [];
  for (let page = 0; page < options.maxMediaPages; page++) {
    const args = ["gallery", "media", `--album-id=${albumId}`, `--index=${page}`, `--limit=${options.pageSize}`];
    const payload = await runAulaCliJson(options, cwd, args);
    const items = extractEntityList(payload);
    if (items.length === 0) {
      break;
    }
    media.push(...items);
    if (items.length < options.pageSize) {
      break;
    }
  }
  return media;
};

const downloadImage = async (url: string, cookieHeader: string | undefined): Promise<ArrayBuffer> => {
  const response = await fetch(url, {
    headers: cookieHeader
      ? {
          Cookie: cookieHeader
        }
      : undefined
  });
  if (!response.ok) {
    throw new Error(`Image download failed (${response.status} ${response.statusText}) for ${url}`);
  }
  return response.arrayBuffer();
};

const sanitizeFileChunk = (value: string): string => value.replace(/[^a-zA-Z0-9_-]+/g, "_").slice(0, 60);

const buildMediaKey = (albumId: number, media: JsonObject, imageUrl: string): string => {
  const id = getStr(media.id) ?? getNum(media.id)?.toString() ?? getStr(media.mediaId) ?? "no-id";
  const uploaded = getStr(media.uploadedAt) ?? getStr(media.createdAt) ?? "";
  return `${albumId}:${id}:${uploaded}:${imageUrl}`;
};

const syncNewImages = async (cwd: string, options: RuntimeOptions): Promise<NewDownload[]> => {
  const sessionPath = expandHome(options.sessionPath);
  const cookieHeader = await cookieHeaderFromSession(sessionPath);
  const outputDir = resolve(cwd, options.outputDir);
  const cacheDir = resolve(cwd, options.cacheDir);
  const statePath = join(cacheDir, "download-state.json");
  const state = await loadState(statePath);
  const processedSet = new Set(state.processedMediaKeys);
  const albums = await listAlbums(options, cwd);

  await ensureDir(outputDir);
  await ensureDir(cacheDir);

  const newDownloads: NewDownload[] = [];
  for (const albumId of albums) {
    const mediaItems = await listMediaForAlbum(options, cwd, albumId);
    for (const media of mediaItems) {
      const mediaUrlCandidate = pickMediaUrl(media);
      if (!mediaUrlCandidate) {
        continue;
      }
      const sourceUrl = toAbsoluteUrl(options.baseUrl, mediaUrlCandidate);
      const mediaKey = buildMediaKey(albumId, media, sourceUrl);
      if (processedSet.has(mediaKey)) {
        continue;
      }

      const mediaId = getStr(media.id) ?? getNum(media.id)?.toString() ?? `${newDownloads.length}`;
      const title = sanitizeFileChunk(getStr(media.title) ?? getStr(media.name) ?? "image");
      const ext = imageExtFromUrl(sourceUrl);
      const fileName = `${albumId}_${mediaId}_${title}${ext}`;
      const outputPath = join(outputDir, fileName);

      if (!options.dryRun) {
        const bytes = await downloadImage(sourceUrl, cookieHeader);
        await Bun.write(outputPath, new Uint8Array(bytes));
      }
      processedSet.add(mediaKey);
      newDownloads.push({
        mediaKey,
        albumId,
        mediaId,
        outputPath,
        sourceUrl
      });
    }
  }

  state.processedMediaKeys = [...processedSet];
  if (!options.dryRun) {
    await saveState(statePath, state);
  }
  return newDownloads;
};

const runPythonFilter = async (cwd: string, options: RuntimeOptions, downloads: NewDownload[]): Promise<FaceReport | undefined> => {
  if (downloads.length === 0) {
    return undefined;
  }
  const cacheDir = resolve(cwd, options.cacheDir);
  const manifestPath = join(cacheDir, "new-downloads.json");
  const reportPath = join(cacheDir, "last-report.json");
  await writeFile(manifestPath, JSON.stringify(downloads, null, 2), "utf8");

  const cmd = [
    options.pythonBin,
    resolve(cwd, options.pythonScript),
    `--manifest=${manifestPath}`,
    `--reference-dir=${resolve(cwd, options.referenceDir)}`,
    `--report-path=${reportPath}`,
    `--tolerance=${options.tolerance.toString()}`
  ];
  if (options.deleteUnmatched) {
    cmd.push("--delete-unmatched");
  }

  await runCommand(cmd, cwd);
  return readJsonFile<FaceReport>(reportPath);
};

const slugifyFolderName = (value: string): string => {
  const base = value
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "");
  return base.length > 0 ? base : "child";
};

const extractFirstName = (fullName: string): string => {
  const first = fullName.trim().split(/\s+/)[0];
  return first ?? fullName.trim();
};

const getUniqueFolderName = (base: string, existing: Set<string>): string => {
  if (!existing.has(base)) {
    existing.add(base);
    return base;
  }
  let suffix = 2;
  while (existing.has(`${base}-${suffix}`)) {
    suffix += 1;
  }
  const unique = `${base}-${suffix}`;
  existing.add(unique);
  return unique;
};

const extractChildren = (payload: unknown): ChildSummary[] => {
  if (!isObject(payload)) {
    return [];
  }
  const rawChildren = payload.children;
  if (!Array.isArray(rawChildren)) {
    return [];
  }
  const children: ChildSummary[] = [];
  for (const child of rawChildren) {
    if (!isObject(child)) {
      continue;
    }
    const name = getStr(child.shortName) ?? getStr(child.name);
    if (!name) {
      continue;
    }
    children.push({
      name: getStr(child.name) ?? name,
      shortName: getStr(child.shortName)
    });
  }
  return children;
};

const runInit = async (cwd: string, options: RuntimeOptions): Promise<void> => {
  const referenceDir = resolve(cwd, options.referenceDir);
  const photosDir = resolve(cwd, options.outputDir);
  const cacheDir = resolve(cwd, options.cacheDir);

  await Promise.all([ensureDir(referenceDir), ensureDir(photosDir), ensureDir(cacheDir)]);

  const mePayload = await runAulaCliJson(options, cwd, ["me"]);
  const children = extractChildren(mePayload);

  if (children.length === 0) {
    console.log("Initialized folders but could not detect children from `aula-cli me`.");
    console.log(`Add child folders manually under: ${referenceDir}`);
    return;
  }

  const created: string[] = [];
  const usedFolderNames = new Set<string>();
  for (const child of children) {
    const display = child.name ? extractFirstName(child.name) : child.shortName ?? "child";
    const baseFolder = slugifyFolderName(display);
    const folder = getUniqueFolderName(baseFolder, usedFolderNames);
    const fullPath = join(referenceDir, folder);
    await ensureDir(fullPath);
    created.push(fullPath);
  }

  console.log("Initialization complete.");
  console.log("Created kid reference folders:");
  for (const path of created) {
    console.log(`- ${basename(path)}`);
  }
  console.log("");
  console.log("Next step:");
  console.log(`Place 3-10 clear face photos per kid in ${referenceDir}/<kid-folder>/`);
  console.log("Then run: aula-cli-my-kids-photos sync");
};

const runSync = async (cwd: string, options: RuntimeOptions): Promise<void> => {
  await Promise.all([ensureDir(resolve(cwd, options.referenceDir)), ensureDir(resolve(cwd, options.outputDir)), ensureDir(resolve(cwd, options.cacheDir))]);
  const downloads = await syncNewImages(cwd, options);
  const report = await runPythonFilter(cwd, options, downloads);

  console.log(`Downloaded new images: ${downloads.length}`);
  if (!report) {
    console.log("No new images to analyze.");
    return;
  }
  console.log(`Kept images with kid matches: ${report.kept}`);
  console.log(`Removed non-matching images: ${report.removed}`);
  console.log(`Face-analysis errors: ${report.errors}`);
  if (Object.keys(report.keptByKid).length > 0) {
    console.log("Matches by kid:");
    for (const [kid, count] of Object.entries(report.keptByKid)) {
      console.log(`  ${kid}: ${count}`);
    }
  }
};

const printHelp = (): void => {
  console.log("Usage:");
  console.log("  aula-cli-my-kids-photos init");
  console.log("  aula-cli-my-kids-photos sync [--dry-run] [--keep-unmatched] [--tolerance=0.47]");
  console.log("");
  console.log("Optional flags:");
  console.log("  --aula-cli=<command>   (default: aula-cli)");
  console.log("  --session=<path>       (default: ~/.aula-cli/latest-storage-state.json)");
  console.log("  --base-url=<url>       (default: https://www.aula.dk)");
  console.log("  --python=<command>     (default: python3)");
  console.log("  --page-size=<number>   (default: 50)");
};

const main = async (): Promise<void> => {
  const args = process.argv.slice(2);
  const command = args[0] ?? "sync";
  const options = runtimeOptionsFromArgs(args.slice(1));
  const cwd = process.cwd();

  if (command === "help" || command === "--help" || command === "-h") {
    printHelp();
    return;
  }

  if (command === "init") {
    await runInit(cwd, options);
    return;
  }

  if (command === "sync") {
    await runSync(cwd, options);
    return;
  }

  throw new Error(`Unknown command: ${command}`);
};

await main();
