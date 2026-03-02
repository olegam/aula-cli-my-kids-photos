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
  insightfaceModel: string | undefined;
  insightfaceDetSize: string | undefined;
  insightfaceProviders: string | undefined;
  tolerance: number;
  minFacePx: number;
  minVotes: number;
  distanceMargin: number;
  deleteUnmatched: boolean;
  latestAlbums: number;
  moveUnmatchedToDir: string | undefined;
  dryRun: boolean;
};

type DownloadState = {
  filteredMediaKeys: string[];
  pendingDownloads: NewDownload[];
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
  entries: Array<{
    path?: string;
    matchedKids?: string[];
  }>;
};

type ChildSummary = {
  name: string;
  shortName?: string;
};

type AlbumSummary = {
  id: number;
  title: string;
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
  insightfaceModel: undefined,
  insightfaceDetSize: undefined,
  insightfaceProviders: undefined,
  tolerance: 0.45,
  minFacePx: 80,
  minVotes: 2,
  distanceMargin: 0.06,
  deleteUnmatched: true,
  latestAlbums: 0,
  moveUnmatchedToDir: undefined,
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
  const insightfaceModel = parseCliFlag(args, "insightface-model") ?? DEFAULTS.insightfaceModel;
  const insightfaceDetSize = parseCliFlag(args, "insightface-det-size") ?? DEFAULTS.insightfaceDetSize;
  const insightfaceProviders = parseCliFlag(args, "insightface-providers") ?? DEFAULTS.insightfaceProviders;
  const tolerance = parseNumberFlag(args, "tolerance", DEFAULTS.tolerance);
  const minFacePx = Math.max(1, parseNumberFlag(args, "min-face-px", DEFAULTS.minFacePx));
  const minVotes = Math.max(1, parseNumberFlag(args, "min-votes", DEFAULTS.minVotes));
  const distanceMargin = Math.max(0, parseNumberFlag(args, "distance-margin", DEFAULTS.distanceMargin));
  const pageSize = parseNumberFlag(args, "page-size", DEFAULTS.pageSize);
  const dryRun = hasFlag(args, "dry-run");
  const deleteUnmatched = !hasFlag(args, "keep-unmatched");
  const latestAlbums = Math.max(0, parseNumberFlag(args, "latest-albums", DEFAULTS.latestAlbums));
  const moveUnmatchedToDir = hasFlag(args, "move-not-detected") ? "not-detected" : undefined;

  return {
    ...DEFAULTS,
    aulaCliCommand,
    baseUrl,
    sessionPath,
    pythonBin,
    insightfaceModel,
    insightfaceDetSize,
    insightfaceProviders,
    tolerance,
    minFacePx,
    minVotes,
    distanceMargin,
    pageSize,
    dryRun,
    deleteUnmatched,
    latestAlbums,
    moveUnmatchedToDir
  };
};

const runCommand = async (cmd: string[], cwd: string, env?: Record<string, string>): Promise<string> => {
  const proc = Bun.spawn(cmd, {
    cwd,
    env: env ? { ...process.env, ...env } : process.env,
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

const insightfaceEnv = (options: RuntimeOptions): Record<string, string> => {
  const env: Record<string, string> = {};
  if (typeof options.insightfaceModel === "string" && options.insightfaceModel.trim().length > 0) {
    env.INSIGHTFACE_MODEL = options.insightfaceModel.trim();
  }
  if (typeof options.insightfaceDetSize === "string" && options.insightfaceDetSize.trim().length > 0) {
    env.INSIGHTFACE_DET_SIZE = options.insightfaceDetSize.trim();
  }
  if (typeof options.insightfaceProviders === "string" && options.insightfaceProviders.trim().length > 0) {
    env.INSIGHTFACE_PROVIDERS = options.insightfaceProviders.trim();
  }
  return env;
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
  if (isObject(data)) {
    const results = data.results;
    if (Array.isArray(results)) {
      return results.filter((item): item is JsonObject => isObject(item));
    }
  }
  if (Array.isArray(data)) {
    return data.filter((item): item is JsonObject => isObject(item));
  }

  const arrays = collectObjectArrays(data);
  if (arrays.length > 0) {
    return arrays;
  }
  return [];
};

const extractPaginatedResults = (value: unknown): { items: JsonObject[]; totalSize?: number } => {
  const items = extractEntityList(value);
  const data = unwrapApiData(value);
  if (!isObject(data)) {
    return { items };
  }
  const totalSize = getNum(data.totalSize) ?? getNum(data.mediaCount) ?? getNum(data.count);
  return { items, totalSize };
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
    (lower.startsWith("/") && (lower.includes(".jpg") || lower.includes(".jpeg") || lower.includes(".png") || lower.includes(".webp") || lower.includes(".heic")))
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
  if (v.includes("media-prod.aula.dk")) score += 80;
  if (v.startsWith("https://") || v.startsWith("http://")) score += 20;
  if (v.includes(".jpg") || v.includes(".jpeg") || v.includes(".png") || v.includes(".webp") || v.includes(".heic")) score += 15;
  if (v.includes("/api/") || v.includes("/gallery/")) score += 10;
  if (v.includes("thumb")) score -= 30;
  if (!v.startsWith("http://") && !v.startsWith("https://") && !v.startsWith("/")) score -= 100;
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
    const raw = await readJsonFile<{
      filteredMediaKeys?: unknown;
      pendingDownloads?: unknown;
      processedMediaKeys?: unknown;
    }>(statePath);
    const filteredMediaKeys = Array.isArray(raw.filteredMediaKeys)
      ? raw.filteredMediaKeys.filter((key): key is string => typeof key === "string")
      : Array.isArray(raw.processedMediaKeys)
        ? raw.processedMediaKeys.filter((key): key is string => typeof key === "string")
        : [];
    const pendingDownloads = Array.isArray(raw.pendingDownloads)
      ? raw.pendingDownloads.filter((entry): entry is NewDownload => isObject(entry) && typeof entry.mediaKey === "string")
      : [];
    return {
      filteredMediaKeys,
      pendingDownloads
    };
  } catch {
    return { filteredMediaKeys: [], pendingDownloads: [] };
  }
};

const saveState = async (statePath: string, state: DownloadState): Promise<void> => {
  await ensureDir(dirname(statePath));
  await writeFile(statePath, JSON.stringify(state, null, 2), "utf8");
};

const listAlbums = async (options: RuntimeOptions, cwd: string): Promise<AlbumSummary[]> => {
  const albumsById = new Map<number, string>();
  const maxOffset = options.maxAlbumPages * options.pageSize;
  for (let index = 0; index < maxOffset; index += options.pageSize) {
    const payload = await runAulaCliJson(options, cwd, ["gallery", "albums", `--index=${index}`, `--limit=${options.pageSize}`]);
    const { items: albums, totalSize } = extractPaginatedResults(payload);
    if (albums.length === 0) {
      break;
    }

    for (const album of albums) {
      const id = getNum(album.id);
      if (id) {
        const title = getStr(album.title) ?? `album-${id}`;
        if (!albumsById.has(id)) {
          albumsById.set(id, title);
        }
      }
    }

    if (albums.length < options.pageSize) {
      break;
    }
    if (typeof totalSize === "number" && index + options.pageSize >= totalSize) {
      break;
    }
  }
  const albums = [...albumsById.entries()].map(([id, title]) => ({ id, title }));
  if (options.latestAlbums > 0) {
    return albums.slice(0, options.latestAlbums);
  }
  return albums;
};

const listMediaForAlbum = async (options: RuntimeOptions, cwd: string, albumId: number): Promise<JsonObject[]> => {
  const media: JsonObject[] = [];
  const maxOffset = options.maxMediaPages * options.pageSize;
  for (let index = 0; index < maxOffset; index += options.pageSize) {
    const args = ["gallery", "media", `--album-id=${albumId}`, `--index=${index}`, `--limit=${options.pageSize}`];
    const payload = await runAulaCliJson(options, cwd, args);
    const { items, totalSize } = extractPaginatedResults(payload);
    if (items.length === 0) {
      break;
    }
    media.push(...items);
    if (items.length < options.pageSize) {
      break;
    }
    if (typeof totalSize === "number" && index + options.pageSize >= totalSize) {
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

const syncNewImages = async (cwd: string, options: RuntimeOptions, state: DownloadState, statePath: string): Promise<NewDownload[]> => {
  const sessionPath = expandHome(options.sessionPath);
  const cookieHeader = await cookieHeaderFromSession(sessionPath);
  const outputDir = resolve(cwd, options.outputDir);
  const cacheDir = resolve(cwd, options.cacheDir);
  const pendingDir = join(cacheDir, "pending-images");
  const filteredSet = new Set(state.filteredMediaKeys);
  const pendingSet = new Set(state.pendingDownloads.map((entry) => entry.mediaKey));
  const albums = await listAlbums(options, cwd);

  await ensureDir(outputDir);
  await ensureDir(cacheDir);
  await ensureDir(pendingDir);

  const newDownloads: NewDownload[] = [];
  let sinceLastSave = 0;
  let downloadedTotal = 0;
  let skippedKnown = 0;
  let skippedNoUrl = 0;
  let failedDownloads = 0;
  console.log(`Found ${albums.length} albums to scan.`);

  for (let albumIndex = 0; albumIndex < albums.length; albumIndex += 1) {
    const album = albums[albumIndex] as AlbumSummary;
    console.log(`[${albumIndex + 1}/${albums.length}] Scanning album "${album.title}" (#${album.id})...`);
    const mediaItems = await listMediaForAlbum(options, cwd, album.id);
    console.log(`Found ${mediaItems.length} media items in "${album.title}".`);
    let albumDownloads = 0;
    for (const media of mediaItems) {
      const mediaUrlCandidate = pickMediaUrl(media);
      if (!mediaUrlCandidate) {
        skippedNoUrl += 1;
        continue;
      }
      const sourceUrl = toAbsoluteUrl(options.baseUrl, mediaUrlCandidate);
      const mediaKey = buildMediaKey(album.id, media, sourceUrl);
      if (filteredSet.has(mediaKey) || pendingSet.has(mediaKey)) {
        skippedKnown += 1;
        continue;
      }

      const mediaId = getStr(media.id) ?? getNum(media.id)?.toString() ?? `${newDownloads.length}`;
      const title = sanitizeFileChunk(getStr(media.title) ?? getStr(media.name) ?? "image");
      const ext = imageExtFromUrl(sourceUrl);
      const fileName = `${album.id}_${mediaId}_${title}${ext}`;
      const outputPath = join(pendingDir, fileName);

      if (!options.dryRun) {
        try {
          const bytes = await downloadImage(sourceUrl, cookieHeader);
          await Bun.write(outputPath, new Uint8Array(bytes));
        } catch (error) {
          failedDownloads += 1;
          console.warn(`Skipping media ${mediaId}: ${String(error)}`);
          continue;
        }
      }
      const downloadEntry: NewDownload = {
        mediaKey,
        albumId: album.id,
        mediaId,
        outputPath,
        sourceUrl
      };
      if (!options.dryRun) {
        state.pendingDownloads.push(downloadEntry);
        pendingSet.add(mediaKey);
      }
      sinceLastSave += 1;
      newDownloads.push(downloadEntry);
      albumDownloads += 1;
      downloadedTotal += 1;
      if (downloadedTotal % 25 === 0) {
        console.log(`Downloaded ${downloadedTotal} new images so far...`);
      }
      if (!options.dryRun && sinceLastSave >= 25) {
        await saveState(statePath, state);
        sinceLastSave = 0;
      }
    }
    console.log(`Finished album "${album.title}": downloaded ${albumDownloads} new images.`);
  }

  if (!options.dryRun && sinceLastSave > 0) {
    await saveState(statePath, state);
  }
  console.log(
    `Download phase complete: ${downloadedTotal} new, ${skippedKnown} already-processed, ${skippedNoUrl} no-usable-url, ${failedDownloads} failed downloads.`
  );
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
    `--output-dir=${resolve(cwd, options.outputDir)}`,
    `--report-path=${reportPath}`,
    `--tolerance=${options.tolerance.toString()}`,
    `--min-face-px=${options.minFacePx.toString()}`,
    `--min-votes=${options.minVotes.toString()}`,
    `--distance-margin=${options.distanceMargin.toString()}`
  ];
  if (options.moveUnmatchedToDir) {
    cmd.push(`--unmatched-dir=${resolve(cwd, options.outputDir, options.moveUnmatchedToDir)}`);
  }
  if (options.deleteUnmatched) {
    cmd.push("--delete-unmatched");
  }

  await runCommand(cmd, cwd, insightfaceEnv(options));
  return readJsonFile<FaceReport>(reportPath);
};

const mergeFaceReports = (reports: FaceReport[]): FaceReport | undefined => {
  if (reports.length === 0) {
    return undefined;
  }
  const merged: FaceReport = {
    total: 0,
    kept: 0,
    removed: 0,
    errors: 0,
    keptByKid: {},
    entries: []
  };
  for (const report of reports) {
    merged.total += report.total;
    merged.kept += report.kept;
    merged.removed += report.removed;
    merged.errors += report.errors;
    for (const [kid, count] of Object.entries(report.keptByKid)) {
      merged.keptByKid[kid] = (merged.keptByKid[kid] ?? 0) + count;
    }
    if (Array.isArray(report.entries)) {
      merged.entries.push(...report.entries);
    }
  }
  return merged;
};

const flushPendingDownloads = async (
  cwd: string,
  options: RuntimeOptions,
  state: DownloadState,
  statePath: string
): Promise<FaceReport | undefined> => {
  if (options.dryRun || state.pendingDownloads.length === 0) {
    return undefined;
  }
  console.log(`Running face recognition on ${state.pendingDownloads.length} pending images...`);
  const report = await runPythonFilter(cwd, options, state.pendingDownloads);
  if (!report) {
    return undefined;
  }
  const filteredSet = new Set(state.filteredMediaKeys);
  for (const pending of state.pendingDownloads) {
    filteredSet.add(pending.mediaKey);
  }
  state.filteredMediaKeys = [...filteredSet];
  state.pendingDownloads = [];
  await saveState(statePath, state);
  console.log(`Face recognition complete: kept ${report.kept}, removed ${report.removed}, errors ${report.errors}.`);
  return report;
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
  console.log("Starting sync...");
  const statePath = join(resolve(cwd, options.cacheDir), "download-state.json");
  const state = await loadState(statePath);
  const reports: FaceReport[] = [];

  const resumedPendingCount = state.pendingDownloads.length;
  if (resumedPendingCount > 0) {
    const resumed = await flushPendingDownloads(cwd, options, state, statePath);
    if (resumed) {
      reports.push(resumed);
    }
  }

  const downloads = await syncNewImages(cwd, options, state, statePath);
  const current = await flushPendingDownloads(cwd, options, state, statePath);
  if (current) {
    reports.push(current);
  }
  const report = mergeFaceReports(reports);

  console.log(`Downloaded new images: ${downloads.length}`);
  if (resumedPendingCount > 0) {
    console.log(`Resumed pending analysis items: ${resumedPendingCount}`);
  }
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
  const keptFileNames = Array.from(
    new Set(
      (report.entries ?? [])
        .filter((entry) => Array.isArray(entry.matchedKids) && entry.matchedKids.length > 0 && typeof entry.path === "string")
        .map((entry) => basename(entry.path as string))
    )
  );
  if (keptFileNames.length > 0) {
    console.log("New images with your kids since last sync:");
    for (const fileName of keptFileNames) {
      console.log(`- ${fileName}`);
    }
  }
};

const runTune = async (cwd: string, options: RuntimeOptions): Promise<void> => {
  const positiveDir = parseCliFlag(process.argv.slice(2), "positive-dir") ?? "data/tuning/contains-kids";
  const negativeDir = parseCliFlag(process.argv.slice(2), "negative-dir") ?? "data/tuning/not-kids";
  const iterations = Math.max(1, parseNumberFlag(process.argv.slice(2), "iterations", 10));
  const reportPath = resolve(cwd, options.cacheDir, "tuning-report.json");
  const cmd = [
    options.pythonBin,
    resolve(cwd, "scripts/tune_thresholds.py"),
    `--reference-dir=${resolve(cwd, options.referenceDir)}`,
    `--positive-dir=${resolve(cwd, positiveDir)}`,
    `--negative-dir=${resolve(cwd, negativeDir)}`,
    `--iterations=${iterations.toString()}`,
    `--report-path=${reportPath}`
  ];
  console.log(`Starting threshold tuning with ${iterations} iterations...`);
  await runCommand(cmd, cwd, insightfaceEnv(options));
  const report = await readJsonFile<{
    best?: {
      tolerance: number;
      minFacePx: number;
      minVotes: number;
      distanceMargin: number;
      totalErrors: number;
      accuracy: number;
      precision: number;
      recall: number;
    };
    runs?: Array<unknown>;
  }>(reportPath);
  if (!report.best) {
    console.log("No tuning result found.");
    return;
  }
  console.log(`Best thresholds after ${report.runs?.length ?? iterations} runs:`);
  console.log(
    `tolerance=${report.best.tolerance}, minFacePx=${report.best.minFacePx}, minVotes=${report.best.minVotes}, distanceMargin=${report.best.distanceMargin}`
  );
  console.log(
    `accuracy=${report.best.accuracy.toFixed(4)}, precision=${report.best.precision.toFixed(4)}, recall=${report.best.recall.toFixed(4)}, totalErrors=${report.best.totalErrors}`
  );
  console.log("Use with sync:");
  console.log(
    `aula-cli-my-kids-photos sync --tolerance=${report.best.tolerance} --min-face-px=${report.best.minFacePx} --min-votes=${report.best.minVotes} --distance-margin=${report.best.distanceMargin}`
  );
};

const runBenchmark = async (cwd: string, options: RuntimeOptions): Promise<void> => {
  const imagesDir = parseCliFlag(process.argv.slice(2), "images-dir") ?? "data/photos";
  const limit = Math.max(0, parseNumberFlag(process.argv.slice(2), "limit", 0));
  const reportPath = resolve(cwd, options.cacheDir, "benchmark-report.json");
  const cmd = [
    options.pythonBin,
    resolve(cwd, "scripts/benchmark_filter.py"),
    `--reference-dir=${resolve(cwd, options.referenceDir)}`,
    `--images-dir=${resolve(cwd, imagesDir)}`,
    `--report-path=${reportPath}`,
    `--tolerance=${options.tolerance.toString()}`,
    `--min-face-px=${options.minFacePx.toString()}`,
    `--min-votes=${options.minVotes.toString()}`,
    `--distance-margin=${options.distanceMargin.toString()}`
  ];
  if (limit > 0) {
    cmd.push(`--limit=${limit.toString()}`);
  }
  console.log(`Running face benchmark on ${imagesDir}${limit > 0 ? ` (limit ${limit})` : ""}...`);
  await runCommand(cmd, cwd, insightfaceEnv(options));
  const report = await readJsonFile<{
    totalImages: number;
    processedImages: number;
    provider: string;
    imagesPerSecond: number;
    p50Ms: number;
    p95Ms: number;
    totalTimeSeconds: number;
    matchedImages: number;
    statusCounts: Record<string, number>;
  }>(reportPath);
  console.log(
    `Benchmark complete: provider=${report.provider}, processed=${report.processedImages}/${report.totalImages}, ` +
      `throughput=${report.imagesPerSecond.toFixed(2)} img/s, p50=${report.p50Ms.toFixed(1)}ms, p95=${report.p95Ms.toFixed(1)}ms, ` +
      `matched=${report.matchedImages}, total=${report.totalTimeSeconds.toFixed(2)}s`
  );
  const gatePass = report.processedImages > 0 && report.p95Ms > 0;
  console.log(`Benchmark gate (valid timing sample): ${gatePass ? "PASS" : "FAIL"}`);
};

const printHelp = (): void => {
  console.log("Usage:");
  console.log("  aula-cli-my-kids-photos init");
  console.log("  aula-cli-my-kids-photos sync [--dry-run] [--keep-unmatched] [--tolerance=0.45] [--latest-albums=10]");
  console.log("  aula-cli-my-kids-photos tune [--positive-dir=...] [--negative-dir=...] [--iterations=10]");
  console.log("  aula-cli-my-kids-photos benchmark [--images-dir=...] [--limit=500]");
  console.log("");
  console.log("Optional flags:");
  console.log("  --aula-cli=<command>   (default: aula-cli)");
  console.log("  --session=<path>       (default: ~/.aula-cli/latest-storage-state.json)");
  console.log("  --base-url=<url>       (default: https://www.aula.dk)");
  console.log("  --python=<command>     (default: python3)");
  console.log("  --page-size=<number>   (default: 50)");
  console.log("  --latest-albums=<n>    (limit to newest n albums)");
  console.log("  --move-not-detected    (move unmatched photos to data/photos/not-detected)");
  console.log("  --min-face-px=<n>      (default: 80)");
  console.log("  --min-votes=<n>        (default: 2)");
  console.log("  --distance-margin=<n>  (default: 0.06)");
  console.log("  --insightface-model=<name>      (default: buffalo_l)");
  console.log("  --insightface-det-size=<n|w,h>  (default: 640,640)");
  console.log("  --insightface-providers=<csv>   (override provider order)");
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

  if (command === "tune") {
    await runTune(cwd, options);
    return;
  }

  if (command === "benchmark") {
    await runBenchmark(cwd, options);
    return;
  }

  throw new Error(`Unknown command: ${command}`);
};

await main();
