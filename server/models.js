import fs from 'node:fs/promises';
import path from 'node:path';

const quantPattern = /(?:^|[-_])(IQ\d_[A-Z]+|Q\d(?:_[A-Z])?(?:_[A-Z])?|Q\d_K_[MSL]|Q\d_[01]|F16|BF16|F32)(?:\.gguf)?$/i;

function normalizeSlashes(value) {
  return value.replaceAll('\\', '/');
}

function isInside(parent, child) {
  const parentPath = path.resolve(parent).toLowerCase();
  const childPath = path.resolve(child).toLowerCase();
  return childPath === parentPath || childPath.startsWith(`${parentPath}${path.sep}`);
}

function parseRepo(snapshotPath, hfCache) {
  const rel = path.relative(hfCache, snapshotPath).split(path.sep);
  const modelRoot = rel.find((part) => part.startsWith('models--'));
  if (!modelRoot) return 'local';
  return modelRoot.replace(/^models--/, '').replaceAll('--', '/');
}

function parseQuant(fileName) {
  const base = fileName.replace(/\.gguf$/i, '');
  const match = base.match(quantPattern);
  return match ? match[1].toUpperCase() : 'unknown';
}

function displayName(fileName) {
  return fileName.replace(/\.gguf$/i, '');
}

async function walk(dir, out) {
  let entries;
  try {
    entries = await fs.readdir(dir, { withFileTypes: true });
  } catch {
    return;
  }

  for (const entry of entries) {
    const full = path.join(dir, entry.name);
    if (entry.isDirectory()) {
      await walk(full, out);
    } else if (entry.isFile() && entry.name.toLowerCase().endsWith('.gguf')) {
      out.push(full);
    }
  }
}

export async function scanModels(hfCache) {
  const files = [];
  await walk(hfCache, files);
  const models = [];

  for (const file of files) {
    const stat = await fs.stat(file);
    const snapshotPath = file.includes(`${path.sep}snapshots${path.sep}`)
      ? file.slice(0, file.lastIndexOf(`${path.sep}snapshots${path.sep}`))
      : path.dirname(file);

    models.push({
      id: normalizeSlashes(file),
      name: displayName(path.basename(file)),
      fileName: path.basename(file),
      repo: parseRepo(snapshotPath, hfCache),
      quant: parseQuant(path.basename(file)),
      sizeBytes: stat.size,
      sizeGb: +(stat.size / 1024 / 1024 / 1024).toFixed(2),
      path: normalizeSlashes(file),
      modifiedMs: stat.mtimeMs,
    });
  }

  return models.sort((a, b) => {
    const qwen = Number(b.fileName.includes('Qwen_Qwen3.6-27B-Q6_K')) - Number(a.fileName.includes('Qwen_Qwen3.6-27B-Q6_K'));
    if (qwen) return qwen;
    return b.modifiedMs - a.modifiedMs;
  });
}

export async function deleteModel(hfCache, modelPath) {
  if (!modelPath || typeof modelPath !== 'string') throw new Error('Missing model path.');

  const resolvedCache = path.resolve(hfCache);
  const resolvedModel = path.resolve(modelPath);
  if (!isInside(resolvedCache, resolvedModel)) {
    throw new Error('Refusing to delete a file outside the Hugging Face cache.');
  }
  if (!resolvedModel.toLowerCase().endsWith('.gguf')) {
    throw new Error('Refusing to delete anything except a GGUF model file.');
  }

  const stat = await fs.stat(resolvedModel);
  if (!stat.isFile()) throw new Error('Selected model path is not a file.');

  await fs.unlink(resolvedModel);

  let dir = path.dirname(resolvedModel);
  while (isInside(resolvedCache, dir) && path.resolve(dir).toLowerCase() !== resolvedCache.toLowerCase()) {
    try {
      await fs.rmdir(dir);
      dir = path.dirname(dir);
    } catch {
      break;
    }
  }

  return { deleted: normalizeSlashes(resolvedModel) };
}
