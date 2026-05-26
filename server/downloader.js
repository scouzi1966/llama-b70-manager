import { spawn } from 'node:child_process';
import fs from 'node:fs';
import { config } from './config.js';

let job = null;
const logs = [];
const maxLogs = 800;

function pushLog(source, text) {
  const lines = String(text).split(/\r?\n/).filter(Boolean);
  for (const line of lines) {
    logs.push({ t: Date.now(), source, line });
    if (logs.length > maxLogs) logs.shift();
  }
}

function hasValue(value) {
  return value !== undefined && value !== null && String(value).trim() !== '';
}

function normalizeRepo(repo) {
  return String(repo || '').trim().replace(/^https:\/\/huggingface\.co\//i, '').replace(/\/tree\/.*$/i, '');
}

function validateRepo(repo) {
  if (!repo || !/^[A-Za-z0-9_.-]+\/[A-Za-z0-9_.-]+$/.test(repo)) {
    throw new Error('Enter a Hugging Face repo id like unsloth/Qwen3.6-27B-MTP-GGUF.');
  }
}

export function getDownloadLogs() {
  return logs;
}

export function getDownloadStatus() {
  if (!job) return { state: 'idle', running: false };
  return {
    state: job.state,
    running: job.state === 'running',
    pid: job.pid,
    startedAt: job.startedAt,
    finishedAt: job.finishedAt,
    exitCode: job.exitCode,
    repo: job.repo,
    include: job.include,
    command: job.command,
  };
}

export async function startDownload(options) {
  if (job?.state === 'running') throw new Error('A Hugging Face download is already running.');
  if (!fs.existsSync(config.huggingFaceCli)) {
    throw new Error(`huggingface-cli.exe not found at ${config.huggingFaceCli}`);
  }

  const repo = normalizeRepo(options.repo);
  validateRepo(repo);

  const include = hasValue(options.include) ? String(options.include).trim() : '*.gguf';
  const args = ['download', repo, '--include', include];
  const command = `"${config.huggingFaceCli}" ${args.map((arg) => `"${arg}"`).join(' ')}`;

  logs.length = 0;
  pushLog('manager', `Downloading repo=${repo}`);
  pushLog('manager', `Include pattern=${include}`);
  pushLog('manager', command);

  const child = spawn(config.huggingFaceCli, args, {
    windowsHide: false,
    stdio: ['ignore', 'pipe', 'pipe'],
  });

  job = {
    child,
    state: 'running',
    pid: child.pid,
    startedAt: Date.now(),
    finishedAt: null,
    exitCode: null,
    repo,
    include,
    command,
  };

  child.stdout.on('data', (data) => pushLog('stdout', data));
  child.stderr.on('data', (data) => pushLog('stderr', data));
  child.on('exit', (code) => {
    pushLog('manager', `huggingface-cli exited with code ${code}`);
    if (job?.child === child) {
      job.state = code === 0 ? 'completed' : 'failed';
      job.finishedAt = Date.now();
      job.exitCode = code;
    }
  });

  return getDownloadStatus();
}

export async function stopDownload() {
  if (!job || job.state !== 'running') return getDownloadStatus();
  const pid = job.child.pid;
  pushLog('manager', `Stopping download process tree pid=${pid}`);
  await new Promise((resolve) => {
    const killer = spawn('taskkill.exe', ['/PID', String(pid), '/T', '/F'], { windowsHide: true });
    killer.on('exit', resolve);
    killer.on('error', resolve);
  });
  job.state = 'stopped';
  job.finishedAt = Date.now();
  job.exitCode = 0;
  return getDownloadStatus();
}
