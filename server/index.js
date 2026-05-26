import http from 'node:http';
import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { config } from './config.js';
import { getDownloadLogs, getDownloadStatus, startDownload, stopDownload } from './downloader.js';
import { inspectRepo } from './hfRepo.js';
import { deleteModel, scanModels } from './models.js';
import { getLogs, getStatus, startServer, stopServer, syncStatus } from './processManager.js';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(__dirname, '..');
const dist = path.join(root, 'dist');

function send(res, code, body, type = 'application/json') {
  res.writeHead(code, { 'content-type': type, 'access-control-allow-origin': '*' });
  res.end(type === 'application/json' ? JSON.stringify(body) : body);
}

async function readBody(req) {
  const chunks = [];
  for await (const chunk of req) chunks.push(chunk);
  const raw = Buffer.concat(chunks).toString('utf8');
  return raw ? JSON.parse(raw) : {};
}

async function serveStatic(req, res) {
  const url = new URL(req.url, `http://${req.headers.host}`);
  const requested = url.pathname === '/' ? '/index.html' : url.pathname;
  const filePath = path.normalize(path.join(dist, requested));
  if (!filePath.startsWith(dist)) return send(res, 403, 'Forbidden', 'text/plain');
  try {
    const data = await fs.readFile(filePath);
    const ext = path.extname(filePath).toLowerCase();
    const type = ext === '.html' ? 'text/html' : ext === '.js' ? 'text/javascript' : ext === '.css' ? 'text/css' : 'application/octet-stream';
    send(res, 200, data, type);
  } catch {
    const index = await fs.readFile(path.join(dist, 'index.html'));
    send(res, 200, index, 'text/html');
  }
}

const server = http.createServer(async (req, res) => {
  try {
    if (req.method === 'OPTIONS') return send(res, 204, {});
    const url = new URL(req.url, `http://${req.headers.host}`);

    if (url.pathname === '/api/config') return send(res, 200, config);
    if (url.pathname === '/api/models') return send(res, 200, { models: await scanModels(config.hfCache) });
    if (url.pathname === '/api/models/delete' && req.method === 'POST') {
      const body = await readBody(req);
      const status = getStatus();
      if (status.running && status.options?.model === body.path) throw new Error('Stop the running server before deleting its loaded model.');
      return send(res, 200, await deleteModel(config.hfCache, body.path));
    }
    if (url.pathname === '/api/status') return send(res, 200, await syncStatus({ port: url.searchParams.get('port'), host: url.searchParams.get('host') }));
    if (url.pathname === '/api/logs') return send(res, 200, { logs: getLogs() });
    if (url.pathname === '/api/download/status') return send(res, 200, getDownloadStatus());
    if (url.pathname === '/api/download/logs') return send(res, 200, { logs: getDownloadLogs() });
    if (url.pathname === '/api/download/start' && req.method === 'POST') return send(res, 200, await startDownload(await readBody(req)));
    if (url.pathname === '/api/download/stop' && req.method === 'POST') return send(res, 200, await stopDownload());
    if (url.pathname === '/api/hf/inspect' && req.method === 'POST') return send(res, 200, await inspectRepo((await readBody(req)).repo));
    if (url.pathname === '/api/start' && req.method === 'POST') return send(res, 200, await startServer(await readBody(req)));
    if (url.pathname === '/api/stop' && req.method === 'POST') return send(res, 200, await stopServer());

    return serveStatic(req, res);
  } catch (error) {
    send(res, 500, { error: error.message || String(error) });
  }
});

server.listen(config.port, '127.0.0.1', () => {
  console.log(`llama B70 Manager listening on http://127.0.0.1:${config.port}`);
});
