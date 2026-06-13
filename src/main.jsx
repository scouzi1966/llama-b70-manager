import React, { useEffect, useMemo, useRef, useState } from 'react';
import { createRoot } from 'react-dom/client';
import { Cpu, Download, FolderSearch, Gauge, Globe, MonitorPlay, Power, RefreshCw, RotateCcw, Search, Square, Terminal, Trash2, Zap } from 'lucide-react';
import './styles.css';

const api = async (path, options = {}) => {
  const res = await fetch(`/api/${path}`, {
    headers: { 'content-type': 'application/json' },
    ...options,
  });
  const data = await res.json();
  if (!res.ok || data.error) throw new Error(data.error || 'Request failed');
  return data;
};

const blankOptions = {
  backend: 'opencl',
  openGui: true,
  enableSysman: true,
  host: '0.0.0.0',
  port: '',
  ctx: '',
  predict: '',
  gpuLayers: '',
  threads: '',
  threadsBatch: '',
  batch: '',
  ubatch: '',
  keep: '',
  flashAttn: '',
  cacheTypeK: '',
  cacheTypeV: '',
  splitMode: '',
  tensorSplit: '',
  mainGpu: '',
  device: '',
  fit: '',
  fitTarget: '',
  fitCtx: '',
  noWebUi: false,
  noKvOffload: false,
  noMmap: false,
  mlock: false,
  noWarmup: false,
  noOpOffload: false,
  cpuMoe: false,
  metrics: false,
  slots: false,
  props: false,
  cachePrompt: false,
  noCachePrompt: false,
  jinja: false,
  noJinja: false,
  offline: false,
  logDisable: false,
  parallel: '',
  timeout: '',
  threadsHttp: '',
  apiPrefix: '',
  alias: '',
  tags: '',
  mediaPath: '',
  modelsDir: '',
  modelsPreset: '',
  modelsMax: '',
  tools: '',
  apiKey: '',
  apiKeyFile: '',
  sslKeyFile: '',
  sslCertFile: '',
  seed: '',
  samplers: '',
  samplerSeq: '',
  temperature: '',
  topK: '',
  topP: '',
  minP: '',
  repeatLastN: '',
  repeatPenalty: '',
  presencePenalty: '',
  frequencyPenalty: '',
  reasoning: '',
  reasoningFormat: '',
  reasoningBudget: '',
  chatTemplate: '',
  chatTemplateFile: '',
  verbosity: '',
  logFile: '',
  extraArgs: '',
};

const optionGroups = [
  {
    title: 'Core',
    fields: [
      ['host', 'Host', '0.0.0.0'], ['port', 'Port', '8080'], ['ctx', 'Context (-c)', 'server default'], ['predict', 'Predict (-n)', 'server default'], ['gpuLayers', 'GPU layers (-ngl)', 'auto/default'], ['threads', 'Threads (-t)', 'auto'], ['threadsBatch', 'Batch threads (-tb)', 'auto'], ['batch', 'Batch (-b)', 'default'], ['ubatch', 'Microbatch (-ub)', 'default'], ['keep', 'Keep', 'default'],
    ],
  },
  {
    title: 'GPU / Memory',
    fields: [
      ['device', 'Device (-dev)', 'blank'], ['splitMode', 'Split mode', 'none/layer/row/tensor'], ['tensorSplit', 'Tensor split', '3,1'], ['mainGpu', 'Main GPU', '0'], ['fit', 'Fit', 'on/off'], ['fitTarget', 'Fit target MiB', '1024'], ['fitCtx', 'Fit ctx', '4096'], ['cacheTypeK', 'KV cache K', 'f16/q8_0/...'], ['cacheTypeV', 'KV cache V', 'f16/q8_0/...'], ['flashAttn', 'Flash attention', 'on/off/auto'],
    ],
  },
  {
    title: 'Server',
    fields: [
      ['parallel', 'Parallel slots (-np)', 'auto'], ['timeout', 'Timeout seconds', '600'], ['threadsHttp', 'HTTP threads', 'auto'], ['apiPrefix', 'API prefix', '/llama'], ['alias', 'Alias', 'model name'], ['tags', 'Tags', 'tag1,tag2'], ['mediaPath', 'Media path', 'folder'], ['modelsDir', 'Models dir', 'folder'], ['modelsPreset', 'Models preset', 'ini path'], ['modelsMax', 'Models max', '4'], ['tools', 'Built-in tools', 'all/read_file'],
    ],
  },
  {
    title: 'Sampling / Chat',
    fields: [
      ['seed', 'Seed', '-1'], ['samplers', 'Samplers', 'default'], ['samplerSeq', 'Sampler seq', 'edskypmxt'], ['temperature', 'Temperature', '0.8'], ['topK', 'Top K', '40'], ['topP', 'Top P', '0.95'], ['minP', 'Min P', '0.05'], ['repeatLastN', 'Repeat last N', '64'], ['repeatPenalty', 'Repeat penalty', '1.0'], ['presencePenalty', 'Presence penalty', '0'], ['frequencyPenalty', 'Frequency penalty', '0'], ['reasoning', 'Reasoning', 'on/off/auto'], ['reasoningFormat', 'Reasoning format', 'none/deepseek'], ['reasoningBudget', 'Reasoning budget', '-1'], ['chatTemplate', 'Chat template', 'template'], ['chatTemplateFile', 'Chat template file', 'path'],
    ],
  },
  {
    title: 'Logging / Security',
    fields: [
      ['verbosity', 'Verbosity (-lv)', '3'], ['logFile', 'Log file', 'path'], ['apiKey', 'API key', 'key'], ['apiKeyFile', 'API key file', 'path'], ['sslKeyFile', 'SSL key file', 'path'], ['sslCertFile', 'SSL cert file', 'path'],
    ],
  },
];

const flags = [
  ['noWebUi', 'API only, no web UI'], ['noKvOffload', 'No KV offload'], ['noMmap', 'No mmap'], ['mlock', 'mlock'], ['noWarmup', 'No warmup'], ['noOpOffload', 'No op offload'], ['cpuMoe', 'CPU MoE'], ['metrics', 'Metrics endpoint'], ['slots', 'Slots endpoint'], ['props', 'Props endpoint'], ['cachePrompt', 'Cache prompt'], ['noCachePrompt', 'No cache prompt'], ['jinja', 'Jinja'], ['noJinja', 'No Jinja'], ['offline', 'Offline'], ['logDisable', 'Disable logs'],
];

function formatTime(ms) {
  return new Date(ms).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
}

function Field({ spec, options, setOptions, disabled }) {
  const [key, label, placeholder] = spec;
  return (
    <label>
      {label}
      <input value={options[key] ?? ''} placeholder={placeholder} onChange={(e) => setOptions({ ...options, [key]: e.target.value })} disabled={disabled} />
    </label>
  );
}

function App() {
  const [models, setModels] = useState([]);
  const [selected, setSelected] = useState('');
  const [status, setStatus] = useState({ state: 'stopped', running: false });
  const [logs, setLogs] = useState([]);
  const [downloadStatus, setDownloadStatus] = useState({ state: 'idle', running: false });
  const [downloadLogs, setDownloadLogs] = useState([]);
  const [downloadRepo, setDownloadRepo] = useState('unsloth/Qwen3.6-27B-MTP-GGUF');
  const [downloadInclude, setDownloadInclude] = useState('*.gguf');
  const [repoInfo, setRepoInfo] = useState(null);
  const [repoLoading, setRepoLoading] = useState(false);
  const [options, setOptions] = useState(blankOptions);
  const [filter, setFilter] = useState('');
  const [optionFilter, setOptionFilter] = useState('');
  const [error, setError] = useState('');
  const [loadingModels, setLoadingModels] = useState(false);
  const repoRequestId = useRef(0);

  async function refreshModels() {
    setLoadingModels(true);
    setError('');
    try {
      const data = await api('models');
      setModels(data.models);
      if (!selected && data.models.length) setSelected(data.models[0].path);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoadingModels(false);
    }
  }

  async function refreshStatus() {
    try {
      const [s, l, ds, dl] = await Promise.all([
        api('status'),
        api('logs'),
        api('download/status'),
        api('download/logs'),
      ]);
      setStatus(s);
      setLogs(l.logs || []);
      setDownloadStatus(ds);
      setDownloadLogs(dl.logs || []);
      if ((ds.state === 'completed' || ds.state === 'failed' || ds.state === 'stopped') && downloadStatus.state === 'running') {
        refreshModels();
      }
    } catch (e) {
      setError(e.message);
    }
  }

  useEffect(() => {
    refreshModels();
    refreshStatus();
    const id = setInterval(refreshStatus, 1500);
    return () => clearInterval(id);
  }, []);

  useEffect(() => {
    if (downloadStatus.state === 'completed') refreshModels();
  }, [downloadStatus.state]);

  useEffect(() => {
    const repo = downloadRepo.trim();
    repoRequestId.current += 1;
    setRepoInfo(null);
    if (!/^[A-Za-z0-9_.-]+\/[A-Za-z0-9_.-]+$/.test(repo) || downloadStatus.running) {
      setRepoLoading(false);
      return undefined;
    }

    const requestId = repoRequestId.current;
    setRepoLoading(true);
    const timer = setTimeout(async () => {
      try {
        const info = await api('hf/inspect', {
          method: 'POST',
          body: JSON.stringify({ repo }),
        });
        if (repoRequestId.current !== requestId) return;
        setRepoInfo(info);
        if (info.files?.length === 1) setDownloadInclude(info.files[0].include);
      } catch (e) {
        if (repoRequestId.current === requestId) setError(e.message);
      } finally {
        if (repoRequestId.current === requestId) setRepoLoading(false);
      }
    }, 700);

    return () => {
      clearTimeout(timer);
      if (repoRequestId.current === requestId) setRepoLoading(false);
    };
  }, [downloadRepo, downloadStatus.running]);

  const filtered = useMemo(() => {
    const term = filter.trim().toLowerCase();
    if (!term) return models;
    return models.filter((m) => `${m.name} ${m.repo} ${m.quant}`.toLowerCase().includes(term));
  }, [models, filter]);

  const visibleGroups = useMemo(() => {
    const term = optionFilter.trim().toLowerCase();
    if (!term) return optionGroups;
    return optionGroups
      .map((group) => ({ ...group, fields: group.fields.filter(([key, label]) => `${key} ${label}`.toLowerCase().includes(term)) }))
      .filter((group) => group.fields.length);
  }, [optionFilter]);

  const visibleFlags = useMemo(() => {
    const term = optionFilter.trim().toLowerCase();
    if (!term) return flags;
    return flags.filter(([key, label]) => `${key} ${label}`.toLowerCase().includes(term));
  }, [optionFilter]);

  const selectedModel = models.find((m) => m.path === selected);
  const localHost = !options.host || ['0.0.0.0', '::', '*'].includes(String(options.host).trim()) ? '127.0.0.1' : options.host;

  async function start() {
    setError('');
    try {
      await api('start', {
        method: 'POST',
        body: JSON.stringify({ ...options, model: selected }),
      });
      await refreshStatus();
    } catch (e) {
      setError(e.message);
    }
  }

  async function stop() {
    setError('');
    try {
      await api('stop', { method: 'POST', body: '{}' });
      await refreshStatus();
    } catch (e) {
      setError(e.message);
    }
  }

  async function startDownload() {
    setError('');
    try {
      await api('download/start', {
        method: 'POST',
        body: JSON.stringify({ repo: downloadRepo, include: downloadInclude }),
      });
      await refreshStatus();
    } catch (e) {
      setError(e.message);
    }
  }

  async function stopDownload() {
    setError('');
    try {
      await api('download/stop', { method: 'POST', body: '{}' });
      await refreshStatus();
      await refreshModels();
    } catch (e) {
      setError(e.message);
    }
  }

  async function deleteSelectedModel(model) {
    if (!model) return;
    const ok = window.confirm(`Delete this model from the Hugging Face cache?\n\n${model.fileName}\n${model.sizeGb} GB\n\nThis removes the local GGUF file only.`);
    if (!ok) return;
    setError('');
    try {
      await api('models/delete', {
        method: 'POST',
        body: JSON.stringify({ path: model.path }),
      });
      const data = await api('models');
      setModels(data.models);
      if (selected === model.path) setSelected(data.models[0]?.path || '');
    } catch (e) {
      setError(e.message);
    }
  }

  async function inspectDownloadRepo() {
    setError('');
    setRepoLoading(true);
    const requestId = repoRequestId.current + 1;
    repoRequestId.current = requestId;
    try {
      const info = await api('hf/inspect', {
        method: 'POST',
        body: JSON.stringify({ repo: downloadRepo }),
      });
      if (repoRequestId.current !== requestId) return;
      setRepoInfo(info);
      if (info.files?.length === 1) setDownloadInclude(info.files[0].include);
    } catch (e) {
      setError(e.message);
    } finally {
      if (repoRequestId.current === requestId) setRepoLoading(false);
    }
  }

  const running = status.running || status.state === 'starting';
  const url = status.url || `http://${localHost}:${options.port || '8080'}`;
  const inferenceUrl = status.apiUrl || `${url}/v1`;
  const lanInferenceUrl = status.lanApiUrl || (String(options.host).trim() === '0.0.0.0' ? `http://<this-pc-ip>:${options.port || '8080'}/v1` : inferenceUrl);

  return (
    <main className="shell">
      <header className="topbar">
        <div>
          <h1>llama B70 Manager</h1>
          <p>Blank option fields are omitted from the llama-server command line.</p>
        </div>
        <div className={`status ${status.state}`}>{status.state}</div>
      </header>

      {error && <div className="error">{error}</div>}

      <section className="grid">
        <aside className="panel models-panel">
          <div className="panel-title">
            <FolderSearch size={18} />
            <span>HF GGUF Models</span>
            <button className="icon-btn" onClick={refreshModels} title="Refresh models"><RefreshCw size={16} /></button>
          </div>
          <div className="download-box">
            <div className="download-title">
              <Download size={16} />
              <span>Download from Hugging Face</span>
              <b className={`mini-status ${downloadStatus.state}`}>{downloadStatus.state}</b>
            </div>
            <label>
              Repo
              <input value={downloadRepo} onChange={(e) => setDownloadRepo(e.target.value)} placeholder="unsloth/Qwen3.6-27B-MTP-GGUF" disabled={downloadStatus.running} />
            </label>
            <label>
              Include
              <input value={downloadInclude} onChange={(e) => setDownloadInclude(e.target.value)} placeholder="*.gguf or *Q6_K*.gguf" disabled={downloadStatus.running} />
            </label>
            <div className="download-actions">
              <button onClick={inspectDownloadRepo} disabled={downloadStatus.running || repoLoading || !downloadRepo.trim()}><Search size={15} /> {repoLoading ? 'Listing...' : 'List files'}</button>
              <button onClick={startDownload} disabled={downloadStatus.running || !downloadRepo.trim()}><Download size={15} /> Download</button>
              <button className="danger" onClick={stopDownload} disabled={!downloadStatus.running}><Square size={15} /> Stop</button>
            </div>
            {repoInfo && (
              <div className="repo-options">
                <div className="repo-summary">
                  <strong>{repoInfo.repo}</strong>
                  <span>{repoInfo.files.length} GGUF options · {repoInfo.collections.length} collections</span>
                </div>
                <div className="repo-chips">
                  <button onClick={() => setDownloadInclude('*.gguf')} disabled={downloadStatus.running}>All GGUFs</button>
                  {repoInfo.collections.map((collection) => (
                    <button key={collection.family} onClick={() => setDownloadInclude(`${collection.family}*.gguf`)} disabled={downloadStatus.running} title={collection.quants.join(', ')}>
                      {collection.family} <span>{collection.files.length}</span>
                    </button>
                  ))}
                </div>
                <div className="repo-file-list">
                  {repoInfo.files.map((file) => (
                    <button key={file.file} className={downloadInclude === file.include ? 'repo-file selected' : 'repo-file'} onClick={() => setDownloadInclude(file.include)} disabled={downloadStatus.running}>
                      <span>{file.name}</span>
                      <b>{file.quant}</b>
                      <em>{file.sizeGb ? `${file.sizeGb} GB` : 'size n/a'}</em>
                    </button>
                  ))}
                </div>
                {repoInfo.tags?.length > 0 && (
                  <div className="repo-tags">{repoInfo.tags.slice(0, 10).map((tag) => <span key={tag}>{tag}</span>)}</div>
                )}
              </div>
            )}
            <div className="download-log">
              {downloadLogs.length === 0 && <span className="muted">No download yet.</span>}
              {downloadLogs.slice(-8).map((log, idx) => (
                <div key={idx} className={`download-line ${log.source}`}>{log.line}</div>
              ))}
            </div>
          </div>
          <input className="search" value={filter} onChange={(e) => setFilter(e.target.value)} placeholder="Filter models" />
          <div className="model-list">
            {loadingModels && <div className="muted pad">Scanning cache...</div>}
            {filtered.map((m) => (
              <div key={m.path} className={`model-row ${selected === m.path ? 'selected' : ''}`} onClick={() => setSelected(m.path)} role="button" tabIndex={0} onKeyDown={(e) => { if (e.key === 'Enter') setSelected(m.path); }}>
                <div className="model-text">
                  <span className="model-name">{m.name}</span>
                  <span className="model-meta">{m.repo} · {m.quant} · {m.sizeGb} GB</span>
                </div>
                <button className="delete-model" onClick={(e) => { e.stopPropagation(); deleteSelectedModel(m); }} disabled={running && selected === m.path} title="Delete local model file">
                  <Trash2 size={15} />
                </button>
              </div>
            ))}
          </div>
        </aside>

        <section className="panel controls-panel">
          <div className="panel-title"><Gauge size={18} /><span>Launch Options</span></div>

          <div className="selected-model">
            <span className="label">Selected model</span>
            <strong>{selectedModel?.fileName || 'No model selected'}</strong>
            <code>{selectedModel?.path || ''}</code>
          </div>

          <div className="quick-row">
            <label>
              Backend
              <select value={options.backend} onChange={(e) => setOptions({ ...options, backend: e.target.value })} disabled={running}>
                <option value="opencl">OpenCL GPU</option>
                <option value="levelzero">Level Zero</option>
                <option value="cpu">CPU fallback</option>
              </select>
            </label>
            <label className="checkbox-label"><input type="checkbox" checked={options.openGui} onChange={(e) => setOptions({ ...options, openGui: e.target.checked })} disabled={running} /> Open GUI after start</label>
            <label className="checkbox-label"><input type="checkbox" checked={options.enableSysman} onChange={(e) => setOptions({ ...options, enableSysman: e.target.checked })} disabled={running} /> ZES sysman</label>
            <button onClick={() => setOptions(blankOptions)} disabled={running}><RotateCcw size={16} /> Reset blanks</button>
          </div>

          <div className="option-search"><Search size={15} /><input value={optionFilter} onChange={(e) => setOptionFilter(e.target.value)} placeholder="Search llama-server options" /></div>

          <div className="options-scroll">
            {visibleGroups.map((group) => (
              <details key={group.title} open={group.title === 'Core'} className="option-group">
                <summary>{group.title}</summary>
                <div className="field-grid">
                  {group.fields.map((field) => <Field key={field[0]} spec={field} options={options} setOptions={setOptions} disabled={running} />)}
                </div>
              </details>
            ))}

            <details open={Boolean(optionFilter)} className="option-group">
              <summary>Flags</summary>
              <div className="flag-grid">
                {visibleFlags.map(([key, label]) => (
                  <label key={key} className="checkbox-label"><input type="checkbox" checked={Boolean(options[key])} onChange={(e) => setOptions({ ...options, [key]: e.target.checked })} disabled={running} /> {label}</label>
                ))}
              </div>
            </details>

            <details className="option-group">
              <summary>Raw extra arguments</summary>
              <textarea value={options.extraArgs} onChange={(e) => setOptions({ ...options, extraArgs: e.target.value })} disabled={running} placeholder='Example: --flash-attn on --cache-type-k q8_0' />
            </details>
          </div>

          <div className="actions">
            <button className="primary" onClick={start} disabled={running || !selected}><Power size={17} /> Start server</button>
            <button className="danger" onClick={stop} disabled={!running && status.state !== 'crashed'}><Square size={17} /> Stop</button>
            <button onClick={() => window.open(url, '_blank')}><Globe size={17} /> Open URL</button>
          </div>

          <div className="runtime-strip">
            <div><Cpu size={16} /> {options.backend === 'opencl' ? 'ONEAPI_DEVICE_SELECTOR=opencl:gpu' : options.backend === 'levelzero' ? 'ONEAPI_DEVICE_SELECTOR=level_zero:*' : 'CPU / -ngl 0'}</div>
            <div><MonitorPlay size={16} /> {status.external ? 'resynced external server' : url}</div>
            <div><Globe size={16} /> Inference API {lanInferenceUrl}</div>
            <div><Zap size={16} /> blank fields omitted</div>
          </div>
        </section>

        <section className="panel logs-panel">
          <div className="panel-title"><Terminal size={18} /><span>Server Logs</span></div>
          <div className="logs">
            {logs.length === 0 && <span className="muted">No logs yet.</span>}
            {logs.map((log, idx) => (
              <div key={idx} className={`log ${log.source}`}><span>{formatTime(log.t)}</span><b>{log.source}</b>{log.line}</div>
            ))}
          </div>
        </section>
      </section>
    </main>
  );
}

createRoot(document.getElementById('root')).render(<App />);
