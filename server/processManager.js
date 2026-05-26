import { spawn } from 'node:child_process';
import fs from 'node:fs';
import net from 'node:net';
import { config } from './config.js';

const backendEnv = {
  opencl: 'opencl:gpu',
  levelzero: 'level_zero:*',
  cpu: '',
};

const optionSpecs = [
  ['host', '--host'], ['port', '--port'], ['ctx', '-c'], ['predict', '-n'], ['threads', '-t'], ['threadsBatch', '-tb'], ['batch', '-b'], ['ubatch', '-ub'],
  ['keep', '--keep'], ['cpuMask', '-C'], ['cpuRange', '-Cr'], ['cpuStrict', '--cpu-strict'], ['prio', '--prio'], ['poll', '--poll'],
  ['cpuMaskBatch', '-Cb'], ['cpuRangeBatch', '-Crb'], ['cpuStrictBatch', '--cpu-strict-batch'], ['prioBatch', '--prio-batch'], ['pollBatch', '--poll-batch'],
  ['ropeScaling', '--rope-scaling'], ['ropeScale', '--rope-scale'], ['ropeFreqBase', '--rope-freq-base'], ['ropeFreqScale', '--rope-freq-scale'],
  ['yarnOrigCtx', '--yarn-orig-ctx'], ['yarnExtFactor', '--yarn-ext-factor'], ['yarnAttnFactor', '--yarn-attn-factor'], ['yarnBetaSlow', '--yarn-beta-slow'], ['yarnBetaFast', '--yarn-beta-fast'],
  ['cacheTypeK', '-ctk'], ['cacheTypeV', '-ctv'], ['defragThold', '-dt'], ['numa', '--numa'], ['device', '-dev'], ['gpuLayers', '-ngl'], ['splitMode', '-sm'], ['tensorSplit', '-ts'], ['mainGpu', '-mg'],
  ['fit', '-fit'], ['fitTarget', '-fitt'], ['fitCtx', '-fitc'], ['overrideTensor', '-ot'], ['nCpuMoe', '-ncmoe'],
  ['lora', '--lora'], ['loraScaled', '--lora-scaled'], ['controlVector', '--control-vector'], ['controlVectorScaled', '--control-vector-scaled'], ['controlVectorLayerStart', '--control-vector-layer-range'],
  ['seed', '-s'], ['samplers', '--samplers'], ['samplerSeq', '--sampler-seq'], ['temperature', '--temp'], ['topK', '--top-k'], ['topP', '--top-p'], ['minP', '--min-p'], ['topNSigma', '--top-nsigma'], ['xtcProbability', '--xtc-probability'], ['xtcThreshold', '--xtc-threshold'], ['typicalP', '--typical-p'], ['repeatLastN', '--repeat-last-n'], ['repeatPenalty', '--repeat-penalty'], ['presencePenalty', '--presence-penalty'], ['frequencyPenalty', '--frequency-penalty'], ['dryMultiplier', '--dry-multiplier'], ['dryBase', '--dry-base'], ['dryAllowedLength', '--dry-allowed-length'], ['dryPenaltyLastN', '--dry-penalty-last-n'],
  ['parallel', '-np'], ['alias', '-a'], ['tags', '--tags'], ['apiPrefix', '--api-prefix'], ['uiConfig', '--ui-config'], ['uiConfigFile', '--ui-config-file'], ['tools', '--tools'], ['apiKey', '--api-key'], ['apiKeyFile', '--api-key-file'], ['sslKeyFile', '--ssl-key-file'], ['sslCertFile', '--ssl-cert-file'], ['timeout', '-to'], ['threadsHttp', '--threads-http'], ['slotPromptSimilarity', '-sps'], ['mediaPath', '--media-path'], ['modelsDir', '--models-dir'], ['modelsPreset', '--models-preset'], ['modelsMax', '--models-max'],
  ['reasoningFormat', '--reasoning-format'], ['reasoning', '--reasoning'], ['reasoningBudget', '--reasoning-budget'], ['reasoningBudgetMessage', '--reasoning-budget-message'], ['chatTemplate', '--chat-template'], ['chatTemplateFile', '--chat-template-file'], ['chatTemplateKwargs', '--chat-template-kwargs'],
  ['logFile', '--log-file'], ['logColors', '--log-colors'], ['verbosity', '-lv'], ['modelUrl', '-mu'], ['hfRepo', '-hf'], ['hfFile', '-hff'], ['hfToken', '-hft'],
];

const flagSpecs = [
  ['swaFull', '--swa-full'], ['kvOffload', '--kv-offload'], ['noKvOffload', '--no-kv-offload'], ['repack', '--repack'], ['noRepack', '--no-repack'], ['noHost', '--no-host'], ['mlock', '--mlock'], ['mmap', '--mmap'], ['noMmap', '--no-mmap'], ['directIo', '-dio'], ['noDirectIo', '-ndio'], ['checkTensors', '--check-tensors'], ['opOffload', '--op-offload'], ['noOpOffload', '--no-op-offload'], ['cpuMoe', '-cmoe'],
  ['ignoreEos', '--ignore-eos'], ['escape', '-e'], ['noEscape', '--no-escape'], ['special', '-sp'], ['warmup', '--warmup'], ['noWarmup', '--no-warmup'], ['spmInfill', '--spm-infill'],
  ['contBatching', '-cb'], ['noContBatching', '-nocb'], ['mmprojOffload', '--mmproj-offload'], ['noMmprojOffload', '--no-mmproj-offload'], ['ui', '--ui'], ['noWebUi', '--no-ui'], ['embedding', '--embedding'], ['rerank', '--rerank'], ['uiMcpProxy', '--ui-mcp-proxy'], ['metrics', '--metrics'], ['props', '--props'], ['slots', '--slots'], ['noSlots', '--no-slots'], ['cachePrompt', '--cache-prompt'], ['noCachePrompt', '--no-cache-prompt'], ['jinja', '--jinja'], ['noJinja', '--no-jinja'], ['skipChatParsing', '--skip-chat-parsing'], ['prefillAssistant', '--prefill-assistant'], ['noPrefillAssistant', '--no-prefill-assistant'], ['modelsAutoload', '--models-autoload'], ['noModelsAutoload', '--no-models-autoload'], ['offline', '--offline'], ['logDisable', '--log-disable'], ['logPrefix', '--log-prefix'], ['noLogPrefix', '--no-log-prefix'], ['logTimestamps', '--log-timestamps'], ['noLogTimestamps', '--no-log-timestamps'],
];

let managed = null;
const logs = [];
const maxLogs = 800;

function pushLog(source, text) {
  const lines = String(text).split(/\r?\n/).filter(Boolean);
  for (const line of lines) {
    logs.push({ t: Date.now(), source, line });
    if (logs.length > maxLogs) logs.shift();
  }
}

function quoteCmd(value) {
  return `"${String(value).replaceAll('"', '""')}"`;
}

function hasValue(value) {
  return value !== undefined && value !== null && String(value).trim() !== '';
}

function addValuedArg(args, flag, value) {
  if (hasValue(value)) args.push(flag, String(value).trim());
}

function buildLlamaArgs(options) {
  const args = [];
  if (hasValue(options.model)) args.push('-m', options.model);

  for (const [key, flag] of optionSpecs) {
    if (key === 'gpuLayers' && options.backend === 'cpu') continue;
    if (key === 'controlVectorLayerStart') {
      if (hasValue(options.controlVectorLayerStart) && hasValue(options.controlVectorLayerEnd)) {
        args.push(flag, String(options.controlVectorLayerStart).trim(), String(options.controlVectorLayerEnd).trim());
      }
      continue;
    }
    addValuedArg(args, flag, options[key]);
  }

  if (options.backend === 'cpu') args.push('-ngl', '0');

  for (const [key, flag] of flagSpecs) {
    if (options[key] === true) args.push(flag);
  }

  return args;
}

export function getLogs() {
  return logs;
}

export function getStatus() {
  if (!managed) return { state: 'stopped', running: false };
  const host = hasValue(managed.options.host) ? managed.options.host : '127.0.0.1';
  const port = hasValue(managed.options.port) ? Number(managed.options.port) : 8080;
  return {
    state: managed.state,
    running: managed.state === 'starting' || managed.state === 'running',
    pid: managed.pid,
    startedAt: managed.startedAt,
    command: managed.command,
    args: managed.args,
    options: managed.options,
    exitCode: managed.exitCode,
    port,
    url: `http://${host}:${port}`,
  };
}

export async function isPortOpen(port, host = '127.0.0.1') {
  return new Promise((resolve) => {
    const socket = net.createConnection({ host, port, timeout: 700 }, () => {
      socket.destroy();
      resolve(true);
    });
    socket.on('error', () => resolve(false));
    socket.on('timeout', () => {
      socket.destroy();
      resolve(false);
    });
  });
}

async function waitForPort(port, host) {
  for (let i = 0; i < 240; i += 1) {
    if (await isPortOpen(port, host)) return true;
    await new Promise((resolve) => setTimeout(resolve, 1000));
  }
  return false;
}

export async function startServer(options) {
  if (managed && (managed.state === 'starting' || managed.state === 'running')) {
    throw new Error('A managed llama-server instance is already running. Stop it first.');
  }

  const normalized = {
    ...options,
    backend: options.backend || 'opencl',
    host: hasValue(options.host) ? String(options.host).trim() : '',
    port: hasValue(options.port) ? Number(options.port) : '',
    openGui: Boolean(options.openGui),
  };

  const probeHost = normalized.host || '127.0.0.1';
  const probePort = normalized.port || 8080;

  if (!normalized.model || !fs.existsSync(normalized.model)) throw new Error('Selected model file does not exist.');
  if (!fs.existsSync(config.llamaServer)) throw new Error(`llama-server.exe not found at ${config.llamaServer}`);
  if (!fs.existsSync(config.oneApiSetvars)) throw new Error(`oneAPI setvars.bat not found at ${config.oneApiSetvars}`);
  if (await isPortOpen(probePort, probeHost)) throw new Error(`Port ${probePort} is already in use.`);

  const args = buildLlamaArgs(normalized);
  const selector = backendEnv[normalized.backend] ?? backendEnv.opencl;
  const extraArgs = hasValue(normalized.extraArgs) ? ` ${String(normalized.extraArgs).trim()}` : '';
  const envLines = [
    `call ${quoteCmd(config.oneApiSetvars)} >nul`,
    selector ? `set "ONEAPI_DEVICE_SELECTOR=${selector}"` : 'set "ONEAPI_DEVICE_SELECTOR="',
    normalized.enableSysman === false ? 'set "ZES_ENABLE_SYSMAN="' : 'set "ZES_ENABLE_SYSMAN=1"',
    `cd /d ${quoteCmd(config.llamaDir)}`,
    `${quoteCmd(config.llamaServer)} ${args.map(quoteCmd).join(' ')}${extraArgs}`,
  ];
  const command = envLines.join(' && ');

  logs.length = 0;
  pushLog('manager', `Starting backend=${normalized.backend} model=${normalized.model}`);
  pushLog('manager', command);

  const child = spawn('cmd.exe', ['/d', '/c', command], {
    cwd: config.llamaDir,
    windowsHide: false,
    windowsVerbatimArguments: true,
    stdio: ['ignore', 'pipe', 'pipe'],
  });

  managed = {
    child,
    pid: child.pid,
    state: 'starting',
    startedAt: Date.now(),
    command,
    args,
    options: normalized,
    exitCode: null,
  };

  child.stdout.on('data', (data) => pushLog('stdout', data));
  child.stderr.on('data', (data) => pushLog('stderr', data));
  child.on('exit', (code) => {
    pushLog('manager', `llama-server exited with code ${code}`);
    if (managed?.child === child) {
      managed.state = code === 0 ? 'stopped' : 'crashed';
      managed.exitCode = code;
    }
  });

  waitForPort(probePort, probeHost).then((ready) => {
    if (managed?.child !== child) return;
    if (ready) {
      managed.state = 'running';
      pushLog('manager', `Server ready at http://${probeHost}:${probePort}`);
      if (normalized.openGui) {
        spawn('cmd.exe', ['/c', 'start', '', `http://${probeHost}:${probePort}`], { windowsHide: true, detached: true });
      }
    } else if (managed?.state === 'starting') {
      managed.state = 'crashed';
      pushLog('manager', 'Server did not open its port before timeout.');
    }
  });

  return getStatus();
}

export async function stopServer() {
  if (!managed || !managed.child || managed.state === 'stopped') return getStatus();
  const pid = managed.child.pid;
  pushLog('manager', `Stopping process tree pid=${pid}`);
  await new Promise((resolve) => {
    const killer = spawn('taskkill.exe', ['/PID', String(pid), '/T', '/F'], { windowsHide: true });
    killer.on('exit', resolve);
    killer.on('error', resolve);
  });
  managed.state = 'stopped';
  managed.exitCode = 0;
  return getStatus();
}

