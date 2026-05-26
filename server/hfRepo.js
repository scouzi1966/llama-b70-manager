function normalizeRepo(repo) {
  return String(repo || '').trim().replace(/^https:\/\/huggingface\.co\//i, '').replace(/\/tree\/.*$/i, '');
}

function validateRepo(repo) {
  if (!repo || !/^[A-Za-z0-9_.-]+\/[A-Za-z0-9_.-]+$/.test(repo)) {
    throw new Error('Enter a Hugging Face repo id like unsloth/Qwen3.6-27B-MTP-GGUF.');
  }
}

function quantFromName(name) {
  const match = name.match(/(?:^|[-_])(IQ\d_[A-Z0-9]+|Q\d(?:_[A-Z0-9]+)+|Q\d_[01]|MXFP\d(?:_[A-Z0-9]+)*|F16|BF16|F32)(?:-\d{5}-of-\d{5})?(?:\.gguf)?$/i);
  return match ? match[1].toUpperCase() : 'unknown';
}

function familyFromName(name) {
  return name
    .replace(/\.gguf$/i, '')
    .replace(/-\d{5}-of-\d{5}$/i, '')
    .replace(/[-_](?:IQ\d_[A-Z0-9]+|Q\d(?:_[A-Z0-9]+)+|Q\d_[01]|MXFP\d(?:_[A-Z0-9]+)*|F16|BF16|F32)$/i, '');
}

function sizeGb(size) {
  return typeof size === 'number' ? +(size / 1024 / 1024 / 1024).toFixed(2) : null;
}

export async function inspectRepo(repoInput) {
  const repo = normalizeRepo(repoInput);
  validateRepo(repo);

  const url = `https://huggingface.co/api/models/${repo}?blobs=false`;
  const response = await fetch(url, {
    headers: { 'user-agent': 'llama-b70-manager' },
  });
  if (!response.ok) {
    throw new Error(`Hugging Face returned ${response.status} for ${repo}. Check the repo name or login/private access.`);
  }

  const data = await response.json();
  const files = (data.siblings || [])
    .filter((sibling) => sibling.rfilename?.toLowerCase().endsWith('.gguf'))
    .map((sibling) => ({
      file: sibling.rfilename,
      include: sibling.rfilename,
      name: sibling.rfilename.split('/').pop(),
      quant: quantFromName(sibling.rfilename.split('/').pop()),
      family: familyFromName(sibling.rfilename.split('/').pop()),
      sizeBytes: sibling.size ?? null,
      sizeGb: sizeGb(sibling.size),
    }))
    .sort((a, b) => a.name.localeCompare(b.name));

  const collections = Object.values(files.reduce((acc, file) => {
    const key = file.family || 'unknown';
    acc[key] ||= { family: key, files: [], quants: [] };
    acc[key].files.push(file);
    if (!acc[key].quants.includes(file.quant)) acc[key].quants.push(file.quant);
    return acc;
  }, {})).sort((a, b) => a.family.localeCompare(b.family));

  return {
    repo,
    id: data.id || repo,
    author: data.author || repo.split('/')[0],
    private: Boolean(data.private),
    gated: data.gated || false,
    tags: data.tags || [],
    downloads: data.downloads ?? null,
    likes: data.likes ?? null,
    files,
    collections,
  };
}
