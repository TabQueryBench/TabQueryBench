import http from 'node:http';
import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const here = path.dirname(fileURLToPath(import.meta.url));
const repo = path.resolve(here, '..');
const port = Number(process.env.PORT || 4173);
const roots = ['Synthesizing/raw_data', 'Synthesizing/synthetic_data', 'Synthesizing/release_manifests', 'Query/Queries', 'Query/Query_Templates', 'Scoring/Scoring_Standard'];
const skipped = new Set(['.git', 'node_modules', '__pycache__', '_tabdiff_runtime', '_tabddpm_runtime', 'wandb']);
const textExtensions = new Set(['.csv', '.tsv', '.json', '.jsonl', '.sql', '.txt', '.md', '.yaml', '.yml', '.log']);

const relative = (absolute) => path.relative(repo, absolute).split(path.sep).join('/');
const safePath = (value) => {
  const target = path.resolve(repo, value || '');
  return target === repo || target.startsWith(repo + path.sep) ? target : null;
};
async function walk(directory, files = []) {
  let entries = [];
  try { entries = await fs.readdir(directory, { withFileTypes: true }); } catch { return files; }
  for (const entry of entries) {
    if (skipped.has(entry.name)) continue;
    const absolute = path.join(directory, entry.name);
    if (entry.isDirectory()) await walk(absolute, files);
    else if (entry.isFile()) {
      const stat = await fs.stat(absolute);
      files.push({ path: relative(absolute), name: entry.name, size: stat.size, modified: stat.mtimeMs });
    }
  }
  return files;
}
function classify(file) {
  const lower = file.path.toLowerCase();
  if (lower.startsWith('synthesizing/raw_data/')) return 'dataset';
  if (/\.sql$|generated_sql/.test(lower)) return 'sql';
  if (/(query_)?results?|final_answer|answer/.test(lower)) return 'result';
  if (/(score|metric|analysis)/.test(lower)) return 'score';
  if (/(cost|timing|runtime|duration|latency)/.test(lower)) return 'costTiming';
  if (/(manifest|bundle|index)/.test(lower)) return 'bundle';
  if (lower.startsWith('synthesizing/synthetic_data/')) return 'synthetic';
  return 'template';
}
function summary(files) {
  const counts = Object.fromEntries(['dataset','synthetic','sql','result','score','costTiming','bundle','template'].map(k => [k, 0]));
  for (const f of files) counts[classify(f)]++;
  return counts;
}
async function inventory() {
  const all = (await Promise.all(roots.map(root => walk(path.join(repo, root))))).flat();
  all.sort((a,b) => a.path.localeCompare(b.path));
  return { repo, scannedAt: new Date().toISOString(), files: all, counts: summary(all) };
}
function respond(res, code, body, type = 'application/json; charset=utf-8') { res.writeHead(code, { 'content-type': type, 'cache-control': 'no-store' }); res.end(body); }
async function preview(target) {
  const stat = await fs.stat(target);
  if (!stat.isFile()) throw new Error('Not a file');
  const ext = path.extname(target).toLowerCase();
  if (!textExtensions.has(ext)) return { binary: true, message: 'Binary/large asset: preview is intentionally unavailable. Use the repository path link.' };
  const handle = await fs.open(target, 'r');
  try {
    const bytes = Math.min(stat.size, 64 * 1024);
    const buffer = Buffer.alloc(bytes);
    await handle.read(buffer, 0, bytes, 0);
    return { text: buffer.toString('utf8'), truncated: stat.size > bytes, size: stat.size };
  } finally { await handle.close(); }
}
const mime = { '.html': 'text/html; charset=utf-8', '.css': 'text/css; charset=utf-8', '.js': 'text/javascript; charset=utf-8' };
http.createServer(async (req, res) => {
  const url = new URL(req.url, `http://${req.headers.host}`);
  try {
    if (url.pathname === '/api/inventory') return respond(res, 200, JSON.stringify(await inventory()));
    if (url.pathname === '/api/preview') {
      const target = safePath(url.searchParams.get('path'));
      if (!target) return respond(res, 400, JSON.stringify({ error: 'Invalid path' }));
      return respond(res, 200, JSON.stringify(await preview(target)));
    }
    if (url.pathname === '/api/path') {
      const target = safePath(url.searchParams.get('path'));
      if (!target) return respond(res, 400, 'Invalid path', 'text/plain');
      return respond(res, 200, target, 'text/plain; charset=utf-8');
    }
    const requested = url.pathname === '/' ? 'index.html' : url.pathname.slice(1);
    const staticFile = path.resolve(here, 'public', requested);
    if (!staticFile.startsWith(path.join(here, 'public') + path.sep)) return respond(res, 403, 'Forbidden', 'text/plain');
    return respond(res, 200, await fs.readFile(staticFile), mime[path.extname(staticFile)] || 'application/octet-stream');
  } catch (error) { respond(res, error.code === 'ENOENT' ? 404 : 500, JSON.stringify({ error: error.message })); }
}).listen(port, '127.0.0.1', () => console.log(`TabQueryBench Browser: http://127.0.0.1:${port}`));
