// The demo rebuilds the system prompt in JavaScript. It must be byte-identical to the
// one the model was trained on (lab/contractnli.py build_system), or "bring your own NDA"
// would send the fine-tuned model a prompt it never saw.
import { readFileSync } from 'node:fs';
import { execFileSync } from 'node:child_process';
const html = readFileSync(new URL('../demo/index.html', import.meta.url), 'utf8');
const meta = JSON.parse(readFileSync(new URL('../demo/data/index.json', import.meta.url), 'utf8'));
const src = html.match(/function buildSystem\(\)\{[\s\S]*?\n\}/)[0];
const buildSystem = new Function('META', src + '\nreturn buildSystem();');
const js = buildSystem(meta);
const py = execFileSync('python3', ['-c', `
import os, sys; sys.path.insert(0, 'lab'); os.environ.setdefault('CONTRACTNLI_DIR', 'data/contract-nli')
import contractnli as C
_, labels = C.load('test'); sys.stdout.write(C.build_system(labels))`], { encoding: 'utf8' });
if (js !== py) { console.error('MISMATCH\n--- js\n' + js.slice(0, 400) + '\n--- py\n' + py.slice(0, 400)); process.exit(1); }
console.log(`prompt parity OK (${js.length} chars)`);
