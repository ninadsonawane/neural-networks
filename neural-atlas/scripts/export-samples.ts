import { mkdir, writeFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import { validateTrace } from '../src/simulation/adapter';
import { buildTrace, DEFAULT_CONFIG } from '../src/simulation/engine';
import type { Architecture } from '../src/simulation/types';

const cat = 'the cat that chased the mouse was ...';
const hinglish = 'chai thandi thi par yaad garam hai';
const architectures: Architecture[] = ['mlp', 'rnn', 'lstm', 'attention', 'transformer'];
const samples = [
  ...architectures.map((architecture) => ({ filename: `${architecture}.json`, architecture, input: cat })),
  { filename: 'transformer-hinglish.json', architecture: 'transformer' as const, input: hinglish },
];
const destination = new URL('../public/traces/', import.meta.url);

// Eight decimal places keep these inspectable exports compact while preserving
// distribution normalization and state norms within the public schema tolerance.
const outputs = samples.map(({ filename, architecture, input }) => {
  const trace = buildTrace(input, architecture, DEFAULT_CONFIG);
  trace.metadata.source = 'Neural Atlas deterministic educational sample';
  const json = JSON.stringify(trace, (_key: string, value: unknown) =>
    typeof value === 'number' ? Number(value.toFixed(8)) : value) + '\n';
  const bytes = Buffer.byteLength(json, 'utf8');
  if (bytes > 2 * 1024 * 1024) throw new Error(`${filename} exceeds the 2 MB import limit`);
  validateTrace(JSON.parse(json) as unknown);
  return { filename, json, bytes };
});

const totalBytes = outputs.reduce((sum, output) => sum + output.bytes, 0);
if (totalBytes > 500 * 1024) throw new Error(`Sample traces exceed the 500 KB total target: ${totalBytes} bytes`);
await mkdir(destination, { recursive: true });
for (const { filename, json, bytes } of outputs) {
  const output = new URL(filename, destination);
  await writeFile(output, json, 'utf8');
  process.stdout.write(`${fileURLToPath(output)} · ${bytes.toLocaleString('en-US')} bytes\n`);
}
process.stdout.write(`Validated ${outputs.length} educational traces, ${totalBytes.toLocaleString('en-US')} bytes total.\n`);
