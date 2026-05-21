import re
import sys
import time
import torch
from transformers import MarianMTModel, MarianTokenizer

INPUT_FILE = r"c:\xing_lab\xing-skill\input (1).md"
OUTPUT_FILE = r"c:\xing_lab\xing-skill\input (1)_cn_gpu.md"
MODEL_NAME = "Helsinki-NLP/opus-mt-en-zh"
BATCH_SIZE = 32
MAX_LENGTH = 512

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {device}", flush=True)
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_mem / 1024**3:.1f} GB", flush=True)

print(f"Loading model {MODEL_NAME}...", flush=True)
tokenizer = MarianTokenizer.from_pretrained(MODEL_NAME)
model = MarianMTModel.from_pretrained(MODEL_NAME).to(device)
model.eval()
print("Model loaded.", flush=True)


def protect_special(text):
    protected = {}
    counter = [0]

    def _save(m):
        key = f"XPROT{counter[0]}X"
        protected[key] = m.group(0)
        counter[0] += 1
        return key

    text = re.sub(r'```[\s\S]*?```', _save, text)
    text = re.sub(r'`[^`]+`', _save, text)
    text = re.sub(r'\!\[.*?\]\(.*?\)', _save, text)
    text = re.sub(r'\$\$[\s\S]*?\$\$', _save, text)
    text = re.sub(r'(?<!\$)\$[^$\n]+?\$(?!\$)', _save, text)
    text = re.sub(r'\\\[.*?\\\]', _save, text)
    text = re.sub(r'\\\([^)]*?\\\)', _save, text)
    text = re.sub(r'\\boxed\{[^}]*\}', _save, text)
    text = re.sub(r'https?://\S+', _save, text)

    return text, protected


def restore_special(text, protected):
    for key, val in sorted(protected.items(), key=lambda x: -len(x[0])):
        text = text.replace(key, val)
    return text


def parse_blocks(lines):
    blocks = []
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        if stripped.startswith('```'):
            code_lines = [line]
            i += 1
            while i < len(lines) and not lines[i].strip().startswith('```'):
                code_lines.append(lines[i])
                i += 1
            if i < len(lines):
                code_lines.append(lines[i])
                i += 1
            blocks.append(('code', ''.join(code_lines)))
        elif not stripped:
            blocks.append(('blank', line))
            i += 1
        elif re.match(r'^!\[.*?\]\(.*?\)', stripped):
            blocks.append(('image', line))
            i += 1
        elif re.match(r'^\|[\s\-:|]+\|$', stripped):
            blocks.append(('table_sep', line))
            i += 1
        else:
            text_lines = []
            while i < len(lines):
                s = lines[i].strip()
                if s.startswith('```') or not s:
                    break
                if re.match(r'^!\[.*?\]\(.*?\)', s):
                    break
                if re.match(r'^\|[\s\-:|]+\|$', s):
                    break
                text_lines.append(lines[i])
                i += 1
            blocks.append(('text', ''.join(text_lines)))

    return blocks


def extract_segments_from_block(text):
    segments = []
    lines = text.split('\n')

    for line in lines:
        if not line.strip():
            segments.append({'type': 'blank', 'line': line})
            continue

        stripped = line.strip()
        indent = len(line) - len(line.lstrip())

        prefix = ''
        content = stripped

        if stripped.startswith('#'):
            m = re.match(r'^(#+\s*)(.*)', stripped)
            if m:
                prefix = m.group(1)
                content = m.group(2)
        elif re.match(r'^[-*]\s+', stripped):
            m = re.match(r'^([-*]\s+)(.*)', stripped)
            if m:
                prefix = m.group(1)
                content = m.group(2)
        elif re.match(r'^\d+\.\s+', stripped):
            m = re.match(r'^(\d+\.\s+)(.*)', stripped)
            if m:
                prefix = m.group(1)
                content = m.group(2)
        elif stripped.startswith('|') and stripped.endswith('|'):
            cells = stripped.split('|')
            cell_infos = []
            for cell in cells:
                if cell.strip():
                    prot_content, prot = protect_special(cell)
                    cell_infos.append({
                        'type': 'cell',
                        'original': cell,
                        'content': prot_content,
                        'protected': prot,
                    })
                else:
                    cell_infos.append({'type': 'blank_cell', 'original': cell})
            segments.append({'type': 'table_row', 'cells': cell_infos, 'indent': indent})
            continue

        prot_content, prot = protect_special(content)
        segments.append({
            'type': 'text',
            'indent': indent,
            'prefix': prefix,
            'content': prot_content,
            'protected': prot,
        })

    return segments


def collect_translatable_texts(all_segments):
    texts = []
    for seg in all_segments:
        if seg['type'] == 'text':
            c = seg['content']
            if c and c.strip() and len(c.strip()) >= 2:
                texts.append(c)
            else:
                texts.append(None)
        elif seg['type'] == 'table_row':
            for cell in seg['cells']:
                if cell['type'] == 'cell':
                    c = cell['content']
                    if c and c.strip() and len(c.strip()) >= 2:
                        texts.append(c)
                    else:
                        texts.append(None)
    return texts


def batch_translate_gpu(texts):
    valid_indices = [i for i, t in enumerate(texts) if t is not None]
    valid_texts = [texts[i] for i in valid_indices]

    if not valid_texts:
        return [None] * len(texts)

    results = [None] * len(texts)
    total = len(valid_texts)
    done = 0

    with torch.no_grad():
        for start in range(0, total, BATCH_SIZE):
            batch = valid_texts[start:start + BATCH_SIZE]

            try:
                inputs = tokenizer(
                    batch,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=MAX_LENGTH,
                ).to(device)

                outputs = model.generate(
                    **inputs,
                    max_length=MAX_LENGTH,
                    num_beams=4,
                    no_repeat_ngram_size=3,
                )

                decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)

                for j, translated in enumerate(decoded):
                    results[valid_indices[start + j]] = translated

            except Exception as e:
                print(f"Batch error at {start}: {e}", file=sys.stderr)
                sys.stderr.flush()
                for j in range(len(batch)):
                    results[valid_indices[start + j]] = batch[j]

            done += len(batch)
            if done % (BATCH_SIZE * 10) == 0 or done == total:
                print(f"  GPU translated {done}/{total} segments...", flush=True)

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    return results


def reconstruct_line(seg, translated_pool, idx_holder):
    if seg['type'] == 'blank':
        return seg['line']

    if seg['type'] == 'table_row':
        translated_cells = []
        for cell in seg['cells']:
            if cell['type'] == 'cell':
                tr = translated_pool[idx_holder[0]]
                idx_holder[0] += 1
                if tr is None:
                    tr = cell['content']
                tr = restore_special(tr, cell['protected'])
                translated_cells.append(tr)
            else:
                translated_cells.append(cell['original'])
        return ' ' * seg['indent'] + '|'.join(translated_cells)

    if seg['type'] == 'text':
        tr = translated_pool[idx_holder[0]]
        idx_holder[0] += 1
        if tr is None:
            tr = seg['content']
        tr = restore_special(tr, seg['protected'])
        return ' ' * seg['indent'] + seg['prefix'] + tr

    return ''


def main():
    print(f"Reading {INPUT_FILE}...", flush=True)
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        content = f.read()
    lines = content.split('\n')
    lines = [l + '\n' for l in lines[:-1]] + [lines[-1]] if lines else []

    print(f"Total lines: {len(lines)}", flush=True)

    blocks = parse_blocks(lines)
    print(f"Total blocks: {len(blocks)}", flush=True)

    text_block_indices = [i for i, (btype, _) in enumerate(blocks) if btype == 'text']
    print(f"Text blocks to translate: {len(text_block_indices)}", flush=True)

    print("Phase 1: Extracting segments...", flush=True)
    all_segments = []
    block_seg_ranges = {}

    for bi in text_block_indices:
        _, bcontent = blocks[bi]
        segs = extract_segments_from_block(bcontent)
        start = len(all_segments)
        all_segments.extend(segs)
        block_seg_ranges[bi] = (start, len(segs))

    print("Phase 2: Collecting translatable texts...", flush=True)
    translatable_texts = collect_translatable_texts(all_segments)
    non_none = sum(1 for t in translatable_texts if t is not None)
    print(f"Total translatable segments: {non_none}", flush=True)

    print("Phase 3: GPU batch translation...", flush=True)
    t0 = time.time()
    translated_pool = batch_translate_gpu(translatable_texts)
    elapsed = time.time() - t0
    print(f"GPU translation done in {elapsed:.1f}s ({non_none/elapsed:.1f} seg/s)", flush=True)

    print("Phase 4: Reconstructing document...", flush=True)
    idx_holder = [0]
    output_parts = []

    for bi, (btype, bcontent) in enumerate(blocks):
        if btype != 'text':
            output_parts.append(bcontent if bcontent.endswith('\n') or not bcontent else bcontent)
            continue

        start, count = block_seg_ranges[bi]
        result_lines = []
        for seg in all_segments[start:start + count]:
            line = reconstruct_line(seg, translated_pool, idx_holder)
            result_lines.append(line)
        output_parts.append('\n'.join(result_lines))

    print(f"Writing {OUTPUT_FILE}...", flush=True)
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        f.write(''.join(output_parts))

    print(f"Translation complete! {len(text_block_indices)} blocks, {non_none} segments in {elapsed:.1f}s", flush=True)


if __name__ == '__main__':
    main()
