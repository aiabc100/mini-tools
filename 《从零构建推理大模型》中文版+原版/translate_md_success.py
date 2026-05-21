import re
import time
import sys
import translators as ts

INPUT_FILE = r"c:\xing_lab\xing-skill\input (1).md"
OUTPUT_FILE = r"c:\xing_lab\xing-skill\input (1)_cn.md"

def translate_text(text):
    if not text or not text.strip():
        return text
    if len(text.strip()) < 2:
        return text
    for attempt in range(3):
        try:
            result = ts.translate_text(text, translator='google', from_language='en', to_language='zh')
            if result is None:
                return text
            return result
        except Exception as e:
            print(f"Translation error (attempt {attempt+1}): {e}", file=sys.stderr)
            sys.stderr.flush()
            time.sleep(3 * (attempt + 1))
    return text

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

def translate_text_block(text):
    lines = text.split('\n')
    result_lines = []
    
    for line in lines:
        if not line.strip():
            result_lines.append(line)
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
            translated_cells = []
            for cell in cells:
                if cell.strip():
                    prot_content, prot = protect_special(cell)
                    tr = translate_text(prot_content)
                    tr = restore_special(tr, prot)
                    translated_cells.append(tr)
                else:
                    translated_cells.append(cell)
            result_lines.append('|'.join(translated_cells))
            continue
        
        prot_content, prot = protect_special(content)
        tr = translate_text(prot_content)
        tr = restore_special(tr, prot)
        result_lines.append(' ' * indent + prefix + tr)
    
    return '\n'.join(result_lines)

def main():
    print(f"Reading {INPUT_FILE}...", flush=True)
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        content = f.read()
    lines = content.split('\n')
    lines = [l + '\n' for l in lines[:-1]] + [lines[-1]] if lines else []
    
    print(f"Total lines: {len(lines)}", flush=True)
    
    blocks = parse_blocks(lines)
    print(f"Total blocks: {len(blocks)}", flush=True)
    
    text_block_count = sum(1 for t, _ in blocks if t == 'text')
    print(f"Text blocks to translate: {text_block_count}", flush=True)
    
    output_parts = []
    translated_count = 0
    
    for i, (btype, bcontent) in enumerate(blocks):
        if btype == 'text':
            translated = translate_text_block(bcontent)
            output_parts.append(translated)
            translated_count += 1
            if translated_count % 20 == 0:
                print(f"Translated {translated_count}/{text_block_count} text blocks...", flush=True)
        else:
            output_parts.append(bcontent if bcontent.endswith('\n') or not bcontent else bcontent)
    
    print(f"Writing {OUTPUT_FILE}...", flush=True)
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        f.write(''.join(output_parts))
    
    print(f"Translation complete! Translated {translated_count} text blocks.", flush=True)

if __name__ == '__main__':
    main()
