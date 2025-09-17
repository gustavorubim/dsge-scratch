from pathlib import Path

text = Path(r'model_base\\EA_GNSS10\\EA_GNSS10_rep\\EA_GNSS10_rep.mod').read_bytes().decode('cp1252')
# Strip comments similar to parser
clean_lines = []
inside_block = False
buf = []
i = 0
while i < len(text):
    if not inside_block and text.startswith('/*', i):
        inside_block = True
        i += 2
        continue
    if inside_block and text.startswith('*/', i):
        inside_block = False
        i += 2
        continue
    if inside_block:
        i += 1
        continue
    buf.append(text[i])
    i += 1
stripped = ''.join(buf)
for raw in stripped.splitlines():
    line = raw
    for marker in ('//', '%', '#'):
        idx = line.find(marker)
        if idx != -1:
            line = line[:idx]
    clean_lines.append(line)

lines = [line.strip() for line in clean_lines]
state = None
buffer = []

for ln in lines:
    if not ln:
        continue
    lower = ln.lower()
    if state is None:
        if lower == 'var' or lower.startswith('var '):
            print('START VAR:', repr(ln))
            state = 'var'
            buffer = [ln]
        elif lower == 'parameters' or lower.startswith('parameters '):
            state = 'parameters'
        elif lower.startswith('model'):
            state = 'model'
    elif state == 'var':
        print('VAR LINE:', repr(ln))
        buffer.append(ln)
        if ln.endswith(';'):
            print('VAR BLOCK ENDS WITH ; on:', repr(ln))
            break

print('\nFirst VAR buffer preview (first 6 lines):')
for s in buffer[:6]:
    print(repr(s))
print('\nLast VAR line:', repr(buffer[-1]) if buffer else None)
