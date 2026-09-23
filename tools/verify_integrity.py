from pathlib import Path
import hashlib, json
root=Path(__file__).resolve().parents[1]
items=json.loads((root/'MANIFEST_SHA256.json').read_text(encoding='utf-8'))
errors=[]
for item in items:
    p=root/item['file']
    if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest()!=item['sha256']:
        errors.append(item['file'])
if errors:
    raise SystemExit('FAILED: '+', '.join(errors))
print(f'PASS: {len(items)} files match the publication SHA-256 manifest.')
