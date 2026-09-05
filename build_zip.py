from pathlib import Path
import zipfile
root=Path(__file__).parent; out=root/'stream_runtime.zip'
with zipfile.ZipFile(out,'w',zipfile.ZIP_DEFLATED) as z:
    for p in root.rglob('*'):
        if p.is_file() and p.name not in {out.name} and '__pycache__' not in p.parts and '.pytest_cache' not in p.parts: z.write(p,p.relative_to(root.parent))
print(out)
