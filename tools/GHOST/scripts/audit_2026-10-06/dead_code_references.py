"""Every top-level def/class in ghost_backend production code with no production reference: who references it?"""
import ast
import re
from pathlib import Path

ROOT = Path(r'C:\Users\14col\Documents\20261005_GRIM_claude\tools\GHOST')
PKG = ROOT / 'ghost_backend'
EXCLUDE = {'tests', 'validation', 'data_tools', '__pycache__'}


def files(root, suffixes):
    for p in root.rglob('*'):
        if p.is_file() and p.suffix in suffixes and '__pycache__' not in p.parts:
            yield p


prod = [p for p in files(PKG, {'.py'}) if not any(part in EXCLUDE for part in p.relative_to(PKG).parts[:-1])]
other = [p for p in files(ROOT, {'.py', '.c', '.bat', '.toml', '.json', '.txt'}) if p not in prod]
texts_prod = {p: p.read_text(encoding='utf-8-sig', errors='replace') for p in prod}
texts_other = {p: p.read_text(encoding='utf-8-sig', errors='replace') for p in other}

defs = []
for p, text in texts_prod.items():
    try:
        tree = ast.parse(text)
    except SyntaxError:
        continue
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            defs.append((node.name, p, node.lineno, node.end_lineno))

rows = []
for name, p, lineno, end in defs:
    pattern = re.compile(r'(?<![\w.])' + re.escape(name) + r'\b')   # not an attribute access of another object
    attr = re.compile(r'\.' + re.escape(name) + r'\b')
    own = 0
    for q, t in texts_prod.items():
        hits = len(pattern.findall(t)) + len(attr.findall(t))
        if q == p:
            hits -= 1
        own += hits
    if own:
        continue
    refs = {}
    for q, t in texts_other.items():
        hits = len(pattern.findall(t)) + len(attr.findall(t))
        if hits:
            refs[str(q.relative_to(ROOT)).replace('\\', '/')] = hits
    rows.append((str(p.relative_to(PKG)).replace('\\', '/'), lineno, end - lineno + 1, name, refs))

rows.sort()
print(len(rows), 'top-level definitions without production references')
for rel, lineno, size, name, refs in rows:
    print('%-45s %5d %4d  %-48s %s' % (rel, lineno, size, name, ', '.join('%s(%d)' % (k.split('/')[-1], v) for k, v in sorted(refs.items())) or '-'))
