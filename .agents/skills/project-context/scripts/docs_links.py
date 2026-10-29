#!/usr/bin/env python3
"""Relative markdown links: check them all, or move docs and rewrite every link that pointed at the old place.

    python3 docs_links.py check [--root .]
    python3 docs_links.py move OLD NEW [OLD NEW ...] [--skip FILE ...] [--root .]

Run from the repository root. Skips hidden folders, node_modules and any path containing /artifacts/. `move` also recomputes the links
inside the moved markdown files. Files named with --skip (e.g. owned by a running session) are neither moved nor edited.
"""
import argparse
import os
import re
import shutil
from pathlib import Path

LINK = re.compile(r'(\[[^\]]*\]\()([^)\s#]+)((?:#[^)]*)?\))')


def documents(root):
    for md in root.rglob('*.md'):
        rel = md.relative_to(root).as_posix()   # hidden folders (.git, .compat, skill trees) are tooling, not project docs
        if not any(part.startswith('.') for part in rel.split('/')) and 'node_modules/' not in rel and '/artifacts/' not in rel: yield md


def local(target): return not target.startswith(('http:', 'https:', 'mailto:'))


def check(root):
    bad = [f'{md.relative_to(root)}: {m.group(2)}' for md in documents(root) for m in LINK.finditer(md.read_text(errors='replace'))
           if local(m.group(2)) and not (md.parent / m.group(2)).exists()]
    print('\n'.join(bad) if bad else 'all relative markdown links resolve'); return 1 if bad else 0


def move(root, pairs, skip):
    moves = {(root / a).resolve(): (root / b).resolve() for a, b in pairs}
    skip = {(root / s).resolve() for s in skip}
    for a, b in moves.items():
        assert a.exists() and not b.exists() and a not in skip, (a, b)
        b.parent.mkdir(parents=True, exist_ok=True); shutil.move(a, b)
    def remap(p):
        for a, b in moves.items():
            if p == a or a in p.parents: return b / p.relative_to(a)
        return p
    old_dir = {}   # moved markdown file -> directory its links were written against
    for a, b in moves.items():
        for md in ([b] if b.suffix == '.md' else b.rglob('*.md') if b.is_dir() else []):
            old_dir[md] = (a / md.relative_to(b)).parent if b.is_dir() else a.parent
    changed = 0
    for md in documents(root):
        if md.resolve() in skip: continue
        base = old_dir.get(md.resolve(), md.parent); text = md.read_text(errors='replace')
        def fix(m):
            t = m.group(2)
            if not local(t): return m.group(0)
            new = remap((base / t).resolve())
            if not new.exists() or (new == (base / t).resolve() and base == md.parent): return m.group(0)
            rel = os.path.relpath(new, md.parent).replace(os.sep, '/')
            return m.group(1) + rel + ('/' if t.endswith('/') and not rel.endswith('/') else '') + m.group(3)
        out = LINK.sub(fix, text)
        if out != text: md.write_text(out); changed += 1; print('relinked', md.relative_to(root))
    print(len(moves), 'moved,', changed, 'files relinked')


def main():
    p = argparse.ArgumentParser(); p.add_argument('command', choices=['check', 'move']); p.add_argument('paths', nargs='*')
    p.add_argument('--skip', nargs='*', default=[]); p.add_argument('--root', default='.')
    a = p.parse_args(); root = Path(a.root).resolve()
    if a.command == 'check': raise SystemExit(check(root))
    assert a.paths and len(a.paths) % 2 == 0, 'move needs OLD NEW pairs'
    move(root, list(zip(a.paths[::2], a.paths[1::2])), a.skip)


if __name__ == '__main__': main()
