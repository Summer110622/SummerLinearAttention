#!/usr/bin/env python3
"""Rebuild bilingual SLA PDFs from approved Markdown, using Pandoc and XeLaTeX.
No network requests or manuscript edits are made. Generated .tex files are editable.
"""
from pathlib import Path
import argparse, os, shutil, subprocess, sys, re

ROOT = Path(__file__).resolve().parents[1]
TYPESET = ROOT / 'typesetting'
BUILD = ROOT / 'tmp' / 'pdfs'
CACHE = ROOT / 'tmp' / 'texmf-var'

def run(cmd, env=None, cwd=ROOT, capture=False):
    return subprocess.run(cmd, env=env, cwd=cwd, check=True,
        text=True, stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.STDOUT if capture else None)

def environment():
    env = os.environ.copy()
    BUILD.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    # These values cover the provided Debian/TeX Live runtime, whose ls-R and
    # prebuilt format caches may be absent. A standard TeX Live installation
    # already has its own lookup paths and formats and needs no bootstrap.
    if Path('/usr/share/texlive/texmf-dist').exists():
        trees = [str(CACHE), '/usr/share/texlive/texmf-dist', '/usr/share/texmf', '/etc/texmf']
        env['TEXMF'] = '{' + ','.join(trees) + '}'
        env['OPENTYPEFONTS'] = '/usr/share/fonts//:/usr/share/texmf/fonts/opentype//:/usr/share/texlive/texmf-dist/fonts/opentype//:'
    env['TEXMFVAR'] = str(CACHE)
    env['XDG_CACHE_HOME'] = str(ROOT / 'tmp' / 'font-cache')
    env['TEXFORMATS'] = str(CACHE) + '//:'
    has_format = subprocess.run(['kpsewhich', 'xelatex.fmt'], env=env, text=True, stdout=subprocess.PIPE).stdout.strip()
    if not has_format:
        run(['xetex', '-ini', '-etex', '-jobname=xelatex', '-progname=xelatex',
             '-interaction=nonstopmode', '-halt-on-error', 'xelatex.ini'], env, CACHE, True)
        env['TEXFORMATS'] = str(CACHE) + '//:'
    return env

def build(lang, env):
    source = ROOT / f'manuscript_{lang.lower()}.md'
    tex = TYPESET / f'SLA_paper_{lang}.tex'
    if not source.exists():
        raise FileNotFoundError(f'Missing manuscript: {source.name}')
    cmd = ['pandoc', str(source), '--from=markdown+tex_math_dollars+raw_tex',
           '--to=latex', '--standalone', '--template', str(TYPESET / 'paper-template.tex'),
           '--no-highlight', '--wrap=none', '--shift-heading-level-by=-1',
           '--lua-filter', str(TYPESET / 'format.lua'), '-o', str(tex)]
    if lang == 'JA': cmd += ['-V', 'japanese=true']
    run(cmd, env)
    for passno in range(2):
        result = run(['xelatex', '-no-shell-escape', '-interaction=nonstopmode',
                      '-halt-on-error', '-file-line-error',
                      '-output-directory=' + str(BUILD), str(tex)], env, capture=True)
        (BUILD / f'SLA_paper_{lang}_pass{passno+1}.txt').write_text(result.stdout)
    output = ROOT / f'SLA_paper_{lang}.pdf'
    shutil.copy2(BUILD / output.name, output)
    log = (BUILD / f'SLA_paper_{lang}.log').read_text()
    warnings = [x for x in log.splitlines() if 'Overfull' in x or 'Missing character' in x or 'undefined' in x]
    print(f'{output.name}: created; layout/glyph warnings: {len(warnings)}')
    for warning in warnings: print('  ' + warning)
    return output

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--lang', choices=['EN','JA','both'], default='both')
    args = ap.parse_args()
    env = environment()
    for lang in (['EN', 'JA'] if args.lang == 'both' else [args.lang]):
        build(lang, env)

if __name__ == '__main__': main()
