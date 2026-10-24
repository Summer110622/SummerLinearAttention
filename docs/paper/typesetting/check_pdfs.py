#!/usr/bin/env python3
"""Structural QA and page rendering; visual inspection of every PNG is still required."""
from pathlib import Path
import json, subprocess, re
from pypdf import PdfReader
import pdfplumber

ROOT = Path(__file__).resolve().parents[1]
render_dir = ROOT / 'tmp' / 'pdfs' / 'pages'
render_dir.mkdir(parents=True, exist_ok=True)
summary = {}
for lang in ('EN', 'JA'):
    path = ROOT / f'SLA_paper_{lang}.pdf'
    if not path.exists(): continue
    reader = PdfReader(path)
    uri_links = set()
    for page in reader.pages:
        for a in page.get('/Annots', []):
            annotation = a.get_object()
            action = annotation.get('/A', {})
            if action.get('/S') == '/URI': uri_links.add(str(action.get('/URI')))
    pages = []
    with pdfplumber.open(path) as pdf:
        for i, p in enumerate(pdf.pages, 1):
            chars = [c for c in p.chars if c.get('text', '').strip()]
            page_text = p.extract_text() or ''
            # The text block starts 70.9 pt from each side; allow 3 pt for italics.
            overflow = [c for c in chars if c['x0'] < 67.5 or c['x1'] > p.width - 67.5]
            pages.append({'page': i, 'characters': len(chars),
                          'horizontal_margin_violations': len(overflow),
                          'replacement_character': '\ufffd' in page_text,
                          'text_start': page_text[:100]})
    subprocess.run(['pdftoppm', '-png', '-r', '115', str(path), str(render_dir / lang)], check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    summary[lang] = {'pages': len(reader.pages), 'external_links': len(uri_links),
                     'page_checks': pages, 'linked_urls': sorted(uri_links)}
(ROOT / 'tmp' / 'pdfs' / 'qa_report.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2))
print(json.dumps(summary, ensure_ascii=False, indent=2))
