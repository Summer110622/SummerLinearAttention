# Summer Linear Attention: English and Japanese paper drafts

Author: **summer**. Status: **method/reference-implementation drafts**, not peer-reviewed or venue-submitted papers.

## Read the papers

- [English PDF](SLA_paper_EN.pdf) · [editable Markdown](manuscript_en.md) · [editable LaTeX](typesetting/SLA_paper_EN.tex)
- [日本語 PDF](SLA_paper_JA.pdf) · [編集用 Markdown](manuscript_ja.md) · [編集用 LaTeX](typesetting/SLA_paper_JA.tex)
- [Bibliography](references.bib), [source-package notes](README_manuscripts.md), and [author checklist / 著者確認事項](author_checklist.md)
- [Build requirements and instructions](typesetting/README.md)

## Scope and pinned implementation

Both manuscripts describe SLA v0.1.0 at commit
[`036512212ed0f7d2456988cb4b523599064a7e6f`](https://github.com/Summer110622/SummerLinearAttention/tree/036512212ed0f7d2456988cb4b523599064a7e6f).
They share 15 numbered equations and cover the delta-rule state, bounded original-KV FIFO archive, null-aware retrieval, and optional shared-state residual correction.

The papers contain mathematical analysis, limitations, related work (including HOLA and Pointer Sentinel Mixture Models), and a proposed evaluation protocol. **They do not report new benchmark measurements, trained-model results, GPU performance, or demonstrated superiority over other attention methods.** The evaluation protocol has not been executed for these manuscripts. Earlier exploratory observations are not promoted to verified paper results. SLA and the companion Summer Delta Attention project remain distinct.

The PDFs and manuscript bodies preserve the prepared drafts. Adding them to this repository does not imply venue submission, acceptance, peer review, or a selected publication venue/license. See the author checklist before submission or making empirical claims. Repository code remains under the existing MIT license; this PR does not change that license.

## Rebuild and integrity

From this directory, with the dependencies listed in the [build guide](typesetting/README.md):

```sh
python3 typesetting/build_papers.py
python3 typesetting/check_pdfs.py
```

The build uses local Markdown, a shared LaTeX template, and a Pandoc filter. It writes temporary files and QA page images under `tmp/`, which is ignored. Inspect every rendered page after edits; structural checks alone are insufficient. `SHA256SUMS` records the committed paper-package files, excluding itself. Verify before rebuilding with `sha256sum -c SHA256SUMS`. Rebuilt PDF bytes may differ because of typesetting metadata.

## 日本語

英日両稿は同じ固定コミットに基づく手法・参照実装の草稿です。数式、出典、限界、未実施の評価計画を揃えています。新しいベンチマーク結果、学習済みモデルの評価、GPU性能、他手法に対する実証済みの優位性は報告していません。リポジトリでの共有は、学会・論文誌への投稿、採択、査読済みであることを意味しません。
