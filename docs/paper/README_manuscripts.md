# Summer Linear Attention: bilingual manuscript sources

## Contents

- `manuscript_en.md`: English research manuscript draft.
- `manuscript_ja.md`: Japanese counterpart with the same equations and scope.
- `references.bib`: editable BibTeX bibliography, in the manuscripts' reference order.
- `author_checklist.md`: bilingual checklist for evidence, author information, and publication preparation.

Typeset PDFs and their build instructions may be supplied beside these sources. Markdown uses YAML front matter, standard TeX mathematics, and explicit numbered references. The bibliography in each Markdown file is written out for portability; editing the BibTeX alone does not update those reference lists automatically.

## Evidence and version

Both manuscripts describe SummerLinearAttention v0.1.0 at:

https://github.com/Summer110622/SummerLinearAttention/tree/036512212ed0f7d2456988cb4b523599064a7e6f

The code is an untrained NumPy reference with a delta-rule state, bounded original-KV FIFO archive, null-aware softmax readout, and optional shared-state residual correction. It is distinct from the Summer Delta Attention companion project.

The drafts contain mathematical derivations, limitations, and a proposed evaluation protocol. They do not contain new benchmark measurements, trained-model results, speed claims, or asserted architectural priority. Older exploratory observations without a complete artifact trail are not presented as verified results. Functional tests listed in the source repository are not reported as having been executed for this paper.

The author line follows the repository's attribution, `summer`. No affiliation, legal name, contact address, submission status, or publication venue has been invented. Repository code is MIT-licensed; a publication license for these manuscript drafts has not been selected. The files are shared here as drafts; no venue submission, acceptance, or peer review is implied.

## 日本語

英語稿と日本語稿は、同じ固定コミットの手法、数式、限界を記述しています。実験結果は新たに測定しておらず、評価節は未実施の計画です。古い探索的な数値やSDAの図をSLAの検証済み結果として転用していません。

編集時は、数式、既定値、出典、証拠の範囲を日英で揃えてください。Markdown本文中の文献一覧は移植性のため明示的に記載しているので、BibTeXを変更した場合は本文の一覧も更新する必要があります。公開・投稿の前には `author_checklist.md` を確認してください。
