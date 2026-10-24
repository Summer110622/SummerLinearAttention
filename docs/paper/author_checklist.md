# Author checklist / 著者確認事項

These are manuscript preparation items, not completed experiments or submission claims.
以下は原稿の準備事項であり、実験や投稿の完了を意味しません。

1. **Confirm author information.** The draft uses the repository attribution “summer.” Supply any desired full name, affiliation, contact address, acknowledgments, and contribution statement before publication. Do not infer them from a username.
   **著者情報の確認。** 草稿ではリポジトリの著者表記「summer」を使用しています。公開時に必要な氏名、所属、連絡先、謝辞、貢献表記を確認してください。

2. **Keep the scope explicit.** This is a method/reference-implementation draft with mathematical analysis and an unexecuted evaluation protocol. It is not a trained-model or performance-comparison paper yet.
   **位置付けを維持。** 現状は数理解析と未実施の評価計画を含む手法・参照実装の草稿です。学習済みモデルや性能比較の実証論文にはまだなっていません。

3. **Review prior-art positioning.** Check HOLA, Pointer Sentinel Mixture Models, Learnable Token Eviction, and delta-rule work. Do not claim the first recurrent-plus-cache model, first null/sentinel fallback, or a demonstrated improvement over those systems.
   **先行研究との差分を確認。** HOLA、Pointer Sentinel Mixture Models、Learnable Token Eviction、デルタ則の研究との関係を確認してください。構成全体やnull混合の初出、既存手法に対する実証済みの優位性は主張していません。

4. **Validate the pinned implementation.** Preserve the exact source commit, environment versions, functional-test output, and known numerical limitations. Tests must actually be run before reporting a pass count.
   **固定版の実装を検証。** コミット、環境の版、機能テスト出力、数値計算上の限界を保存してください。テスト合格数を記載する前に実行が必要です。

5. **Run controlled experiments before adding results.** Publish generators, calibration/test seed separation, memory-matched baselines, raw per-seed data, paired uncertainty estimates, and plotting scripts. Do not transfer SDA figures or memory numbers to SLA.
   **結果の追加前に統制実験を実施。** 生成器、較正とテストのシード分離、同一メモリ予算の比較、生データ、対応付きの不確実性評価、作図コードを公開してください。SDAの図やメモリ数値はSLAに転用しないでください。

6. **Review causality and target definitions.** State the read/write order, the meaning of an overwritten identity, and whether each target is retained, evicted, or never observed. A null weight is not a correctness or abstention label.
   **因果順序と正解の定義を確認。** 読み書きの順序、ID上書きの意味、保持中・除去済み・未観測の区別を明示してください。null重みを正解確率や棄権ラベルと見なさないでください。

7. **Maintain bilingual consistency.** Keep all equations, defaults, references, limitations, and evidence-status statements synchronized when editing either manuscript. The two texts describe the same method.
   **日英の整合性を維持。** 片方を修正した際は、数式、既定値、文献、限界、実験の実施状況をもう一方にも反映してください。両原稿は同じ手法を記述しています。

8. **Prepare venue submission separately.** Choose a venue or technical-report format, check its authorship and AI-assistance disclosure policies, verify required licenses and releases, and obtain author approval. Sharing these drafts in the repository does not submit them to a venue or imply acceptance or peer review.
   **投稿は別途準備。** 投稿先や技術報告の形式、著者資格、AI支援の開示方針、必要なライセンスと公開条件を確認し、著者の承認を得てください。リポジトリでの草稿の共有は、学会・論文誌への投稿、採択、査読済みであることを意味しません。
