# solver 構成の追補 — dense 表現への切替 — **結果を見る前に凍結** — 2026年9月18日

前提：SOLVER_SPEC_World1G（f6c1ad0 → 503d2f4 → a66e9e4）
経緯：到達可能集合の規模測定で、(1) macro を状態ごとに辿る実装が重複で遅い、(2) primitive global propagation で946倍になり support は完全一致、(3) その版は Python の集合表現でメモリが壁（t=51 で RSS 8GB）と分かった。

**これは近似ではない。状態の表現方法を変えるだけ。** 確率の打ち切り・標本化・粗視化・状態の集約は一切入れない。

---

## 1. 決めること

1. **primitive な1歩 MDP を solver の正本候補とする**（「候補」と書くのは、dense 版と tuple/set 版の一致確認が済むまで確定しないため。一致したら正式採用）
2. 状態は**固定 index の dense 表現**（bool マスク／値の配列）
3. reachability は current / next の**2層**のマスク
4. 後ろ向き帰納も current / next の**2層**（primitive なら常に t → t+1 なので101層は不要）
5. **終端は未来の frontier に保持しない**
6. D1a の48条件は**最初は1条件ずつ**解く（同時に持つとメモリが増える。速度より正しさ）
7. tuple/set 版との **count と SHA-256 digest の一致を必須の検証**とする
8. **RSS watchdog（8GB）を外側から常時使う**（`watchdog_run.py`。2026/9/18 に試験済み：上限1.0GB に対し 1.09GB で0.8秒後に停止）

**dense を選ぶ理由**：状態空間の理論上限が既知で、**メモリ使用量を事前に上限評価できる**から。飽和の予想を理由にしない（飽和するかは t=50 までしか測れていない）。

## 2. MODE の実測（2026/9/18）

定義は9種類だが、**frontier に現れるのは8種類**。`grass` は現れない（decision の歩で草を食べ、その歩の終わりに decision へ戻るため）。

```
decision / search_small / chase_1 / chase_2 / chase_3 / chase_4 / search_large / combat
```

t=0..30 の全 frontier を走査して確認した。

## 3. 配列の寸法

| | 値 |
|---|---|
| decision 状態の直積 | 2001 × 11 × 12 × 4 = **1,056,528** |
| primitive 状態の直積（MODE 8） | **8,452,224** |
| bool マスク1層 | 8.5 MB |
| float64 1層 | **67.6 MB** |
| 値の2層 + マスク数枚 | 200MB 前後の見込み |

index の決め方（固定）：
```
idx = ((mode_i * 2001 + e) * 11 + w) * 12 + (n_small - 1)) * 4 + n_large
mode_i: decision=0, search_small=1, chase_1=2, chase_2=3, chase_3=4,
        chase_4=5, search_large=6, combat=7
e: 0..2000 ／ w: 0..10 ／ n_small: 1..12 ／ n_large: 0..3
```

## 4. 計算の仕方

dense で持っても、毎時刻845万状態を Python のループで舐めたら時間で死ぬ。
**NumPy のベクトル化された更新**を基本にする。到達可能マスク（valid mask）で無効な状態を落とす。

## 5. 検証の順番

```
本追補の凍結（このファイル）
  → dense マスク版の reachability prototype
  → tuple/set 版と count + digest の一致（t=0..11、可能なら t=25、保存済みなら t=50）
  → 一致したら同じ打ち切り条件（30分／8GB／2000万）で本 run
  → 実測を見て primitive dense solver を最終裁定
  → 縮小世界で Q まで完全一致の検証
  → D0 → D_S-exact → D1a → そこで初めて C
```

## 6. 言ってよいことの境界

言える：reachability 実装について、旧版と新版で support が完全一致し、946倍になった。
言わない：solver が946倍速くなった／World_1-G 自体が8GB必要／状態空間が飽和する／C について何か。

## 7. 記録：今日の失敗

- **continuation memoization（全 suffix 保持）は撤回。** RSS 27GB に達して停止。solver 仕様に書いた8GBの停止条件を、新しい実装に入れ忘れていた（手順違反）。この失敗を受けて watchdog を外側に作った
- **計器の定義を混ぜて報告した**（返答57）。raw と unique を分けずに比率を書き、根拠のない「1万倍以上」という数字を出した。定義を分け直した
- **MODE を9種類と概算していた。** 実測は8種類
