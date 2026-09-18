# -*- coding: utf-8 -*-
"""実寸 World_1-G の遷移表を2パスで作り、保存する。
★physics は world1g のまま（probe_kernel_size.branches_with_prob を使う）★

第1パス：各 (state, action) の枝の数だけ数える（配列を作らない＝軽い）
第2パス：正確な総数で np.empty を確保し、直接書き込む（Python のリストを経由しない）
最後に assert で書き込み位置が一致することを確認し、保存＋manifest＋digest。

dtype（凍結）：index=uint32 / 確率=float64 / 価値=float64
  ★float32 には落とさない（新しい数値誤差要因を持ち込まない）★
"""
import os, sys, time, json, hashlib
import numpy as np
import world1g as w
from world1g import State, DECISION, E_MAX
import dense_index as dx
from probe_kernel_size import branches_with_prob
import psutil

_proc = psutil.Process(os.getpid())
ACT_ID = {None: 0, w.ACT_GRASS: 1, w.ACT_SMALL: 2, w.ACT_LARGE: 3}
CAUSE_ID = {None: 0, "starve": 1, "acute": 2, "tenju": 3}
OUT_DIR = "kernel_full"


def rss():
    return _proc.memory_info().rss / 1024**3


def iter_pairs():
    """structural-valid な (state, action) を順に返す。★順序を固定する★"""
    for mi, mode in enumerate(dx.MODES):
        big = mode in (w.SEARCH_LARGE, w.COMBAT)
        for e in range(1, E_MAX + 1):
            for wnd in range(11):
                for ns in range(1, 13):
                    for nl in range(1 if big else 0, 4):
                        if mode == DECISION:
                            s = State(e=e, w=wnd, n_small=ns, n_large=nl,
                                      t=0, mode=DECISION)
                            for a in w.legal_actions(s):
                                yield (mode, e, wnd, ns, nl, a)
                        else:
                            yield (mode, e, wnd, ns, nl, None)


def pass1():
    """第1パス：枝の数だけ数える"""
    print("【第1パス】枝の数を正確に数える（配列を作らない）", flush=True)
    t0 = time.perf_counter()
    n_pairs = 0
    n_nt = 0
    n_tm = 0
    cnt_nt = []
    cnt_tm = []
    for (mode, e, wnd, ns, nl, a) in iter_pairs():
        nt, tm = branches_with_prob(mode, e, wnd, ns, nl, a)
        cnt_nt.append(len(nt))
        cnt_tm.append(len(tm))
        n_nt += len(nt)
        n_tm += len(tm)
        n_pairs += 1
        if n_pairs % 2_000_000 == 0:
            print(f"  {n_pairs:>10,} 組  非終端 {n_nt:>12,}  終端 {n_tm:>11,}"
                  f"  RSS {rss():.2f}GB  {time.perf_counter()-t0:.0f}秒", flush=True)
    el = time.perf_counter() - t0
    print(f"  ★完了★ {n_pairs:,} 組 / 非終端 {n_nt:,} / 終端 {n_tm:,}"
          f" / {el:.0f}秒 / RSS {rss():.2f}GB")
    return (np.array(cnt_nt, dtype=np.int32), np.array(cnt_tm, dtype=np.int32),
            n_pairs, n_nt, n_tm, el)


def pass2(cnt_nt, cnt_tm, n_pairs, n_nt, n_tm):
    """第2パス：事前確保した配列に直接書き込む"""
    print("【第2パス】事前確保して直接書き込む", flush=True)
    t0 = time.perf_counter()
    offs = np.zeros(n_pairs + 1, dtype=np.int64)
    np.cumsum(cnt_nt, out=offs[1:])
    t_offs = np.zeros(n_pairs + 1, dtype=np.int64)
    np.cumsum(cnt_tm, out=t_offs[1:])
    succ = np.empty(n_nt, dtype=np.uint32)
    prob = np.empty(n_nt, dtype=np.float64)
    t_prob = np.empty(n_tm, dtype=np.float64)
    t_cause = np.empty(n_tm, dtype=np.uint8)
    t_rawe = np.empty(n_tm, dtype=np.int32)
    state_idx = np.empty(n_pairs, dtype=np.uint32)
    act_id = np.empty(n_pairs, dtype=np.uint8)
    print(f"  確保した配列の合計 "
          f"{(offs.nbytes+t_offs.nbytes+succ.nbytes+prob.nbytes+t_prob.nbytes+t_cause.nbytes+t_rawe.nbytes+state_idx.nbytes+act_id.nbytes)/1024**3:.3f} GB"
          f"  RSS {rss():.2f}GB", flush=True)
    k, p1, p2 = 0, 0, 0
    for (mode, e, wnd, ns, nl, a) in iter_pairs():
        nt, tm = branches_with_prob(mode, e, wnd, ns, nl, a)
        state_idx[k] = dx.to_index(mode, e, wnd, ns, nl)
        act_id[k] = ACT_ID[a]
        for j, pr in nt:
            succ[p1] = j
            prob[p1] = pr
            p1 += 1
        for pr, cause, raw in tm:
            t_prob[p2] = pr
            t_cause[p2] = CAUSE_ID[cause]
            t_rawe[p2] = raw
            p2 += 1
        k += 1
        if k % 2_000_000 == 0:
            print(f"  {k:>10,} 組  RSS {rss():.2f}GB"
                  f"  {time.perf_counter()-t0:.0f}秒", flush=True)
    el = time.perf_counter() - t0
    # ★assert：書き込み位置が第1パスの数と一致するか★
    assert k == n_pairs, f"組の数が違う: {k} != {n_pairs}"
    assert p1 == n_nt, f"非終端の枝の数が違う: {p1} != {n_nt}"
    assert p2 == n_tm, f"終端の枝の数が違う: {p2} != {n_tm}"
    assert offs[-1] == n_nt and t_offs[-1] == n_tm
    print(f"  ★完了★ assert 通過 / {el:.0f}秒 / RSS {rss():.2f}GB")
    return dict(offs=offs, t_offs=t_offs, succ=succ, prob=prob,
                t_prob=t_prob, t_cause=t_cause, t_rawe=t_rawe,
                state_idx=state_idx, act_id=act_id), el


def save(arrs, manifest):
    os.makedirs(OUT_DIR, exist_ok=True)
    h = hashlib.sha256()
    for nm in sorted(arrs):
        path = os.path.join(OUT_DIR, nm + ".npy")
        np.save(path, arrs[nm])
        h.update(arrs[nm].tobytes())
        manifest["arrays"][nm] = dict(shape=list(arrs[nm].shape),
                                      dtype=str(arrs[nm].dtype),
                                      bytes=int(arrs[nm].nbytes))
    manifest["sha256_all_arrays"] = h.hexdigest()
    with open(os.path.join(OUT_DIR, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(f"  保存先 {OUT_DIR}/  digest {manifest['sha256_all_arrays'][:16]}")


def main():
    print("実寸 World_1-G の遷移表を2パスで作る")
    print(f"index の総数 {dx.TOTAL:,} ／ dtype: index=uint32, 確率=float64")
    print()
    cnt_nt, cnt_tm, n_pairs, n_nt, n_tm, el1 = pass1()
    print()
    arrs, el2 = pass2(cnt_nt, cnt_tm, n_pairs, n_nt, n_tm)
    print()
    tot = sum(a.nbytes for a in arrs.values())
    print(f"【容量】合計 {tot/1024**3:.3f} GB")
    for nm in sorted(arrs):
        print(f"  {nm:10s} {arrs[nm].nbytes/1024**3:7.3f} GB  {arrs[nm].dtype}")
    print()
    manifest = dict(
        world="World_1-G",
        world_spec_commit="386bf76",
        engine_commit="1cde347",
        n_pairs=int(n_pairs), n_nonterminal=int(n_nt), n_terminal=int(n_tm),
        build_sec_pass1=round(el1, 1), build_sec_pass2=round(el2, 1),
        dtypes=dict(index="uint32", probability="float64", value="float64"),
        arrays={})
    save(arrs, manifest)
    print(f"  総 RSS {rss():.2f}GB")


if __name__ == "__main__":
    main()
