"""KWS 고정구문 probe — 학습 없이 "prototype 게이트가 살아있나"를 판정.

문제의식: hard-neg 마이닝은 false↓ 대가로 recall을 깼다(58%). false의 정체는
잡담/문소리/KTX(=다른 음성)인데, "이번 역은"은 KORAIL 단일녹음의 *고정 파형*이라
'그 고정구문과 닮았나?'를 직접 재면 false를 정면으로 칠 수 있다(메모리: KWS가
"이번역은" 자체를 못 배움). 단 (a) 레퍼런스는 clean(서교공)이 아닌 *live*에서 떠야
하고(clean≠live), (b) raw NCC는 이미 죽었으니 여기선 학습 없는 logmel_cmn 임베딩
거리로 *상한선만* 본다 — 이게 살아있으면 학습 임베딩(metric)은 더 잘 된다.

이 스크립트는 빌드 전 ≤10분 결정 실험이다. 아무것도 학습하지 않는다.

측정 (8-trip leave-one-out, seed 고정):
  held-out 트립 h 마다
    prototype p_h = (다른 7트립의 "이번역은"[onset,+0.7s] logmel_cmn 임베딩 평균)
    positives = h의 진짜 마크 12개의 prototype 코사인
    negatives = h의 pure-noise 세그먼트에서 자른 0.7s 윈도우들의 prototype 코사인
    → ROC-AUC(pos>neg 확률), d-prime, cos gap
  front-end 별로 반복:  none / highpass@250 / preemphasis / hp+preemph
  AUC≈1.0 → prototype 게이트 강하게 viable.  AUC≈0.5 → 죽음(학습 임베딩 필요).
  front-end가 AUC를 올리면 음성용 전처리(idea-1)도 검출기에 가치 있음.

CMN이 학습·추론을 한 모드로 묶는다(D.to_logmel == 펌웨어 melspec). front-end는 raw
오디오에 거는 거라 펌웨어 포팅도 한 줄(고역통과 IIR / preemph 1탭)이다.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import path2_dataset as D
import path2_pipeline as PL

# data lives at the REPO ROOT (parent of path2_fullpipe), not under it — same
# override the _diag_*.py runners do. Point the loaders there.
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PL.LIVE_DIR = os.path.join(_REPO, "data", "raw", "line1_live")
LIVE_DIR = PL.LIVE_DIR
list_trips = PL.list_trips

try:
    import librosa
    from scipy.signal import butter, sosfiltfilt
except Exception as e:                                    # pragma: no cover
    raise SystemExit(f"need librosa+scipy: {e}")

PHRASE_WIN = float(os.environ.get("KWS_PROBE_WIN", "0.7"))   # "이번역은" ~[0,0.7]s
N_NEG = 60                                                    # neg windows / trip
HP_HZ = 250.0                                                # rumble/HVAC 아래
PREEMPH = 0.97
SEED = 0


# --------------------------------------------------------------------------- #
# front-ends (raw audio -> raw audio), then the SAME D.to_logmel as KWS uses
# --------------------------------------------------------------------------- #
_SOS = butter(4, HP_HZ / (D.SR / 2), btype="highpass", output="sos")


def _front(w, mode):
    if mode == "hp":
        return sosfiltfilt(_SOS, w).astype(np.float32)
    if mode == "pre":
        return librosa.effects.preemphasis(w, coef=PREEMPH).astype(np.float32)
    if mode == "hp+pre":
        return librosa.effects.preemphasis(
            sosfiltfilt(_SOS, w).astype(np.float32), coef=PREEMPH).astype(np.float32)
    return w                                                 # "none"


def _vec(y, a, n, mode):
    """logmel_cmn embedding (flattened, L2-normalized) of [a, a+n] under front-end."""
    w = y[a:a + n]
    if len(w) < n:
        w = np.pad(w, (0, n - len(w)))
    w = _front(w, mode)
    v = D.to_logmel(w).ravel().astype(np.float32)
    nrm = np.linalg.norm(v)
    return v / nrm if nrm > 0 else v


def _auc(pos, neg):
    """Mann-Whitney AUC = P(random pos scores > random neg). 0.5 = chance."""
    pos, neg = np.asarray(pos), np.asarray(neg)
    allv = np.concatenate([pos, neg])
    order = allv.argsort()
    ranks = np.empty(len(allv)); ranks[order] = np.arange(1, len(allv) + 1)
    # average ranks for ties
    _, inv, cnt = np.unique(allv, return_inverse=True, return_counts=True)
    csum = np.cumsum(cnt)
    avg = {i: (csum[i] - cnt[i] + csum[i] - 1) / 2 + 1 for i in range(len(cnt))}
    ranks = np.array([avg[i] for i in inv])
    rsum_pos = ranks[:len(pos)].sum()
    u = rsum_pos - len(pos) * (len(pos) + 1) / 2
    return u / (len(pos) * len(neg))


def _dprime(pos, neg):
    pos, neg = np.asarray(pos), np.asarray(neg)
    s = np.sqrt(0.5 * (pos.var() + neg.var())) + 1e-9
    return (pos.mean() - neg.mean()) / s


def run_probe(trips=None, modes=("none", "hp", "pre", "hp+pre")):
    trips = trips or list_trips()
    rng = np.random.default_rng(SEED)
    n = int(PHRASE_WIN * D.SR)
    print(f"KWS 고정구문 probe (학습 없음) | win={PHRASE_WIN}s | {len(trips)} trips, "
          f"trip-LOO | neg/trip={N_NEG}\n")

    # preload trips: phrase windows + negative offsets per front-end
    loaded = []
    for tid in trips:
        t = D.load_live_trip(os.path.join(LIVE_DIR, tid))
        # negative offsets from pure-noise segments (>10s from any announcement)
        cand = []
        for seg in t.noise_segs:
            if len(seg) >= n:
                cand.append(seg)
        loaded.append((tid, t, cand))

    for mode in modes:
        # phrase embedding per trip (list of 12 vecs)
        ph = []
        for tid, t, _ in loaded:
            vs = [_vec(t.y, a, n, mode) for _, a in t.marks]
            ph.append(np.asarray(vs))
        # negative embedding per trip (sample N_NEG windows from noise segs)
        ng = []
        for tid, t, cand in loaded:
            vs = []
            if cand:
                for _ in range(N_NEG):
                    seg = cand[rng.integers(len(cand))]
                    a = int(rng.integers(0, len(seg) - n + 1))
                    vs.append(D.to_logmel(_front(seg[a:a + n], mode)).ravel())
            vs = np.asarray(vs, np.float32)
            vs = vs / (np.linalg.norm(vs, axis=1, keepdims=True) + 1e-9)
            ng.append(vs)

        aucs, dps, gaps, lines = [], [], [], []
        for h in range(len(trips)):
            proto = np.concatenate([ph[k] for k in range(len(trips)) if k != h]).mean(0)
            proto = proto / (np.linalg.norm(proto) + 1e-9)
            pos = ph[h] @ proto
            neg = ng[h] @ proto
            a, d = _auc(pos, neg), _dprime(pos, neg)
            aucs.append(a); dps.append(d); gaps.append(pos.mean() - neg.mean())
            lines.append(f"    {trips[h][9:13]} {loaded[h][1].direction}: "
                         f"AUC {a:.2f}  d' {d:+.2f}  cos pos {pos.mean():.3f} / "
                         f"neg {neg.mean():.3f}")
        print(f"[{mode:7s}]  meanAUC {np.mean(aucs):.3f}  d' {np.mean(dps):+.2f}  "
              f"cos gap {np.mean(gaps):+.3f}")
        for ln in lines:
            print(ln)
        print()

    print("verdict: meanAUC >=0.95 -> prototype gate viable now (cut false, keep recall).")
    print("         ~0.5 -> dead with raw embedding = need a LEARNED (metric) embedding.")
    print("         front-end that raises AUC -> speech pre-proc also helps the detector.")


# --------------------------------------------------------------------------- #
# REAL-false probe — negative = the CNN KWS's ACTUAL false triggers (잡담/KTX),
# not random noise. This is the honest test of "does the fixed-phrase prototype
# kill the false that mining couldn't, without touching recall?"
#   per held-out trip: train baseline KWS on the other 7 (== hardneg LOO recipe),
#   slide-detect on held-out, split triggers into TP(matched mark) / FP(false),
#   score every trigger window by cosine to the live prototype, report
#   AUC(TP vs FP) + an operating point that keeps ALL TP (recall untouched).
# --------------------------------------------------------------------------- #
import path2_poc as P                                       # noqa: E402

CLEAN_DIR = os.path.join(_REPO, "data", "processed", "wav")
PL.CLEAN_DIR = CLEAN_DIR
MATCH_S = 10.0                                              # == _kws_recall_false


def _classify_triggers(trg, marks):
    """mark-centric greedy match (== _kws_recall_false). Return TP/FP trigger times."""
    mt = [idx / D.SR for _, idx in marks]
    used = [False] * len(trg)
    for m in mt:
        best = None
        for i, t in enumerate(trg):
            if used[i] or abs(t - m) > MATCH_S:
                continue
            if best is None or abs(t - m) < abs(trg[best] - m):
                best = i
        if best is not None:
            used[best] = True
    tp = [trg[i] for i in range(len(trg)) if used[i]]
    fp = [trg[i] for i in range(len(trg)) if not used[i]]
    return tp, fp


def run_real_false(trips=None):
    trips = trips or list_trips()
    clean = D.load_clean_sources(CLEAN_DIR)
    n = int(PHRASE_WIN * D.SR)
    prev = D.FEATURE_MODE; D.FEATURE_MODE = "logmel_cmn"
    print(f"KWS real-false probe | win={PHRASE_WIN}s | {len(trips)} trips, trip-LOO\n"
          f"  negative = CNN KWS's actual false triggers (NOT random noise)\n")

    # phrase embedding per trip (for the prototype)
    loaded = [D.load_live_trip(os.path.join(LIVE_DIR, t)) for t in trips]
    ph = [np.asarray([_vec(t.y, a, n, "none") for _, a in t.marks]) for t in loaded]

    aucs = []
    f_before = f_after = tp_total = tp_kept = 0
    for h, held in enumerate(trips):
        tr = [loaded[k] for k in range(len(trips)) if k != h]
        test = loaded[h]
        Xk, Yk = D.build_kws(clean, tr, np.random.default_rng(0), snr=(0.0, 25.0),
                             spec_aug=False)
        model, norm, _ = P.train(Xk, Yk, 2, epochs=40, lr=5e-4, patience=12)
        trg = PL.kws_triggers(test.y, (model, norm))
        tp, fp = _classify_triggers(trg, test.marks)

        proto = np.concatenate([ph[k] for k in range(len(trips)) if k != h]).mean(0)
        proto = proto / (np.linalg.norm(proto) + 1e-9)
        s_tp = np.array([_vec(test.y, int(t * D.SR), n, "none") @ proto for t in tp])
        s_fp = np.array([_vec(test.y, int(t * D.SR), n, "none") @ proto for t in fp])

        a = _auc(s_tp, s_fp) if len(s_tp) and len(s_fp) else float("nan")
        # operating point: keep ALL TP (recall untouched) -> tau = min TP score.
        tau = s_tp.min() if len(s_tp) else -1.0
        killed = int((s_fp < tau).sum())
        if not np.isnan(a):
            aucs.append(a)
        f_before += len(fp); f_after += len(fp) - killed
        tp_total += len(tp); tp_kept += len(tp)            # all kept by construction
        print(f"  {held[9:13]} {test.direction}: TP {len(tp):2d} FP {len(fp):2d} | "
              f"AUC(TP>FP) {a:.2f} | cos TP {s_tp.mean() if len(s_tp) else 0:.3f} / "
              f"FP {s_fp.mean() if len(s_fp) else 0:.3f} | keep-all-TP gate kills "
              f"{killed}/{len(fp)} FP")
    D.FEATURE_MODE = prev
    T = len(trips)
    print("\n" + "=" * 60)
    print(f"  mean AUC(TP vs real-FP) : {np.nanmean(aucs):.3f}   (random-noise был 0.731)")
    print(f"  false/trip  before gate : {f_before / T:.1f}")
    print(f"  false/trip  after  gate : {f_after / T:.1f}   (recall 유지: TP {tp_kept}/{tp_total})")
    print("\n  AUC>0.8 & false 큰 폭↓ → prototype 게이트가 마이닝 대체(recall 무손실).")
    print("  AUC~0.6 → 음성 negative엔 약함 = 학습 임베딩으로 끌어올려야.")


if __name__ == "__main__":
    if "random" in sys.argv:
        run_probe()
    else:
        run_real_false()
