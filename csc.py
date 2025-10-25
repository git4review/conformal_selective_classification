
import argparse, os, math, json
import numpy as np
import pandas as pd

def softmax_np(z, T=1.0):
    z = z / max(1e-12, T)
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)

def msp_from_logits(logits):
    p = softmax_np(logits); return p.max(axis=1)

def margin_from_logits(logits):
    p = softmax_np(logits)
    top2 = np.partition(p, -2, axis=1)[:, -2:]
    return top2[:,1] - top2[:,0]

def entropy_from_logits(logits):
    p = softmax_np(logits); eps=1e-12
    return -(p*np.log(p+eps)).sum(axis=1)

def energy_from_logits(logits):
    m = logits.max(axis=1, keepdims=True)
    lse = m + np.log(np.exp(logits - m).sum(axis=1, keepdims=True))
    return (-lse).ravel()

def build_scores(logits_cal, logits_test):
    return {
        "msp":     (msp_from_logits(logits_cal),     msp_from_logits(logits_test)),
        "margin":  (margin_from_logits(logits_cal),  margin_from_logits(logits_test)),
        "entropy": (entropy_from_logits(logits_cal), entropy_from_logits(logits_test)),
        "energy":  (energy_from_logits(logits_cal),  energy_from_logits(logits_test)),
    }

def rank_confidence(cal_scores, scores, lower_is_better):
    cal_sorted = np.sort(cal_scores)
    ranks_cal = np.searchsorted(cal_sorted, cal_scores, side="right")
    u_cal = ranks_cal / (len(cal_sorted)+1.0)
    conf_cal = (1.0 - u_cal) if lower_is_better else u_cal
    ranks = np.searchsorted(cal_sorted, scores, side="right")
    u = ranks / (len(cal_sorted)+1.0)
    conf = (1.0 - u) if lower_is_better else u
    return conf_cal, conf

def coverage_lambda1_hat(conf_cal, xi):
    t = np.quantile(conf_cal, 1 - xi, method='higher') \
        if hasattr(np,"quantile") else np.percentile(conf_cal, 100*(1-xi))
    return float(1.0 - t)

def coverage_lambda1_hat_transductive(conf_cal, conf_test_point, xi):
    combo = np.concatenate([conf_cal, np.asarray([conf_test_point])])
    t = np.quantile(combo, 1 - xi, method='higher') \
        if hasattr(np,"quantile") else np.percentile(combo, 100*(1-xi))
    return float(1.0 - t)

def stratified_sample_idx(y, m, seed=0, replace=False):
    if m<=0 or m>=len(y): return np.arange(len(y))
    rng = np.random.default_rng(seed)
    idxs=[]; classes, counts = np.unique(y, return_counts=True)
    probs = counts / counts.sum()
    targets = np.round(probs*m).astype(int)
    diff = m - targets.sum()
    for _ in range(abs(diff)):
        j = np.argmax(probs) if diff>0 else np.argmax(targets)
        targets[j] += 1 if diff>0 else -1
    for c,k in zip(classes, targets):
        cand = np.where(y==c)[0]
        if len(cand)==0 or k<=0: continue
        choice = rng.choice(cand, size=k, replace=replace)
        idxs.append(choice)
    return np.concatenate(idxs) if idxs else np.array([],dtype=int)

def inclusion_indicator(probs,y,lam2,constructor):
    if constructor=="prob_thresh":
        set_sizes=(probs>=lam2).sum(axis=1)
        inc=(probs[np.arange(len(y)),y]>=lam2)
        return inc.astype(bool), set_sizes.astype(int)
    elif constructor in ("cumprob","topk"):
        order=np.argsort(-probs,axis=1)
        sorted_p=-np.sort(-probs,axis=1)
        cumsum=np.cumsum(sorted_p,axis=1)
        k=(cumsum<lam2).sum(axis=1)+1
        n,K=probs.shape
        ranks=np.empty((n,K),dtype=int)
        rows=np.arange(n)[:,None]
        ranks[rows,order]=np.arange(K)[None,:]
        y_rank=ranks[np.arange(n),y]+1
        inc=(y_rank<=k)
        return inc.astype(bool), k.astype(int)
    elif constructor=="ordinal":
        n,K=probs.shape
        centers = probs.argmax(axis=1)
        set_sizes = np.ones(n, dtype=int)
        inc = (y == centers)
        cum = probs[np.arange(n), centers].copy()
        left = centers - 1
        right = centers + 1
        more = (cum < lam2)
        while np.any(more):
            left_p = np.where((left>=0), probs[np.arange(n), np.clip(left,0,K-1)], -np.inf)
            right_p = np.where((right<K), probs[np.arange(n), np.clip(right,0,K-1)], -np.inf)
            choose_left = (left_p >= right_p) & more
            choose_right = (~choose_left) & more
            if np.any(choose_left):
                idx = np.where(choose_left)[0]
                cum[idx] += probs[idx, left[idx]]
                inc[idx] |= (y[idx] == left[idx])
                left[idx] -= 1
                set_sizes[idx] += 1
            if np.any(choose_right):
                idx = np.where(choose_right)[0]
                cum[idx] += probs[idx, right[idx]]
                inc[idx] |= (y[idx] == right[idx])
                right[idx] += 1
                set_sizes[idx] += 1
            more = (cum < lam2) & ((left>=0) | (right<K))
        return inc.astype(bool), set_sizes.astype(int)
    else:
        raise ValueError("Unknown constructor")

def loss_miscoverage(probs, y, lam2, constructor):
    inc,_=inclusion_indicator(probs,y,lam2,constructor)
    return (~inc).astype(float)

def feasible_EXG(cal_conf, probs_cal, y_cal, lam1, lam2, constructor, alpha):
    sel = (cal_conf >= (1.0 - lam1))
    probs_sel = probs_cal[sel]; y_sel = y_cal[sel]
    m = int(len(y_sel))
    if m == 0:
        return False, {"reason":"no_selected"}
    losses = loss_miscoverage(probs_sel, y_sel, lam2, constructor)
    s = float(losses.sum())
    B = math.ceil((m + 1) * alpha) - 1
    feasible = (s <= B + 1e-12)
    return feasible, {"m": m, "sum_loss": s, "B": int(B)}

def dkw_epsilon(n, delta):
    return math.sqrt(max(0.0, math.log(2.0/max(1e-12,delta)) / (2.0*max(1,n))))

def feasible_CO(cal_conf, probs_cal, y_cal, lam1, lam2, constructor, alpha, delta):
    n = len(y_cal)
    S = (cal_conf >= (1.0 - lam1)).astype(int)
    xi_hat = float(S.mean())
    eps = dkw_epsilon(n, delta)
    xi_LCB = max(0.0, xi_hat - eps)
    B = math.ceil((n + 1) * alpha * xi_LCB) - 1
    if B < 0:
        return False, {"xi_hat": xi_hat, "xi_LCB": xi_LCB, "B": int(B)}
    tilde = S * loss_miscoverage(probs_cal, y_cal, lam2, constructor)
    s = float(tilde.sum())
    feasible = (s <= B + 1e-12)
    return feasible, {"xi_hat": xi_hat, "xi_LCB": xi_LCB, "B": int(B), "sum_tilde": s}

def feasible_CRC_ALL(probs_cal, y_cal, lam2, constructor, alpha):
    m = len(y_cal)
    if m == 0: return False, {"reason":"empty_cal"}
    losses = loss_miscoverage(probs_cal, y_cal, lam2, constructor)
    s = float(losses.sum())
    B = math.ceil((m + 1) * alpha) - 1
    feasible = (s <= B + 1e-12)
    return feasible, {"m": m, "sum_loss": s, "B": int(B)}

def binary_search_lambda2(check_fn, lo, hi, tol, dir_up=True, max_it=30):
    if dir_up:
        ok_hi,info_hi = check_fn(hi)
        if not ok_hi: return float('nan'), {"reason":"infeasible_hi"}, False
        L,R=lo,hi; ans=None; info_best=None
        for _ in range(max_it):
            mid=0.5*(L+R); ok,info=check_fn(mid)
            if ok: ans=mid; info_best=info; R=mid
            else: L=mid
            if R-L<tol: break
        if ans is None: return (hi, info_hi, True)
        return float(ans), info_best, True
    else:
        ok_lo,info_lo = check_fn(lo)
        if not ok_lo: return float('nan'), {"reason":"infeasible_lo"}, False
        L,R=lo,hi; ans=None; info_best=None
        for _ in range(max_it):
            mid=0.5*(L+R); ok,info=check_fn(mid)
            if ok: ans=mid; info_best=info; L=mid
            else: R=mid
            if R-L<tol: break
        if ans is None: return (lo, info_lo, True)
        return float(ans), info_best, True

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--npz_path",required=True)
    ap.add_argument("--constructor",nargs="+",default=["cumprob"],choices=["prob_thresh","cumprob","topk","ordinal"])
    ap.add_argument("--method",nargs="+",default=["EXG","CO","CRC_ALL","RAND","EXG_TX"])
    ap.add_argument("--runs",type=int,default=50)
    ap.add_argument("--cal_size",type=int,default=5000)
    ap.add_argument("--test_size",type=int,default=0)
    ap.add_argument("--seed0",type=int,default=0)
    ap.add_argument("--xis",nargs="+",type=float,default=[0.8])
    ap.add_argument("--alphas",nargs="+",type=float,default=[0.1])
    ap.add_argument("--deltas",nargs="+",type=float,default=[0.05])
    ap.add_argument("--lambda1_grid",type=int,default=11)
    ap.add_argument("--lambda2_min",type=float,default=1e-4)
    ap.add_argument("--lambda2_max",type=float,default=0.99)
    ap.add_argument("--lambda2_tol",type=float,default=1e-3)
    ap.add_argument("--select_best",type=int,default=1)
    ap.add_argument("--rand_mode",type=str,default="xi",choices=["xi","grid"])
    ap.add_argument("--out_csv",default="./artifacts/twostage_runs_raw.csv")
    ap.add_argument("--out_summary",default="./artifacts/twostage_summary.csv")
    args=ap.parse_args()

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)

    data=np.load(args.npz_path)
    y_cal_full=data["y_cal_pool"] if "y_cal_pool" in data else data["y_cal"]
    y_test_full=data["y_test"]
    logits_test_full=data["logits_test"]
    logits_cal_full=data["logits_cal_pool"] if "logits_cal_pool" in data else data.get("logits_cal")
    if logits_cal_full is None: raise RuntimeError("Need logits for calibration.")

    scores = build_scores(logits_cal_full, logits_test_full)
    lower_names={"entropy","energy"}
    probs_test_full=softmax_np(logits_test_full)
    probs_cal_full=softmax_np(logits_cal_full)

    rows=[]

    for run in range(args.runs):
        seed=args.seed0+run
        idx_cal=stratified_sample_idx(y_cal_full,m=args.cal_size,seed=seed)
        y_cal=y_cal_full[idx_cal]; probs_cal=probs_cal_full[idx_cal]
        idx_test=stratified_sample_idx(y_test_full,m=args.test_size,seed=seed+10_000)
        y_test=y_test_full[idx_test]; probs_test=probs_test_full[idx_test]

        for sname,(s_cal_full,s_test_full) in scores.items():
            s_cal=s_cal_full[idx_cal]; s_test=s_test_full[idx_test]
            lower=(sname in lower_names)
            conf_cal,conf_test=rank_confidence(s_cal,s_test,lower)

            for xi in args.xis:
                lam1_hat=coverage_lambda1_hat(conf_cal,xi)
                lam1_grid=np.array([lam1_hat]) if args.lambda1_grid<=1 else np.linspace(lam1_hat,1.0,args.lambda1_grid)

                for lam1 in lam1_grid:
                    sel_cal=(conf_cal>=(1.0 - lam1)); sel_test=(conf_test>=(1.0 - lam1))
                    cov_cal=float(sel_cal.mean()); cov_test=float(sel_test.mean())

                    for alpha in args.alphas:
                        for constructor in args.constructor:
                            dir_up = (constructor in ("cumprob","topk","ordinal"))
                            def chk_exg(l2): return feasible_EXG(conf_cal, probs_cal, y_cal, lam1, l2, constructor, alpha)
                            def make_chk_co(delta):
                                def chk(l2): return feasible_CO(conf_cal, probs_cal, y_cal, lam1, l2, constructor, alpha, delta)
                                return chk
                            def chk_crc_all(l2): return feasible_CRC_ALL(probs_cal, y_cal, l2, constructor, alpha)

                            rng = np.random.default_rng(seed+4242)
                            u_cal = rng.random(len(y_cal)); u_test = rng.random(len(y_test))

                            for method in args.method:
                                if method=="EXG":
                                    lam2,info,feas = binary_search_lambda2(chk_exg, args.lambda2_min, args.lambda2_max, args.lambda2_tol, dir_up=dir_up, max_it=30)
                                    sel_cal_m=sel_cal; sel_test_m=sel_test
                                    cov_cal_m=cov_cal; cov_test_m=cov_test
                                    lam1_used=lam1

                                    if np.isfinite(lam2):
                                        loss_cal = loss_miscoverage(probs_cal[sel_cal_m], y_cal[sel_cal_m], lam2, constructor)
                                        risk_cal = float(loss_cal.mean()) if len(loss_cal)>0 else float("nan")
                                        _, set_sizes_cal = inclusion_indicator(probs_cal[sel_cal_m], y_cal[sel_cal_m], lam2, constructor)
                                        set_size_cal_mean = float(set_sizes_cal.mean()) if len(set_sizes_cal)>0 else float("nan")

                                        loss_test = loss_miscoverage(probs_test[sel_test_m], y_test[sel_test_m], lam2, constructor)
                                        risk_test = float(loss_test.mean()) if len(loss_test)>0 else float("nan")
                                        _, set_sizes_test = inclusion_indicator(probs_test[sel_test_m], y_test[sel_test_m], lam2, constructor)
                                        set_size_test_mean = float(set_sizes_test.mean()) if len(set_sizes_test)>0 else float("nan")
                                    else:
                                        risk_cal=float("nan"); set_size_cal_mean=float("nan")
                                        risk_test=float("nan"); set_size_test_mean=float("nan")

                                    rows.append(dict(run=run,score=sname,method="EXG",constructor=constructor,
                                                     lambda1=float(lam1_used),
                                                     lambda2=float(lam2) if np.isfinite(lam2) else np.nan,
                                                     xi=xi,alpha=alpha,
                                                     cov_cal=cov_cal_m,cov_test=cov_test_m,
                                                     risk_cal=risk_cal,risk_test=risk_test,
                                                     set_size_cal_mean=set_size_cal_mean,
                                                     set_size_mean=set_size_test_mean,
                                                     feasible=int(bool(feas)),
                                                     test_size=int(len(y_test))))

                                elif method=="EXG_TX":
                                    ntest = len(y_test)
                                    set_sizes = []
                                    losses = []
                                    sel_flags = []
                                    lam2_list = []
                                    lam1_list = []
                                    for j in range(ntest):
                                        lam1_j = coverage_lambda1_hat_transductive(conf_cal, conf_test[j], xi)
                                        lam1_list.append(lam1_j)
                                        def chk_exg_j(l2):
                                            return feasible_EXG(conf_cal, probs_cal, y_cal, lam1_j, l2, constructor, alpha)
                                        lam2_j,info,feas = binary_search_lambda2(chk_exg_j, args.lambda2_min, args.lambda2_max, args.lambda2_tol, dir_up=dir_up, max_it=30)
                                        lam2_list.append(lam2_j if np.isfinite(lam2_j) else np.nan)
                                        sel_j = (conf_test[j] >= (1.0 - lam1_j))
                                        sel_flags.append(bool(sel_j))
                                        if np.isfinite(lam2_j) and sel_j:
                                            inc_j, k_j = inclusion_indicator(probs_test[j:j+1], y_test[j:j+1], lam2_j, constructor)
                                            losses.append(float((~inc_j).astype(float).mean()))
                                            set_sizes.append(int(k_j.mean()))
                                        elif sel_j:
                                            losses.append(float("nan"))
                                            set_sizes.append(float("nan"))
                                        else:
                                            losses.append(float("nan"))
                                            set_sizes.append(float("nan"))
                                    sel_flags = np.array(sel_flags, dtype=bool)
                                    cov_test_m = float(sel_flags.mean())
                                    if np.any(sel_flags):
                                        risk_test = float(np.nanmean(np.array(losses)[sel_flags]))
                                        set_size_test_mean = float(np.nanmean(np.array(set_sizes)[sel_flags]))
                                    else:
                                        risk_test = float("nan"); set_size_test_mean = float("nan")
                                    cov_cal_list = []
                                    for lam1_j in lam1_list:
                                        cov_cal_list.append(float((conf_cal >= (1.0 - lam1_j)).mean()))
                                    cov_cal_m = float(np.mean(cov_cal_list))
                                    rows.append(dict(run=run,score=sname,method="EXG_TX",constructor=constructor,
                                                     lambda1=float(np.mean(lam1_list)),
                                                     lambda2=float(np.nanmean(np.array(lam2_list))),
                                                     xi=xi,alpha=alpha,
                                                     cov_cal=cov_cal_m,cov_test=cov_test_m,
                                                     risk_cal=float("nan"),risk_test=risk_test,
                                                     set_size_cal_mean=float("nan"),
                                                     set_size_mean=set_size_test_mean,
                                                     feasible=1,
                                                     test_size=int(len(y_test))))

                                elif method=="CO":
                                    for delta in args.deltas:
                                        def chk_co(l2): return feasible_CO(conf_cal, probs_cal, y_cal, lam1, l2, constructor, alpha, delta)
                                        lam2,info,feas = binary_search_lambda2(chk_co, args.lambda2_min, args.lambda2_max, args.lambda2_tol, dir_up=dir_up, max_it=30)
                                        sel_cal_m=sel_cal; sel_test_m=sel_test
                                        cov_cal_m=cov_cal; cov_test_m=cov_test
                                        lam1_used=lam1
                                        if np.isfinite(lam2):
                                            loss_cal = loss_miscoverage(probs_cal[sel_cal_m], y_cal[sel_cal_m], lam2, constructor)
                                            risk_cal = float(loss_cal.mean()) if len(loss_cal)>0 else float("nan")
                                            _, set_sizes_cal = inclusion_indicator(probs_cal[sel_cal_m], y_cal[sel_cal_m], lam2, constructor)
                                            set_size_cal_mean = float(set_sizes_cal.mean()) if len(set_sizes_cal)>0 else float("nan")

                                            loss_test = loss_miscoverage(probs_test[sel_test_m], y_test[sel_test_m], lam2, constructor)
                                            risk_test = float(loss_test.mean()) if len(loss_test)>0 else float("nan")
                                            _, set_sizes_test = inclusion_indicator(probs_test[sel_test_m], y_test[sel_test_m], lam2, constructor)
                                            set_size_test_mean = float(set_sizes_test.mean()) if len(set_sizes_test)>0 else float("nan")
                                        else:
                                            risk_cal=float("nan"); set_size_cal_mean=float("nan")
                                            risk_test=float("nan"); set_size_test_mean=float("nan")

                                        rows.append(dict(run=run,score=sname,method="CO",constructor=constructor,
                                                         lambda1=float(lam1_used),
                                                         lambda2=float(lam2) if np.isfinite(lam2) else np.nan,
                                                         xi=xi,alpha=alpha,delta=float(delta),
                                                         cov_cal=cov_cal_m,cov_test=cov_test_m,
                                                         risk_cal=risk_cal,risk_test=risk_test,
                                                         set_size_cal_mean=set_size_cal_mean,
                                                         set_size_mean=set_size_test_mean,
                                                         feasible=int(bool(feas)),
                                                         test_size=int(len(y_test))))

                                elif method=="CRC_ALL":
                                    def chk_crc_all(l2): return feasible_CRC_ALL(probs_cal, y_cal, l2, constructor, alpha)
                                    lam2,info,feas = binary_search_lambda2(chk_crc_all, args.lambda2_min, args.lambda2_max, args.lambda2_tol, dir_up=dir_up, max_it=30)
                                    sel_cal_m = np.ones_like(y_cal, dtype=bool)
                                    sel_test_m = np.ones_like(y_test, dtype=bool)
                                    cov_cal_m = 1.0; cov_test_m = 1.0
                                    lam1_used=1.0
                                    if np.isfinite(lam2):
                                        loss_test = loss_miscoverage(probs_test[sel_test_m], y_test[sel_test_m], lam2, constructor)
                                        risk_test = float(loss_test.mean()) if len(loss_test)>0 else float("nan")
                                        _, set_sizes_test = inclusion_indicator(probs_test[sel_test_m], y_test[sel_test_m], lam2, constructor)
                                        set_size_test_mean = float(set_sizes_test.mean()) if len(set_sizes_test)>0 else float("nan")
                                        loss_cal = loss_miscoverage(probs_cal[sel_cal_m], y_cal[sel_cal_m], lam2, constructor)
                                        risk_cal = float(loss_cal.mean())
                                        _, set_sizes_cal = inclusion_indicator(probs_cal[sel_cal_m], y_cal[sel_cal_m], lam2, constructor)
                                        set_size_cal_mean = float(set_sizes_cal.mean())
                                    else:
                                        risk_cal=float("nan"); set_size_cal_mean=float("nan")
                                        risk_test=float("nan"); set_size_test_mean=float("nan")
                                    rows.append(dict(run=run,score=sname,method="CRC_ALL",constructor=constructor,
                                                     lambda1=float(lam1_used),
                                                     lambda2=float(lam2) if np.isfinite(lam2) else np.nan,
                                                     xi=xi,alpha=alpha,
                                                     cov_cal=cov_cal_m,cov_test=cov_test_m,
                                                     risk_cal=risk_cal,risk_test=risk_test,
                                                     set_size_cal_mean=set_size_cal_mean,
                                                     set_size_mean=set_size_test_mean,
                                                     feasible=int(bool(feas)),
                                                     test_size=int(len(y_test))))

                                elif method=="RAND":
                                    if args.rand_mode=="xi":
                                        sel_cal_m = (u_cal < xi)
                                        sel_test_m = (u_test < xi)
                                        lam1_used = xi
                                        cov_cal_m = float(sel_cal_m.mean()); cov_test_m = float(sel_test_m.mean())
                                        def chk_rand(l2):
                                            conf_rand = sel_cal_m.astype(float)
                                            return feasible_EXG(conf_rand, probs_cal, y_cal, 1.0, l2, constructor, alpha)
                                        lam2,info,feas = binary_search_lambda2(chk_rand, args.lambda2_min, args.lambda2_max, args.lambda2_tol, dir_up=dir_up, max_it=30)
                                    else:
                                        lam1_eff = lam1
                                        conf_cal_rand=u_cal; conf_test_rand=u_test
                                        def chk_exg_rand(l2):
                                            return feasible_EXG(conf_cal_rand, probs_cal, y_cal, lam1_eff, l2, constructor, alpha)
                                        lam2,info,feas = binary_search_lambda2(chk_exg_rand, args.lambda2_min, args.lambda2_max, args.lambda2_tol, dir_up=dir_up, max_it=30)
                                        sel_cal_m = (conf_cal_rand >= (1.0 - lam1_eff))
                                        sel_test_m = (conf_test_rand >= (1.0 - lam1_eff))
                                        cov_cal_m = float(sel_cal_m.mean()); cov_test_m = float(sel_test_m.mean())
                                        lam1_used = lam1_eff

                                    if np.isfinite(lam2):
                                        loss_test = loss_miscoverage(probs_test[sel_test_m], y_test[sel_test_m], lam2, constructor)
                                        risk_test = float(loss_test.mean()) if len(loss_test)>0 else float("nan")
                                        _, set_sizes_test = inclusion_indicator(probs_test[sel_test_m], y_test[sel_test_m], lam2, constructor)
                                        set_size_test_mean = float(set_sizes_test.mean()) if len(set_sizes_test)>0 else float("nan")
                                        loss_cal = loss_miscoverage(probs_cal[sel_cal_m], y_cal[sel_cal_m], lam2, constructor)
                                        risk_cal = float(loss_cal.mean())
                                        _, set_sizes_cal = inclusion_indicator(probs_cal[sel_cal_m], y_cal[sel_cal_m], lam2, constructor)
                                        set_size_cal_mean = float(set_sizes_cal.mean())
                                    else:
                                        risk_cal=float("nan"); set_size_cal_mean=float("nan")
                                        risk_test=float("nan"); set_size_test_mean=float("nan")

                                    rows.append(dict(run=run,score=sname,method="RAND",constructor=constructor,
                                                     lambda1=float(lam1_used),
                                                     lambda2=float(lam2) if np.isfinite(lam2) else np.nan,
                                                     xi=xi,alpha=alpha,
                                                     cov_cal=cov_cal_m,cov_test=cov_test_m,
                                                     risk_cal=risk_cal,risk_test=risk_test,
                                                     set_size_cal_mean=set_size_cal_mean,
                                                     set_size_mean=set_size_test_mean,
                                                     feasible=int(bool(feas)),
                                                     test_size=int(len(y_test))))

                                else:
                                    raise ValueError("Unknown method: "+method)

    df=pd.DataFrame(rows); df.to_csv(args.out_csv,index=False)

    if args.select_best:
        def pick_best(group):
            feas = group[group["feasible"]==1]
            if len(feas)==0:
                return group.sort_values(["set_size_cal_mean","cov_cal","lambda1"], ascending=[True,False,True]).iloc[0]
            feas = feas.sort_values(["set_size_cal_mean","cov_cal","lambda1"], ascending=[True,False,True])
            return feas.iloc[0]

        base = []
        for (method,subdf) in df.groupby("method"):
            if method in ("EXG","CO"):
                key_cols = ["run","method","score","constructor","xi","alpha"]
                if method=="CO": key_cols.append("delta")
                best = subdf.groupby(key_cols, as_index=False).apply(pick_best).reset_index(drop=True)
                base.append(best)
            else:
                base.append(subdf)  # RAND, CRC_ALL, EXG_TX
        base = pd.concat(base, ignore_index=True)
    else:
        base = df

    if "delta" not in df.columns: df["delta"]=np.nan
    grp=base.groupby(["method","score","constructor","xi","alpha","delta"],as_index=False).agg(
        coverage_mean=("cov_test","mean"),
        coverage_std=("cov_test","std"),
        risk_mean=("risk_test","mean"),
        risk_std=("risk_test","std"),
        setsize_mean=("set_size_mean","mean"),
        setsize_cal_mean=("set_size_cal_mean","mean"),
        runs=("run","nunique"))
    grp.to_csv(args.out_summary,index=False)

    print(json.dumps({
        "wrote_raw": args.out_csv,
        "wrote_summary": args.out_summary
    }, indent=2))

if __name__=="__main__":
    main()
