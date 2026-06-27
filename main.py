import os
import sys
import json
import math
import numpy as np
from scipy.optimize import least_squares
from scipy.stats import norm
from scipy.interpolate import interp1d

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

def deepcopy_json(obj):
    return json.loads(json.dumps(obj))



_R_CACHE = {}
def rget(t, zc):
    k = json.dumps(zc)
    if k not in _R_CACHE:
        mats  = np.array([x["maturity"] for x in zc])
        rates = np.array([x["rate"]     for x in zc])
        _R_CACHE[k] = interp1d(mats, rates, kind='linear', fill_value="extrapolate")
    return float(_R_CACHE[k](t))

def functionp(t, zc):
    if t <= 0:
        return 1.0
    return math.exp(-rget(t, zc) * t)

def calc_V(T, a, sigmas, bpts):
    v = 0.0
    for j in range(len(sigmas)):
        t0 = bpts[j]
        t1 = bpts[j + 1]
        if T <= t0:
            break
        u0 = t0
        u1 = min(T, t1)
        if abs(a) < 1e-8:
            v += sigmas[j]**2 * (u1 - u0)
        else:
            v += sigmas[j]**2 * (math.exp(-2*a*(T - u1)) - math.exp(-2*a*(T - u0))) / (2*a)
    return v



def find_Z_star(T, t_i_list, c_i_list, v_i_list, P_0_T, P_0_t_i_list):
    Z = 0.0
    for _ in range(50):
        f  = -1.0
        df =  0.0
        for c, v, P_t in zip(c_i_list, v_i_list, P_0_t_i_list):
            term  = c * (P_t / P_0_T) * math.exp(-0.5*v*v - v*Z)
            f    += term
            df   -= v * term
        if abs(f) < 1e-12 or abs(df) < 1e-12:
            break
        if abs(df) > 1e-12:
            Z -= f / df
    return Z

def hw_swaption_price(T, tenor, a, sigmas, bpts, zc):
    t_i_arr = np.array([T + i + 1 for i in range(tenor)])
    P_0_T   = functionp(T, zc)
    P_0_t_i = np.array([functionp(t, zc) for t in t_i_arr])
    A       = float(P_0_t_i.sum())

    K       = (P_0_T - float(P_0_t_i[-1])) / A if A > 1e-12 else 0.0
    c_i     = np.full(tenor, K)
    c_i[-1] += 1.0

    V_T = calc_V(T, a, sigmas, bpts)
    if V_T <= 1e-12:
        return 0.0, A

    if abs(a) < 1e-8:
        B_vals = t_i_arr - T
    else:
        B_vals = (1.0 - np.exp(-a * (t_i_arr - T))) / a

    x_star = 0.0
    for _ in range(15):
        val = 0.0
        deriv = 0.0
        for i in range(tenor):
            bond = (P_0_t_i[i] / P_0_T) * np.exp(-B_vals[i] * x_star - 0.5 * B_vals[i]**2 * V_T)
            val += c_i[i] * bond
            deriv -= c_i[i] * bond * B_vals[i]
        f = val - 1.0
        if abs(f) < 1e-10:
            break
        if abs(deriv) > 1e-12:
            x_star -= f / deriv

    price = 0.0
    sqrt_V = math.sqrt(V_T)
    for i in range(tenor):
        sigma_P = B_vals[i] * sqrt_V
        if sigma_P <= 1e-8: continue
        X_i = (P_0_t_i[i] / P_0_T) * np.exp(-B_vals[i] * x_star - 0.5 * B_vals[i]**2 * V_T)
        d1 = (math.log(P_0_t_i[i] / (X_i * P_0_T)) + 0.5 * sigma_P**2) / sigma_P
        d2 = d1 - sigma_P
        zbp = X_i * P_0_T * norm.cdf(-d2) - P_0_t_i[i] * norm.cdf(-d1)
        price += c_i[i] * zbp

    return float(price), A

def calibration_error(params, swaptions, bpts, zc):
    a      = params[0]
    sigmas = params[1:]
    errors = []
    for sw in swaptions:
        T          = sw["expiry"]
        tenor      = sw["tenor"]
        vol_market = sw["vol_bps"]
        PS, A      = hw_swaption_price(T, tenor, a, sigmas, bpts, zc)
        if A > 1e-12 and T > 1e-12:
            vol_model = PS / (A * math.sqrt(T / (2 * math.pi))) * 10000.0
        else:
            vol_model = 0.0
        errors.append(vol_model - vol_market)
    return errors



def get_tree_bond(t_curr, t_targ, r_nodes, a, sigmas, bpts, zc):
    if t_curr >= t_targ:
        return np.ones_like(r_nodes)
    P_0_t = functionp(t_curr, zc)
    P_0_T = functionp(t_targ, zc)
    B_val = ((1.0 - math.exp(-a*(t_targ - t_curr))) / a
             if abs(a) > 1e-8 else t_targ - t_curr)
    V_0_t = calc_V(t_curr, a, sigmas, bpts)
    eps   = 1e-5
    f_0_t = -(math.log(functionp(t_curr + eps, zc)) -
              math.log(functionp(max(0.0, t_curr - eps), zc))) / (
              (t_curr + eps) - max(0.0, t_curr - eps))
    exponent = -B_val * (r_nodes - f_0_t) - 0.5 * B_val**2 * V_0_t
    return (P_0_T / P_0_t) * np.exp(exponent)



def price_bermudan(a, sigmas, bpts, exercise_dates, swap_end, K_strike, notional, zc):
    spy     = 24
    last_ex = max(exercise_dates)

    landmarks = sorted(list(set([0.0] + list(exercise_dates) + [b for b in bpts if b <= last_ex])))
    if last_ex not in landmarks:
        landmarks.append(last_ex)

    times = []
    for i in range(len(landmarks) - 1):
        t0    = landmarks[i]
        t1    = landmarks[i + 1]
        steps = max(1, int(round((t1 - t0) * spy)))
        dt_s  = (t1 - t0) / steps
        for j in range(steps):
            times.append(t0 + j * dt_s)
    times.append(last_ex)
    M = len(times) - 1

    def get_sigma(t):
        for i in range(len(bpts) - 1):
            if bpts[i] <= t < bpts[i + 1]:
                return sigmas[i]
        return sigmas[-1]

    def B_hw(dt_val):
        if abs(a) < 1e-8:
            return dt_val
        return (1.0 - math.exp(-a * dt_val)) / a

    Q       = np.array([1.0])
    j_min   = 0
    j_max   = 0
    dx_prev = 0.0
    tree_data = []

    for k in range(M):
        t_k    = times[k]
        t_next = times[k + 1]
        dt     = t_next - t_k

        sig = get_sigma(t_k + 0.5 * dt)
        if abs(a) > 1e-8:
            V_k = sig**2 * (1.0 - math.exp(-2*a*dt)) / (2*a)
        else:
            V_k = sig**2 * dt

        dx_next = math.sqrt(3.0 * V_k)

        if k == 0:
            x_k = np.array([0.0])
        else:
            x_k = np.arange(j_min, j_max + 1) * dx_prev

        B_step    = B_hw(dt)
        exp_Bx    = np.exp(-B_step * x_k)
        sum_Q_exp = float(np.sum(Q * exp_Bx))
        c_step    = functionp(t_k, zc) / sum_Q_exp if sum_Q_exp > 1e-20 else 1.0
        D_k       = (functionp(t_next, zc) / functionp(t_k, zc)) * exp_Bx * c_step

        M_k    = x_k * math.exp(-a * dt)
        k_star = np.round(M_k / dx_next).astype(int)
        eta    = M_k - k_star * dx_next

        p_u = 1.0/6.0 + 0.5*(eta/dx_next)**2 + 0.5*(eta/dx_next)
        p_m = 2.0/3.0 -     (eta/dx_next)**2
        p_d = 1.0/6.0 + 0.5*(eta/dx_next)**2 - 0.5*(eta/dx_next)

        nj_min = int(np.min(k_star)) - 1
        nj_max = int(np.max(k_star)) + 1
        Q_next = np.zeros(nj_max - nj_min + 1)
        val    = Q * D_k

        np.add.at(Q_next, k_star + 1 - nj_min, val * p_u)
        np.add.at(Q_next, k_star     - nj_min, val * p_m)
        np.add.at(Q_next, k_star - 1 - nj_min, val * p_d)

        tree_data.append({
            "D_k": D_k, "p_u": p_u, "p_m": p_m, "p_d": p_d,
            "k_star": k_star, "j_min_next": nj_min, "x_k": x_k, "Q": Q.copy()
        })

        Q       = Q_next
        j_min   = nj_min
        j_max   = nj_max
        dx_prev = dx_next

    x_M = np.arange(j_min, j_max + 1) * dx_prev

    def calc_exercise(t_k, x_grid, Q_grid):
        val = 1.0
        T_i = int(round(t_k)) + 1
        while T_i <= swap_end:
            B_i       = B_hw(float(T_i) - t_k)
            exp_Bx_i  = np.exp(-B_i * x_grid)
            sum_Q_exp = float(np.sum(Q_grid * exp_Bx_i))
            c_i_fac   = functionp(t_k, zc) / sum_Q_exp if sum_Q_exp > 1e-20 else 1.0
            bond      = (functionp(float(T_i), zc) / functionp(t_k, zc)) * exp_Bx_i * c_i_fac
            
            if T_i == swap_end:
                val -= bond * (1.0 + K_strike)
            else:
                val -= bond * K_strike
            T_i += 1
        return np.maximum(val, 0.0)

    V = calc_exercise(times[M], x_M, Q)

    for k in range(M - 1, -1, -1):
        data = tree_data[k]
        t_k  = times[k]
        ks   = data["k_star"]
        jn   = data["j_min_next"]

        continuation = data["D_k"] * (
            data["p_u"] * V[ks + 1 - jn] +
            data["p_m"] * V[ks     - jn] +
            data["p_d"] * V[ks - 1 - jn]
        )

        is_ex = False
        for ed in exercise_dates:
            if abs(t_k - ed) < 1e-6:
                is_ex = True
                break
        
        if is_ex:
            ex_val = calc_exercise(t_k, data["x_k"], data["Q"])
            V = np.maximum(ex_val, continuation)
        else:
            V = continuation

    return float(V[0]) * notional



def main():
    input_data = sys.stdin.read()
    if not input_data.strip():
        return

    data           = json.loads(input_data)
    zc             = data["zero_curve"]
    swaption_vols  = data["swaption_vols"]
    spec           = data["bermudan_spec"]

    notional       = spec["notional"]
    strike         = spec["strike"]
    exercise_dates = sorted(spec["exercise_dates"])
    swap_end       = spec["swap_end"]

    bpts = [0.0] + [float(e) for e in exercise_dates]
    if bpts[-1] < swap_end:
        bpts.append(float(swap_end))
    n_sigmas = len(bpts) - 1

    curve_out = []
    mats = [x["maturity"] for x in zc]
    for idx, m in enumerate(mats):
        df = functionp(m, zc)
        if idx < len(mats) - 1:
            df_next = functionp(mats[idx + 1], zc)
            fwd     = -math.log(df_next / df) / (mats[idx + 1] - m)
        else:
            df_prev = functionp(mats[idx - 1], zc)
            fwd     = -math.log(df / df_prev) / (m - mats[idx - 1])
        curve_out.append({
            "maturity":        m,
            "discount_factor": round(df,  4),
            "forward_rate":    round(fwd, 4),
        })

    x0     = [0.05] + [0.008] * n_sigmas
    bounds = ([0.0001] + [0.0001] * n_sigmas, [0.5] + [0.1] * n_sigmas)

    res = least_squares(
        calibration_error, x0, bounds=bounds,
        args=(swaption_vols, bpts, zc),
        ftol=1e-8, xtol=1e-8, gtol=1e-8,
    )
    cal_p   = res.x
    a_cal   = cal_p[0]
    sig_cal = cal_p[1:]

    calibration_out = {
        "model": "Hull-White",
        "parameters": {"mean_reversion": round(a_cal, 6)},
    }
    for i, s in enumerate(sig_cal):
        calibration_out["parameters"][f"sigma_{i+1}"] = round(s, 6)

    base_price = price_bermudan(
        a_cal, sig_cal, bpts, exercise_dates, swap_end, strike, notional, zc
    )

    vega_out = []
    for i, sw in enumerate(swaption_vols):
        bumped_vols = deepcopy_json(swaption_vols)
        bumped_vols[i]["vol_bps"] += 1.0
        
        res_b = least_squares(
            calibration_error, cal_p, bounds=bounds,
            args=(bumped_vols, bpts, zc),
            ftol=1e-8, xtol=1e-8, gtol=1e-8,
        )
        bp = price_bermudan(
            res_b.x[0], res_b.x[1:], bpts, exercise_dates, swap_end, strike, notional, zc
        )
        vega_out.append({
            "expiry":              sw["expiry"],
            "tenor":               sw["tenor"],
            "vega_dollars_per_bp": round(bp - base_price, 4),
        })

    delta_out = []
    for i, zc_point in enumerate(zc):
        bumped_curve = deepcopy_json(zc)
        bumped_curve[i]["rate"] += 0.0001
        
        res_b = least_squares(
            calibration_error, cal_p, bounds=bounds,
            args=(swaption_vols, bpts, bumped_curve),
            ftol=1e-8, xtol=1e-8, gtol=1e-8,
        )
        bp = price_bermudan(
            res_b.x[0], res_b.x[1:], bpts, exercise_dates, swap_end, strike, notional, bumped_curve
        )
        delta_out.append({
            "maturity":              zc_point["maturity"],
            "delta_dollars_per_bp":  round(bp - base_price, 4),
        })

    output = {
        "curve":         curve_out,
        "calibration":   calibration_out,
        "price_dollars": round(base_price, 2),
        "vega":          vega_out,
        "delta":         delta_out,
    }
    print(json.dumps(output, indent=2))

if __name__ == "__main__":
    main()