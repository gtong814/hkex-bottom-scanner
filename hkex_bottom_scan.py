import io
import json
import math
import os
import time
from datetime import datetime, timezone, timedelta
import numpy as np
import pandas as pd
import requests
import yfinance as yf

# ==========================================
# 1. 演算法參數設定 (對應 Pine Script v5)
# ==========================================
CONFIG = {
    "pivot_len": 5,       # 轉折點週期 (W底 / 頭肩底)
    "vol_ma_len": 20,     # 成交量均線週期
    "vol_mult": 1.3,      # 突破放量倍數
    "tol_pct": 0.03,      # 雙底/左右肩容許誤差 (3%)
    "round_len": 40,      # 圓弧底觀察週期
    "v_len": 6,           # V型底單邊急跌/急漲天數
    "v_drop": 0.06,       # V型底最小跌幅/漲幅 (6%)
    "flat_len": 30,       # 潛伏底橫盤週期
    "flat_range": 0.08,   # 潛伏底最大振幅 (8%)
    "min_turnover": 5e6,  # 最低日成交額過濾 (500萬港元，過濾無成交殭屍股)
    "batch_size": 100,    # 每次批量下載股票數量
    "output_path": "data/signals.json"
}

# ==========================================
# 2. 取得全港上市正股名單
# ==========================================
def fetch_hkex_tickers():
    """從港交所官方 Excel 讀取全港正股名單，失敗時退回核心藍籌清單"""
    url = "https://www.hkex.com.hk/eng/services/trading/securities/securitieslists/ListOfSecurities.xlsx"
    try:
        resp = requests.get(url, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()
        df = pd.read_excel(io.BytesIO(resp.content), skiprows=2)
        equities = df[df["Category"].astype(str).str.strip() == "Equity"]
        tickers = {}
        for _, row in equities.iterrows():
            code = str(row["Stock Code"]).strip()
            if code.isdigit():
                fmt_code = f"{int(code):04d}.HK"
                name = str(row["Name of Securities"]).strip()
                tickers[fmt_code] = name
        print(f"成功從港交所載入 {len(tickers)} 隻正股。")
        return tickers
    except Exception as e:
        print(f"港交所名單下載失敗 ({e})，改用核心港股清單。")
        return {
            "0005.HK": "HSBC HOLDINGS", "0700.HK": "TENCENT", "9988.HK": "BABA-W",
            "3690.HK": "MEITUAN-W", "1299.HK": "AIA", "0941.HK": "CHINA MOBILE",
            "0388.HK": "HKEX", "1810.HK": "XIAOMI-W", "1211.HK": "BYD COMPANY",
            "0016.HK": "SHK PPT", "0001.HK": "CKH HOLDINGS", "0883.HK": "CNOOC",
            "0939.HK": "CCB", "1398.HK": "ICBC", "2318.HK": "PING AN",
            "9618.HK": "JD-SW", "9888.HK": "BIDU-SW", "2020.HK": "ANTA SPORTS"
        }

# ==========================================
# 3. 轉折點計算 (對應 ta.pivothigh / ta.pivotlow)
# ==========================================
def find_pivots(series, left, right, is_high=True):
    vals = series.values
    n = len(vals)
    pivots = []
    for i in range(left, n - right):
        window = vals[i - left : i + right + 1]
        center = vals[i]
        if is_high:
            if center == np.max(window) and np.sum(window == center) == 1:
                pivots.append((i, float(center)))
        else:
            if center == np.min(window) and np.sum(window == center) == 1:
                pivots.append((i, float(center)))
    return pivots

# ==========================================
# 4. 五大底部反轉型態偵測引擎
# ==========================================
def detect_patterns(df, cfg):
    if df is None or len(df) < 60:
        return None

    df = df.dropna(subset=["Open", "High", "Low", "Close", "Volume"]).copy()
    if len(df) < 60:
        return None

    close = df["Close"].values
    high = df["High"].values
    low = df["Low"].values
    vol = df["Volume"].values
    n = len(df)
    idx = n - 1  # 最新一根 K 線索引

    # 流動性過濾 (最新一日成交額)
    if close[idx] * vol[idx] < cfg["min_turnover"]:
        return None

    # 成交量確認
    vol_ma = pd.Series(vol).rolling(cfg["vol_ma_len"]).mean().values
    if np.isnan(vol_ma[idx]) or vol_ma[idx] == 0:
        return None
    vol_ratio = float(vol[idx] / vol_ma[idx])
    if vol_ratio < cfg["vol_mult"]:
        return None

    patterns = []
    p_len = cfg["pivot_len"]
    tol = cfg["tol_pct"]

    # 計算轉折點 (依時間由近到遠排序)
    pl_list = find_pivots(df["Low"], p_len, p_len, is_high=False)
    ph_list = find_pivots(df["High"], p_len, p_len, is_high=True)
    pl_list.reverse()
    ph_list.reverse()

    # --- 1. 雙底 (W底) ---
    if len(pl_list) >= 2 and len(ph_list) >= 1:
        b2, l2 = pl_list[0]
        b1, l1 = pl_list[1]
        neck = next((v for b, v in ph_list if b1 < b < b2), None)
        if neck is not None:
            valid_bottoms = abs(l1 - l2) / min(l1, l2) <= tol
            valid_neck = (neck - max(l1, l2)) / max(l1, l2) >= 0.02
            crossover = close[idx - 1] <= neck and close[idx] > neck
            if valid_bottoms and valid_neck and crossover and (idx - b2) < p_len * 4:
                patterns.append({"type": "W", "name": "雙底突破 (W底)", "neckline": round(neck, 3)})

    # --- 2. 頭肩底 ---
    if len(pl_list) >= 3 and len(ph_list) >= 2:
        rs_b, rs = pl_list[0]
        h_b, head = pl_list[1]
        ls_b, ls = pl_list[2]
        neck2 = next((v for b, v in ph_list if h_b < b < rs_b), None)
        neck1 = next((v for b, v in ph_list if ls_b < b < h_b), None)
        if neck1 is not None and neck2 is not None:
            neckline = max(neck1, neck2)
            head_lowest = head < ls and head < rs
            shoulders_aligned = abs(ls - rs) / min(ls, rs) <= (tol * 1.5)
            crossover = close[idx - 1] <= neckline and close[idx] > neckline
            if head_lowest and shoulders_aligned and crossover and (idx - rs_b) < p_len * 4:
                patterns.append({"type": "IHS", "name": "頭肩底突破", "neckline": round(neckline, 3)})

    # --- 3. 圓弧底 (鍋底) ---
    r_len = cfg["round_len"]
    if n > r_len + 10:
        win_high = high[idx - r_len : idx]
        win_low = low[idx - r_len : idx]
        highest_left = float(np.max(win_high))
        lowest_rel_idx = int(np.argmin(win_low))
        lowest_offset = lowest_rel_idx - r_len
        is_u_shape = -(r_len * 0.75) < lowest_offset < -(r_len * 0.25)

        ma10 = pd.Series(close).rolling(10).mean().values
        curving_up = ma10[idx] > ma10[idx - 3] and ma10[idx - 3] > ma10[idx - 6]
        crossover = close[idx - 1] <= highest_left and close[idx] > highest_left
        if is_u_shape and curving_up and crossover:
            patterns.append({"type": "ROUND", "name": "圓弧底突破 (鍋底)", "neckline": round(highest_left, 3)})

    # --- 4. V型底 (尖底) ---
    v_len = cfg["v_len"]
    v_drop = cfg["v_drop"]
    if n > v_len * 2 + 2:
        start_price = close[idx - v_len * 2]
        bottom_price = low[idx - v_len]
        is_sharp_bottom = bottom_price == np.min(low[idx - v_len * 2 : idx + 1])
        drop_rate = (start_price - bottom_price) / start_price
        recovery_rate = (close[idx] - bottom_price) / bottom_price
        if is_sharp_bottom and drop_rate >= v_drop and recovery_rate >= v_drop:
            patterns.append({"type": "V", "name": "V型底 (尖底)", "neckline": round(float(bottom_price), 3)})

    # --- 5. 潛伏底 (長期橫盤) ---
    f_len = cfg["flat_len"]
    f_range = cfg["flat_range"]
    if n > f_len + 2:
        box_high = float(np.max(high[idx - f_len : idx]))
        box_low = float(np.min(low[idx - f_len : idx]))
        box_range = (box_high - box_low) / box_low if box_low > 0 else 999
        crossover = close[idx - 1] <= box_high and close[idx] > box_high
        if box_range <= f_range and crossover:
            patterns.append({"type": "FLAT", "name": "潛伏底突破 (橫盤突破)", "neckline": round(box_high, 3)})

    return patterns, vol_ratio

# ==========================================
# 5. 主掃描流程與 JSON 匯出
# ==========================================
def main():
    tickers_map = fetch_hkex_tickers()
    all_codes = list(tickers_map.keys())
    results = []

    batch_size = CONFIG["batch_size"]
    total_batches = math.ceil(len(all_codes) / batch_size)

    for b in range(total_batches):
        batch = all_codes[b * batch_size : (b + 1) * batch_size]
        print(f"掃描進度: 批次 {b + 1}/{total_batches} ({len(batch)} 隻)...")
        try:
            data = yf.download(
                tickers=batch,
                period="6mo",
                interval="1d",
                group_by="ticker",
                auto_adjust=True,
                threads=True,
                progress=False
            )
        except Exception as e:
            print(f"批次 {b + 1} 下載錯誤: {e}")
            continue

        for code in batch:
            try:
                df = data[code] if len(batch) > 1 else data
                if df.empty or len(df) < 60:
                    continue

                res = detect_patterns(df, CONFIG)
                if not res:
                    continue
                patterns, vol_ratio = res
                if not patterns:
                    continue

                tail = df.dropna(subset=["Close"]).tail(90)
                last_close = float(tail["Close"].iloc[-1])
                prev_close = float(tail["Close"].iloc[-2])
                chg_pct = (last_close - prev_close) / prev_close * 100

                candles = [
                    {
                        "t": idx_dt.strftime("%Y-%m-%d"),
                        "o": round(float(r["Open"]), 3),
                        "h": round(float(r["High"]), 3),
                        "l": round(float(r["Low"]), 3),
                        "c": round(float(r["Close"]), 3),
                        "v": int(r["Volume"])
                    }
                    for idx_dt, r in tail.iterrows()
                ]

                results.append({
                    "code": code,
                    "name": tickers_map.get(code, code),
                    "price": round(last_close, 3),
                    "changePct": round(chg_pct, 2),
                    "volRatio": round(vol_ratio, 2),
                    "patterns": patterns,
                    "candles": candles
                })
            except Exception:
                continue

        time.sleep(1.0)

    hk_tz = timezone(timedelta(hours=8))
    payload = {
        "updatedAt": datetime.now(hk_tz).strftime("%Y-%m-%d %H:%M HKT"),
        "scannedCount": len(all_codes),
        "signalCount": len(results),
        "config": CONFIG,
        "signals": sorted(results, key=lambda x: x["volRatio"], reverse=True)
    }

    os.makedirs(os.path.dirname(CONFIG["output_path"]), exist_ok=True)
    with open(CONFIG["output_path"], "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, separators=(",", ":"))

    print(f"掃描完成！共掃描 {len(all_codes)} 隻港股，發現 {len(results)} 隻符合底部反轉型態。")

if __name__ == "__main__":
    main()
