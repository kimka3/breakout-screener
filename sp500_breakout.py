"""
S&P 500 120일 이동평균선 돌파 + 거래량 급증 스크리너

조건
  1) 종가가 120일 이동평균선을 상향 돌파 (전일: 종가 <= MA120, 당일: 종가 > MA120)
  2) 돌파 당일 거래량 >= 직전 5거래일(1주일) 평균 거래량 * 2

사용 예
  python sp500_breakout.py                    # 최신 거래일 기준 스캔
  python sp500_breakout.py --lookback 5       # 최근 5거래일 내 발생한 신호 모두
  python sp500_breakout.py --date 2026-08-20  # 특정 일자 기준
  python sp500_breakout.py --ma 60 --vol-mult 1.5 --vol-window 5
  python sp500_breakout.py --tickers AAPL,MSFT,NVDA
"""
from __future__ import annotations

import argparse
import hashlib
from collections import Counter
import json
import os
import pickle
import ssl
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

BASE_DIR = Path(__file__).resolve().parent
CACHE_DIR = BASE_DIR / ".cache"
CA_BUNDLE = CACHE_DIR / "system-ca-bundle.pem"

# 한글 출력이 깨지지 않도록 (구형 콘솔에서도 예외 없이 동작)
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


# --------------------------------------------------------------------------
# 사내 TLS 프록시(MITM) 환경 대응: Windows 인증서 저장소를 PEM 번들로 내보내
# curl/requests 가 쓰도록 환경변수를 설정한다. (yfinance import 전에 실행)
# --------------------------------------------------------------------------
def setup_ca_bundle() -> None:
    if os.name != "nt":
        return
    if any(os.environ.get(k) for k in ("CURL_CA_BUNDLE", "REQUESTS_CA_BUNDLE", "SSL_CERT_FILE")):
        return
    try:
        if not CA_BUNDLE.exists():
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            pems = []
            for store in ("ROOT", "CA"):
                for der, _enc, trust in ssl.enum_certificates(store):
                    if trust is True or (isinstance(trust, set) and trust):
                        pems.append(ssl.DER_cert_to_PEM_cert(der))
            try:
                import certifi

                pems.append(Path(certifi.where()).read_text(encoding="utf-8"))
            except Exception:
                pass
            CA_BUNDLE.write_text("".join(pems), encoding="ascii")
        for key in ("CURL_CA_BUNDLE", "REQUESTS_CA_BUNDLE"):
            os.environ[key] = str(CA_BUNDLE)
    except Exception as exc:  # 실패해도 기본 인증서로 시도
        print(f"[warn] CA 번들 생성 실패, 기본 인증서로 진행합니다: {exc}", file=sys.stderr)


setup_ca_bundle()

import pandas as pd  # noqa: E402
import yfinance as yf  # noqa: E402

WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
CONSTITUENTS_CSV = CACHE_DIR / "sp500_constituents.csv"


# --------------------------------------------------------------------------
# 1. S&P 500 구성종목
# --------------------------------------------------------------------------
def _http_get(url: str, timeout: int = 30, encoding: str | None = None) -> str:
    """yfinance 가 쓰는 curl_cffi 로 요청. urllib/requests 는 사내 프록시 인증서에서 자주 실패한다."""
    try:
        from curl_cffi import requests as creq

        resp = creq.get(url, impersonate="chrome", timeout=timeout)
        resp.raise_for_status()
        return resp.content.decode(encoding, errors="replace") if encoding else resp.text
    except ImportError:
        import urllib.request

        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        raw = urllib.request.urlopen(req, timeout=timeout).read()
        return raw.decode(encoding or "utf-8", errors="replace")



def _read_cached_csv(path: Path, max_age_days: float = 7, **kw):
    """캐시 CSV 를 읽되, 없거나 오래됐거나 읽을 수 없으면 None.

    사내 DLP 에이전트가 디스크의 파일을 암호화해 두는 경우가 있어
    (헤더가 CSV 가 아닌 바이너리로 바뀐다) 읽기 실패도 캐시 미스로 처리한다.
    """
    if not path.exists():
        return None
    if (time.time() - path.stat().st_mtime) / 86400 >= max_age_days:
        return None
    try:
        return pd.read_csv(path, encoding="utf-8", **kw)
    except Exception as exc:
        print(f"[warn] 캐시 {path.name} 을 읽을 수 없어 새로 받습니다: {type(exc).__name__}",
              file=sys.stderr)
        return None


def load_sp500(refresh: bool = False) -> pd.DataFrame:
    """Wikipedia 에서 구성종목을 받아온다. 실패하면 캐시를 사용."""
    if not refresh:
        cached = _read_cached_csv(CONSTITUENTS_CSV)
        if cached is not None:
            return cached

    try:
        import io

        html = _http_get(WIKI_URL)
        table = pd.read_html(io.StringIO(html), match="Symbol")[0]
        df = pd.DataFrame(
            {
                "ticker": table["Symbol"].astype(str).str.strip(),
                "name": table["Security"].astype(str).str.strip(),
                "sector": table["GICS Sector"].astype(str).str.strip(),
            }
        )
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        df.to_csv(CONSTITUENTS_CSV, index=False)
        return df
    except Exception as exc:
        cached = _read_cached_csv(CONSTITUENTS_CSV, max_age_days=float("inf"))
        if cached is not None:
            print(f"[warn] 구성종목 갱신 실패({exc}). 캐시를 사용합니다.", file=sys.stderr)
            return cached
        raise SystemExit(f"S&P 500 구성종목을 가져오지 못했습니다: {exc}")


KOSPI200_CSV = CACHE_DIR / "kospi200_constituents.csv"
NAVER_KPI200 = "https://finance.naver.com/sise/entryJongmok.naver?&page={page}&type=KPI200"


def load_kospi200(refresh: bool = False) -> pd.DataFrame:
    """네이버 금융의 코스피200 편입종목 목록. KRX 직접 조회는 사내 프록시에서 막힌다."""
    if not refresh:
        cached = _read_cached_csv(KOSPI200_CSV, dtype={"ticker": str})
        if cached is not None:
            return cached

    try:
        import io
        import re

        found = {}
        for page in range(1, 30):
            html = _http_get(NAVER_KPI200.format(page=page), encoding="cp949")
            codes = list(dict.fromkeys(re.findall(r"code=(\d{6})", html)))
            if not codes:
                break
            table = pd.read_html(io.StringIO(html))[0].dropna(subset=["종목별"])
            names = [str(x).strip() for x in table["종목별"].tolist()]
            for code, name in zip(codes, names):
                found.setdefault(code, name)
        if not found:
            raise RuntimeError("편입종목을 찾지 못했습니다")

        df = pd.DataFrame({"ticker": list(found.keys()), "name": list(found.values())})
        df["sector"] = ""
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        df.to_csv(KOSPI200_CSV, index=False)
        return df
    except Exception as exc:
        cached = _read_cached_csv(KOSPI200_CSV, max_age_days=float("inf"), dtype={"ticker": str})
        if cached is not None:
            print(f"[warn] 코스피200 목록 갱신 실패({exc}). 캐시를 사용합니다.", file=sys.stderr)
            return cached
        raise SystemExit(f"코스피200 구성종목을 가져오지 못했습니다: {exc}")


# --------------------------------------------------------------------------
# 시장 정의 — 통화·거래시간·표기 단위가 다르므로 시장별로 따로 다룬다
# --------------------------------------------------------------------------
MARKETS = {
    "sp500": {
        "label": "S&P 500",
        "currency": "USD",
        "decimals": 2,
        "tz": "America/New_York",
        "close": (16, 0),
        "loader": load_sp500,
        "to_yahoo": lambda t: t.replace(".", "-").upper(),
    },
    "kospi200": {
        "label": "KOSPI 200",
        "currency": "KRW",
        "decimals": 0,
        "tz": "Asia/Seoul",
        "close": (15, 30),
        "loader": load_kospi200,
        "to_yahoo": lambda t: f"{str(t).zfill(6)}.KS",
    },
}


def to_yahoo(ticker: str) -> str:
    """BRK.B -> BRK-B 같은 야후 표기로 변환."""
    return ticker.replace(".", "-").upper()


def drop_partial_bar(df, market: str):
    """장이 아직 안 끝났으면 미완성인 당일 봉을 버린다.

    장중 봉은 거래량이 덜 쌓여 있어 그대로 쓰면 거래량 급증 판정이 왜곡된다.
    """
    if len(df) < 2:
        return df
    spec = MARKETS[market]
    now = datetime.now(ZoneInfo(spec["tz"]))
    if df.index[-1].date() == now.date() and (now.hour, now.minute) < spec["close"]:
        return df.iloc[:-1]
    return df


# --------------------------------------------------------------------------
# 2. 시세 다운로드
# --------------------------------------------------------------------------
def download_prices(tickers, start, end, chunk=60, use_cache=True):
    """티커별 OHLCV DataFrame(dict) 반환.

    묶음으로 두 번 받아보고, 그래도 빠진 종목은 개별 요청으로 한 번 더 시도한다.
    """
    cache_path = None
    if use_cache:
        key = hashlib.md5(
            f"{start}|{end}|{','.join(sorted(tickers))}".encode()
        ).hexdigest()[:12]
        cache_path = CACHE_DIR / f"prices_{key}.pkl"
        if cache_path.exists() and (time.time() - cache_path.stat().st_mtime) < 6 * 3600:
            try:
                with cache_path.open("rb") as fh:
                    cached = pickle.load(fh)
                print(f"  캐시 사용 ({cache_path.name})")
                return cached
            except Exception:
                pass

    out = {}
    pending = list(tickers)

    for attempt in range(2):
        failed = []
        for i in range(0, len(pending), chunk):
            batch = pending[i : i + chunk]
            done = min(i + chunk, len(pending))
            tag = " (재시도)" if attempt else ""
            print(f"  다운로드 {done}/{len(pending)}{tag}", end="\r", flush=True)
            try:
                raw = yf.download(
                    batch,
                    start=start,
                    end=end + timedelta(days=1),
                    interval="1d",
                    auto_adjust=False,
                    actions=False,
                    group_by="ticker",
                    threads=True,
                    progress=False,
                )
            except Exception:
                failed.extend(batch)
                continue

            for t in batch:
                try:
                    df = raw[t] if isinstance(raw.columns, pd.MultiIndex) else raw
                except KeyError:
                    failed.append(t)
                    continue
                df = df.dropna(subset=["Close", "Volume"])
                if df.empty:
                    failed.append(t)
                else:
                    out[t] = df
        print(" " * 60, end="\r")
        pending = failed
        if not pending:
            break
        time.sleep(2)

    # 묶음으로 두 번 다 실패한 종목은 하나씩 다시 받아본다.
    # 배치 실패는 대개 묶음 단위 문제(스로틀링, 한 종목이 응답을 망침)라
    # 개별 요청으로는 대부분 살아난다. 조용히 빠지면 신호를 놓친다.
    if pending:
        recovered = set()
        for i, t in enumerate(pending, 1):
            print(f"  개별 재시도 {i}/{len(pending)} {t}", end="\r", flush=True)
            for delay in (0, 2):
                if delay:
                    time.sleep(delay)
                try:
                    df = yf.download(t, start=start, end=end + timedelta(days=1),
                                     interval="1d", auto_adjust=False, actions=False,
                                     progress=False, threads=False)
                except Exception:
                    continue
                if isinstance(df.columns, pd.MultiIndex):
                    try:
                        df = df[t]
                    except KeyError:
                        df = df.droplevel(-1, axis=1)
                df = df.dropna(subset=["Close", "Volume"])
                if not df.empty:
                    out[t] = df
                    recovered.add(t)
                    break
        print(" " * 60, end="\r")
        if recovered:
            print(f"  개별 재시도로 {len(recovered)}종목 복구")
        pending = [t for t in pending if t not in recovered]

    if pending:
        head = ", ".join(pending[:15])
        more = " ..." if len(pending) > 15 else ""
        print(f"[warn] 데이터 실패 {len(pending)}종목: {head}{more}", file=sys.stderr)
    download_prices.last_failed = list(pending)

    if cache_path is not None and out:
        try:
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            with cache_path.open("wb") as fh:
                pickle.dump(out, fh)
        except Exception:
            pass
    return out


def fetch_fundamentals(tickers):
    """조건을 통과한 종목만 PER 등 기본 지표를 조회한다 (종목당 1회 요청)."""
    out = {}
    for i, t in enumerate(tickers, 1):
        print(f"  PER 조회 {i}/{len(tickers)}", end="\r", flush=True)
        try:
            info = yf.Ticker(t).info
            out[t] = {
                "per": info.get("trailingPE"),
                "forward_per": info.get("forwardPE"),
                "eps": info.get("trailingEps"),
                "sector": info.get("sector") or "",
            }
        except Exception:
            out[t] = {"per": None, "forward_per": None, "eps": None, "sector": ""}
    print(" " * 40, end="\r")
    return out


def _num(v):
    """None/NaN/무한대를 걸러 float 또는 None 을 돌려준다."""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if pd.isna(f) or f in (float("inf"), float("-inf")) else round(f, 2)


# --------------------------------------------------------------------------
# 3. 신호 계산
# --------------------------------------------------------------------------
def find_signals(df, ma_period, vol_window, vol_mult, price_basis, require_hold=True):
    """한 종목의 시계열에서 조건을 만족하는 모든 날짜를 반환."""
    price = df["Adj Close"] if price_basis == "adj" and "Adj Close" in df else df["Close"]
    price = price.astype(float)
    volume = df["Volume"].astype(float)

    ma = price.rolling(ma_period, min_periods=ma_period).mean()
    # 돌파 당일을 제외한 직전 vol_window 거래일의 평균 거래량
    avg_vol = volume.shift(1).rolling(vol_window, min_periods=vol_window).mean()

    crossed_up = (price > ma) & (price.shift(1) <= ma.shift(1)) & ma.notna() & ma.shift(1).notna()
    vol_ratio = volume / avg_vol
    hit = (crossed_up & (vol_ratio >= vol_mult)).fillna(False)

    if not hit.any():
        return pd.DataFrame()

    # 돌파 이후 현재가가 이동평균선 위를 지키고 있는지
    last_price, last_ma = price.iloc[-1], ma.iloc[-1]
    holding = bool(pd.notna(last_ma) and last_price > last_ma)
    if require_hold and not holding:
        return pd.DataFrame()

    last_i = len(df) - 1
    positions = [i for i, v in enumerate(hit.values) if v]

    return pd.DataFrame(
        {
            "date": df.index[hit],
            "close": df["Close"].astype(float)[hit].values,
            "ma": ma[hit].values,
            "above_ma_pct": (price[hit].values / ma[hit].values - 1) * 100,
            "volume": volume[hit].values,
            "avg_vol": avg_vol[hit].values,
            "vol_ratio": vol_ratio[hit].values,
            "last_close": float(df["Close"].astype(float).iloc[-1]),
            "last_ma": float(last_ma) if pd.notna(last_ma) else float("nan"),
            "last_pct": (float(last_price) / float(last_ma) - 1) * 100 if pd.notna(last_ma) else float("nan"),
            "bars_since": [last_i - p for p in positions],
            "holding": holding,
        }
    )


# --------------------------------------------------------------------------
# 4. HTML 리포트 (휴대폰 브라우저용)
# --------------------------------------------------------------------------
CHART_WINDOW = 70  # 카드 차트에 담을 거래일 수 (일봉 몸통이 뭉개지지 않는 상한)


def build_series(df, sig_row, args, ticker, name, sector, market="sp500"):
    """카드 차트에 넣을 시계열 조각을 만든다."""
    price = df["Adj Close"] if args.price_basis == "adj" and "Adj Close" in df else df["Close"]
    ma = price.astype(float).rolling(args.ma, min_periods=args.ma).mean()

    end_pos = len(df.index)
    start_pos = max(0, end_pos - CHART_WINDOW)
    window = df.index[start_pos:end_pos]
    sig_date = pd.Timestamp(sig_row["date"])
    sig_pos = int(df.index.get_loc(sig_date)) - start_pos

    def clean(series):
        vals = series.iloc[start_pos:end_pos]
        return [None if pd.isna(v) else round(float(v), 4) for v in vals]

    return {
        "market": market,
        "ticker": ticker,
        "name": name,
        "sector": sector,
        "date": str(pd.Timestamp(sig_row["date"]).date()),
        "close": round(float(sig_row["close"]), 2),
        "ma": round(float(sig_row["ma"]), 2),
        "abovePct": round(float(sig_row["above_ma_pct"]), 2),
        "volume": int(sig_row["volume"]),
        "avgVol": int(sig_row["avg_vol"]),
        "volRatio": round(float(sig_row["vol_ratio"]), 2),
        "lastClose": round(float(sig_row["last_close"]), 2),
        "lastMa": round(float(sig_row["last_ma"]), 2),
        "lastPct": round(float(sig_row["last_pct"]), 2),
        # 돌파일 종가 대비 현재 종가 등락률 (표시되는 실제 종가끼리의 비교)
        "retPct": round((float(sig_row["last_close"]) / float(sig_row["close"]) - 1) * 100, 2),
        "barsSince": int(sig_row["bars_since"]),
        "sigIndex": sig_pos,
        "dates": [d.strftime("%Y-%m-%d") for d in window],
        "opens": clean(df["Open"].astype(float)),
        "highs": clean(df["High"].astype(float)),
        "lows": clean(df["Low"].astype(float)),
        "closes": clean(df["Close"].astype(float)),
        "mas": clean(ma),
        "volumes": [int(v) for v in df["Volume"].astype(float).iloc[start_pos:end_pos]],
    }


def write_html(charts, args, markets, out_path: Path):
    """템플릿에 데이터를 주입해 리포트 2종(단독 실행용 / 아티팩트용)을 쓴다.

    markets 는 시장별 요약 목록. 거래일·통화·표기 단위가 시장마다 다르므로
    하나로 뭉뚱그리지 않고 각각 들고 간다.
    """
    template_path = BASE_DIR / "report_template.html"
    if not template_path.exists():
        print(f"[warn] {template_path.name} 이 없어 HTML 생성을 건너뜁니다.", file=sys.stderr)
        return

    payload = {
        "generatedAt": datetime.now(ZoneInfo("Asia/Seoul")).strftime("%Y-%m-%d %H:%M") + " KST",
        "markets": markets,
        "maPeriod": args.ma,
        "volWindow": args.vol_window,
        "volMult": args.vol_mult,
        "lookback": args.lookback,
        "priceBasis": args.price_basis,
        "requireHold": not args.no_hold,
        "hits": charts,
    }
    fragment = template_path.read_text(encoding="utf-8").replace(
        "/*__DATA__*/null", json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    )

    artifact_path = out_path.with_suffix(".artifact.html")
    artifact_path.write_text(fragment, encoding="utf-8")

    # 홈 화면 아이콘은 head 에 있어야 iOS 가 인식한다. 없으면 페이지를 축소한
    # 스크린샷을 아이콘으로 써서 알아보기 어렵다.
    standalone = (
        '<!doctype html>\n<html lang="ko">\n<head>\n<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        '<link rel="apple-touch-icon" href="apple-touch-icon.png">\n'
        '<link rel="icon" type="image/png" sizes="32x32" href="favicon.png">\n'
        '<link rel="icon" type="image/png" sizes="512x512" href="icon-512.png">\n'
        '<meta name="apple-mobile-web-app-title" content="돌파 스크린">\n'
        '<meta name="theme-color" media="(prefers-color-scheme: light)" content="#EEF1F2">\n'
        '<meta name="theme-color" media="(prefers-color-scheme: dark)" content="#0E1315">\n'
        "</head>\n<body>\n" + fragment + "\n</body>\n</html>\n"
    )
    out_path.write_text(standalone, encoding="utf-8")
    print(f"HTML: {out_path.resolve()}")
    print(f"      {artifact_path.resolve()}  (Artifact 게시용)")


# --------------------------------------------------------------------------
# 5. 메인
# --------------------------------------------------------------------------
def main() -> int:
    p = argparse.ArgumentParser(
        description="S&P 500 120일선 상향돌파 + 거래량 급증 스크리너",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--ma", type=int, default=120, help="이동평균 기간(거래일)")
    p.add_argument("--vol-window", type=int, default=5, help="직전 평균거래량 산출 기간(거래일)")
    p.add_argument("--vol-mult", type=float, default=2.0, help="거래량 배수 기준")
    p.add_argument("--lookback", type=int, default=1, help="최근 N 거래일 내의 신호를 검색")
    p.add_argument("--date", default=None, help="기준일 YYYY-MM-DD (미지정시 최신 거래일)")
    p.add_argument("--price-basis", choices=["adj", "raw"], default="adj",
                   help="신호 계산 기준가: adj=수정주가(배당/분할 반영), raw=종가 그대로")
    p.add_argument("--market", choices=["all", "sp500", "kospi200"], default="all",
                   help="스캔할 시장")
    p.add_argument("--tickers", default=None, help="쉼표구분 티커 목록(테스트용)")
    p.add_argument("--refresh-list", action="store_true", help="구성종목 목록 강제 갱신")
    p.add_argument("--out", default="breakouts.csv", help="결과 CSV 경로")
    p.add_argument("--html", nargs="?", const="report.html", default=None,
                   help="휴대폰용 HTML 리포트 생성 (경로 생략시 report.html)")
    p.add_argument("--no-cache", action="store_true", help="시세 캐시를 쓰지 않고 새로 받기")
    p.add_argument("--no-hold", action="store_true",
                   help="돌파 후 현재가가 이동평균선 아래로 다시 내려간 종목도 포함")
    args = p.parse_args()

    end_date = datetime.strptime(args.date, "%Y-%m-%d").date() if args.date else date.today()
    # MA 기간 + 여유분 확보 (거래일 -> 달력일 환산 약 1.55배 + 버퍼)
    # HTML 리포트는 차트 구간 내내 MA 선이 그려져야 하므로 그만큼 더 받는다.
    span = args.ma + args.vol_window + args.lookback + (CHART_WINDOW if args.html else 0)
    start_date = end_date - timedelta(days=int(span * 1.55) + 40)

    print(f"기간: {start_date} ~ {end_date}")
    print(f"조건: 종가 > MA{args.ma} 상향돌파 & 거래량 >= 직전 {args.vol_window}거래일 평균 x {args.vol_mult}"
          + ("" if args.no_hold else f" & 현재가 > MA{args.ma} 유지"))

    selected = list(MARKETS) if args.market == "all" else [args.market]
    rows, charts, market_summaries = [], [], []

    for mk in selected:
        spec = MARKETS[mk]
        if args.tickers:
            meta = pd.DataFrame({"ticker": [t.strip().upper() for t in args.tickers.split(",")]})
            meta["name"] = ""
            meta["sector"] = ""
        else:
            meta = spec["loader"](refresh=args.refresh_list)
        meta["ticker"] = meta["ticker"].astype(str)
        meta["yahoo"] = meta["ticker"].map(spec["to_yahoo"])

        print(f"\n[{spec['label']}] 대상 {len(meta)}종목")
        prices = download_prices(meta["yahoo"].tolist(), start_date, end_date,
                                 use_cache=not args.no_cache)
        n_failed = len(getattr(download_prices, "last_failed", []) or [])
        prices = {t: drop_partial_bar(df, mk) for t, df in prices.items()}
        # 과거 데이터가 짧으면 이동평균이 계산되지 않아 신호가 조용히 사라진다.
        # MA 는 앞선 ma 봉이 있어야 나오므로, 검색 구간 전체를 평가하려면
        # ma + lookback 봉이 필요하다. 그에 못 미치는 종목을 따로 센다.
        need_full = args.ma + args.lookback
        n_no_hist = sum(1 for df in prices.values() if len(df) <= args.ma)
        prices = {t: df for t, df in prices.items() if len(df) > args.ma}
        short = {t: len(df) for t, df in prices.items() if len(df) < need_full}
        print(f"[{spec['label']}] 시세 확보 {len(prices)}종목"
              + (f" (다운로드 실패 {n_failed}종목 — 스캔 제외)" if n_failed else ""))
        if n_no_hist:
            print(f"[{spec['label']}] 이력이 MA{args.ma} 에 못 미쳐 제외 {n_no_hist}종목")
        if short:
            sample = ", ".join(f"{t}({n}봉)" for t, n in list(short.items())[:8])
            print(f"[{spec['label']}] 이력이 짧아 검색 구간 일부만 평가됨 {len(short)}종목: {sample}"
                  + (" ..." if len(short) > 8 else ""))
        if not prices:
            print(f"[warn] {spec['label']} 시세를 받지 못해 건너뜁니다.", file=sys.stderr)
            continue

        # 기준 거래일은 최댓값이 아니라 최빈값으로 잡는다. 한두 종목이 남들보다
        # 하루 앞선 봉을 갖고 있는 경우가 있어, 최댓값을 쓰면 헤더에 찍히는 날짜가
        # 정작 대부분 종목의 데이터보다 하루 앞서게 된다.
        # 야후가 종목에 따라 거래일을 통째로 빠뜨리는 일이 있다(환경에 따라 다르다).
        # 빠진 날이 검색 구간 안에 있으면 "전일 아래 → 당일 위" 가 실제로는
        # 없었는데 있는 것처럼 보여 가짜 돌파가 만들어진다. 놓치는 것보다 나쁘다.
        # 전 종목의 날짜 분포로 그 시장의 거래일 달력을 만들어 대조한다.
        day_count = Counter()
        for df in prices.values():
            day_count.update(df.index.normalize())
        # 기준은 "다수결"이 아니라 "존재 증거"다. 어느 날짜가 소수 종목에만
        # 있더라도 그건 실제 거래일이고, 없는 쪽이 결손이다. 다수결로 잡으면
        # 결손이 광범위할 때 그 날짜가 달력에서 통째로 사라져 탐지가 무력해진다.
        # 같은 시장의 서로 다른 종목 여러 개가 같은 날짜를 갖고 있다면 실제 거래일이다.
        # 기준을 높이면 결손이 심할수록 그 날짜가 달력에서 사라져 탐지가 무력해진다.
        quorum = max(5, int(len(prices) * 0.03))
        calendar = sorted(d for d, n in day_count.items() if n >= quorum)
        check_days = calendar[-(args.lookback + args.vol_window + 2):]
        gapped = []
        for t, df in prices.items():
            have = set(df.index.normalize())
            first = df.index.min()
            if any(d not in have for d in check_days if d >= first):
                gapped.append(t)

        degraded = len(gapped) > len(prices) * 0.2
        if gapped and not degraded:
            sample = ", ".join(gapped[:8]) + (" ..." if len(gapped) > 8 else "")
            print(f"[{spec['label']}] 거래일 누락으로 제외 {len(gapped)}종목: {sample}")
            prices = {t: df for t, df in prices.items() if t not in set(gapped)}
        elif degraded:
            # 이 정도면 종목이 아니라 실행 환경의 문제다. 개별 제외는 의미가 없고
            # (대부분이 빠진다) 그대로 두면 없던 교차가 신호로 잡힌다. 결과 전체에
            # 신뢰 불가 표시를 단다.
            miss = sorted({d.date().isoformat() for t in gapped
                           for d in check_days if d not in set(prices[t].index.normalize())})
            print(f"[{spec['label']}] ★ 거래일 결손이 광범위합니다: "
                  f"{len(gapped)}/{len(prices)}종목. 빠진 날짜: {', '.join(miss[:5])}"
                  + (" ..." if len(miss) > 5 else ""), file=sys.stderr)
            print(f"[{spec['label']}] ★ 없던 교차가 신호로 잡힐 수 있어 이 결과는 "
                  f"신뢰할 수 없습니다.", file=sys.stderr)

        last_dates = Counter(df.index.max().normalize() for df in prices.values())
        latest, n_at_latest = last_dates.most_common(1)[0]
        n_stale = len(prices) - n_at_latest
        note = f" (그 외 {n_stale}종목은 날짜 다름)" if n_stale else ""
        print(f"[{spec['label']}] 기준 거래일 {latest.date()} — {n_at_latest}종목{note}")

        info = meta.set_index("yahoo")[["ticker", "name", "sector"]].to_dict("index")
        n_before = len(rows)

        for yt, df in prices.items():
            sig = find_signals(df, args.ma, args.vol_window, args.vol_mult, args.price_basis,
                               require_hold=not args.no_hold)
            if sig.empty:
                continue
            # 최근 lookback 거래일 이내의 신호만
            cutoff = df.index[-args.lookback] if len(df.index) >= args.lookback else df.index[0]
            sig = sig[sig["date"] >= cutoff]
            for _, r in sig.iterrows():
                m = info.get(yt, {})
                dec = spec["decimals"]
                rows.append(
                    {
                        "market": mk,
                        "date": pd.Timestamp(r["date"]).date(),
                        "ticker": m.get("ticker", yt),
                        "name": m.get("name", ""),
                        "sector": m.get("sector", ""),
                        "close": round(r["close"], dec),
                        f"ma{args.ma}": round(r["ma"], dec),
                        "above_ma_%": round(r["above_ma_pct"], 2),
                        "volume": int(r["volume"]),
                        f"avg_vol_{args.vol_window}d": int(r["avg_vol"]),
                        "vol_ratio": round(r["vol_ratio"], 2),
                        "last_close": round(r["last_close"], dec),
                        "last_above_ma_%": round(r["last_pct"], 2),
                        "return_since_%": round((r["last_close"] / r["close"] - 1) * 100, 2),
                        "bars_since": int(r["bars_since"]),
                        "_y": yt,
                    }
                )
                if args.html:
                    c = build_series(df, r, args, m.get("ticker", yt), m.get("name", ""),
                                     m.get("sector", ""), market=mk)
                    c["_y"] = yt
                    charts.append(c)

        market_summaries.append({
            "id": mk,
            "label": spec["label"],
            "currency": spec["currency"],
            "decimals": spec["decimals"],
            "latest": str(latest.date()),
            "scanned": len(prices),
            "failed": n_failed,
            "noHistory": n_no_hist,
            "shortHistory": len(short),
            "gapped": len(gapped),
            "degraded": bool(degraded),
            "stale": n_stale,
            "hits": len(rows) - n_before,
        })

    for s in market_summaries:
        print(f"\n[{s['label']}] 조건 충족 {s['hits']}건 (기준 {s['latest']}, {s['scanned']}종목 스캔)")

    if not rows:
        print("\n조건을 만족하는 종목이 없습니다.")
        if args.html:
            write_html([], args, market_summaries, Path(args.html))
        return 0

    funda = fetch_fundamentals(sorted({r["_y"] for r in rows}))
    for r in rows:
        f = funda.get(r.pop("_y"), {})
        r["per"] = _num(f.get("per"))
        r["forward_per"] = _num(f.get("forward_per"))
        r["eps"] = _num(f.get("eps"))
        if not str(r.get("sector") or "").strip():
            r["sector"] = f.get("sector", "")
    for c in charts:
        f = funda.get(c.pop("_y"), {})
        c["per"] = _num(f.get("per"))
        c["forwardPer"] = _num(f.get("forward_per"))
        c["eps"] = _num(f.get("eps"))
        if not str(c.get("sector") or "").strip():
            c["sector"] = f.get("sector", "")

    res = pd.DataFrame(rows).sort_values(["market", "date", "vol_ratio"],
                                         ascending=[True, False, False])
    with pd.option_context("display.max_rows", None, "display.width", 240,
                           "display.max_colwidth", 22):
        print(f"\n합계 {len(res)}건\n")
        print(res.to_string(index=False))

    out_path = Path(args.out)
    res.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\n저장: {out_path.resolve()}")

    if args.html:
        charts.sort(key=lambda c: (c["date"], c["volRatio"]), reverse=True)
        write_html(charts, args, market_summaries, Path(args.html))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
