"""
S&P 500·KOSPI·KOSDAQ 150 이동평균선 돌파 스크리너

조건
  1) 종가가 120일 이동평균선을 상향 돌파 (전일: 종가 <= MA120, 당일: 종가 > MA120)
  2) 돌파 당일 거래량 >= 직전 5거래일(1주일) 평균 거래량 * 2

사용 예
  python sp500_breakout.py                    # 최신 거래일 기준 스캔
  python sp500_breakout.py --lookback 5       # 최근 5거래일 내 발생한 신호 모두
  python sp500_breakout.py --date 2026-08-20  # 특정 일자 기준
  python sp500_breakout.py --ma 60 --vol-mult 1.5 --vol-window 5
  python sp500_breakout.py --market sp500 --tickers AAPL,MSFT,NVDA
"""
from __future__ import annotations

import argparse
import hashlib
import io
from collections import Counter
import json
import math
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
from relative_strength import (  # noqa: E402
    MARKET_COUNTRY,
    RS_CSV_COLUMNS,
    calculate_relative_strength,
    csv_fields as rs_csv_fields,
    report_fields as rs_report_fields,
)

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


KOSDAQ150_CSV = CACHE_DIR / "kosdaq150_constituents.csv"
KODEX_KOSDAQ150 = "https://www.samsungfund.com/api/v1/kodex/product/2ETF54.do"


KRX_CORP_LIST = "https://kind.krx.co.kr/corpgeneral/corpList.do"
KOSPI_ALL_CSV = CACHE_DIR / "kospi_all_constituents.csv"


def load_kospi_all(refresh: bool = False) -> pd.DataFrame:
    """유가증권시장 전체 상장사. 지수 편입 명단에 기대지 않는다.

    코스피200 편입종목은 네이버 페이지가 폐지(HTTP 410)되면서 받을 곳이 없어졌고,
    증권사 오픈API도 지수 구성종목은 제공하지 않는다. 대신 KRX 상장법인목록을 쓴다.
    인증도 차단도 없고, 회사 단위 목록이라 우선주와 스팩이 섞여 들어오지 않는다.

    편입 명단이 필요 없으므로 분기 정기변경에도 손댈 일이 없다.
    """
    if not refresh:
        cached = _read_cached_csv(KOSPI_ALL_CSV, dtype={"ticker": str})
        if cached is not None:
            return cached

    try:
        html = _http_get(f"{KRX_CORP_LIST}?method=download&marketType=stockMkt",
                         encoding="cp949")
        tables = pd.read_html(io.StringIO(html))
        if not tables:
            raise RuntimeError("상장법인목록에서 표를 찾지 못했습니다")
        source = tables[0]
        for column in ("회사명", "종목코드"):
            if column not in source.columns:
                raise RuntimeError(f"'{column}' 열이 없습니다 (목록 형식 변경)")

        df = pd.DataFrame({
            "ticker": source["종목코드"].astype(str).str.strip().str.zfill(6),
            "name": source["회사명"].astype(str).str.strip(),
            "sector": source.get("업종", "").astype(str).str.strip(),
        })
        # 야후가 해석하지 못하는 신주인수권·영문 혼합 코드는 버린다. 소수이고,
        # 남겨두면 매 실행마다 시세 실패로 집계돼 경고를 흐린다.
        df = df[df["ticker"].str.fullmatch(r"\d{6}")]
        df = df[~df["name"].str.contains("스팩|기업인수목적", na=False)]
        df = df.drop_duplicates(subset="ticker").reset_index(drop=True)
        if len(df) < 500:
            raise RuntimeError(f"상장사 수가 비정상적으로 적습니다: {len(df)}")

        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        df.to_csv(KOSPI_ALL_CSV, index=False)
        return df
    except Exception as exc:
        cached = _read_cached_csv(KOSPI_ALL_CSV, max_age_days=float("inf"),
                                  dtype={"ticker": str})
        if cached is not None:
            print(f"[warn] 코스피 상장사 목록 갱신 실패({exc}). 캐시를 사용합니다.",
                  file=sys.stderr)
            return cached
        raise SystemExit(f"코스피 상장사 목록을 가져오지 못했습니다: {exc}")


def _validate_kosdaq150(df: pd.DataFrame) -> pd.DataFrame:
    """KODEX PDF가 현금이나 일부 종목만 담긴 응답이면 사용하지 않는다."""
    import re

    required = {"ticker", "name", "sector"}
    if not required.issubset(df.columns):
        raise ValueError("필수 열이 없습니다")
    checked = df.copy()
    checked["ticker"] = checked["ticker"].astype(str).str.strip().str.upper()
    checked["name"] = checked["name"].fillna("").astype(str).str.strip()
    valid_codes = checked["ticker"].map(lambda code: re.fullmatch(r"[A-Z0-9]{6}", code) is not None)
    if len(checked) != 150 or checked["ticker"].nunique() != 150 or not valid_codes.all():
        raise ValueError(f"주식 종목 수가 150개가 아닙니다: {len(checked)}")
    if (checked["name"] == "").any():
        raise ValueError("종목명이 비어 있습니다")
    return checked


def load_kosdaq150(refresh: bool = False) -> pd.DataFrame:
    """KODEX 코스닥150의 공개 PDF 구성내역에서 종목 코드와 이름을 받는다.

    KRX 구성종목 API는 로그인 세션이 필요해 무인 GitHub Actions에 맞지 않는다.
    KODEX 2ETF54 응답은 현금 행을 제외하면 코스닥150 주식 150개를 제공한다.
    """
    if not refresh:
        cached = _read_cached_csv(KOSDAQ150_CSV, dtype={"ticker": str})
        if cached is not None:
            try:
                return _validate_kosdaq150(cached)
            except ValueError as exc:
                print(f"[warn] 코스닥150 캐시가 유효하지 않아 새로 받습니다: {exc}", file=sys.stderr)

    try:
        import re

        payload = json.loads(_http_get(KODEX_KOSDAQ150))
        product = payload.get("info", {}).get("product", {})
        pdf = payload.get("pdf", {})
        holdings = pdf.get("list")
        product_name = (str(product.get("fNm") or "") + " "
                        + str(product.get("bmIdx") or "")).replace(" ", "")
        basis_date = str(pdf.get("gijunYMD") or "")
        if (product.get("fId") != "2ETF54" or "코스닥150" not in product_name
                or re.fullmatch(r"\d{8}", basis_date) is None or not isinstance(holdings, list)):
            raise RuntimeError("예상한 KODEX 코스닥150 응답이 아닙니다")
        for count_field in ("totalCnt", "nowCnt"):
            try:
                if int(pdf[count_field]) != len(holdings):
                    raise RuntimeError("KODEX 구성내역 응답이 일부만 내려왔습니다")
            except (KeyError, TypeError, ValueError):
                raise RuntimeError("KODEX 구성내역 전체 건수를 확인할 수 없습니다") from None

        found = {}
        for holding in holdings:
            if not isinstance(holding, dict):
                continue
            code = str(holding.get("itmNo") or "").strip().upper()
            name = str(holding.get("secNm") or "").strip()
            if re.fullmatch(r"[A-Z0-9]{6}", code) and name:
                found.setdefault(code, name)

        df = pd.DataFrame({"ticker": list(found), "name": list(found.values())})
        df["sector"] = ""
        df = _validate_kosdaq150(df)
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        df.to_csv(KOSDAQ150_CSV, index=False, encoding="utf-8")
        return df
    except Exception as exc:
        cached = _read_cached_csv(
            KOSDAQ150_CSV, max_age_days=float("inf"), dtype={"ticker": str})
        if cached is not None:
            try:
                cached = _validate_kosdaq150(cached)
                print(f"[warn] 코스닥150 목록 갱신 실패({exc}). 캐시를 사용합니다.", file=sys.stderr)
                return cached
            except ValueError:
                pass
        raise SystemExit(f"코스닥150 구성종목을 가져오지 못했습니다: {exc}")


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
    "kospi": {
        "label": "KOSPI",
        "currency": "KRW",
        "decimals": 0,
        "tz": "Asia/Seoul",
        "close": (15, 30),
        "loader": load_kospi_all,
        "to_yahoo": lambda t: f"{str(t).zfill(6)}.KS",
        # 유가증권 전체에는 하루 몇 천만 원어치만 거래되는 종목이 섞여 있다.
        # 그런 종목은 '직전 평균의 2배' 가 몇 백 주만으로도 성립해 신호가 아니라
        # 잡음이 된다. 거래대금 하한으로 실제로 사고팔 수 있는 종목만 남긴다.
        "min_turnover": 3_000_000_000,   # 최근 60거래일 평균 일거래대금 30억원
    },
    "kosdaq150": {
        "label": "KOSDAQ 150",
        "currency": "KRW",
        "decimals": 0,
        "tz": "Asia/Seoul",
        "close": (15, 30),
        "loader": load_kosdaq150,
        "to_yahoo": lambda t: f"{str(t).zfill(6)}.KQ",
    },
}


def to_yahoo(ticker: str) -> str:
    """BRK.B -> BRK-B 같은 야후 표기로 변환."""
    return ticker.replace(".", "-").upper()


def average_turnover(df, window: int = 60) -> float:
    """최근 거래일의 평균 일거래대금(종가 x 거래량). 시가총액 대신 쓴다.

    시가총액은 따로 받아와야 하지만 거래대금은 이미 가진 시세만으로 구해진다.
    돌파 신호가 실제로 체결 가능한 규모인지 보는 데는 이쪽이 더 직접적이다.
    """
    if df is None or getattr(df, "empty", True):
        return 0.0
    tail = df.tail(window)
    if tail.empty or "Close" not in tail or "Volume" not in tail:
        return 0.0
    value = (tail["Close"] * tail["Volume"]).median()
    return float(value) if pd.notna(value) else 0.0


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
def price_cache_path(tickers, start, end):
    key = hashlib.md5(
        f"priced-sessions-v2|{start}|{end}|{','.join(sorted(tickers))}".encode()
    ).hexdigest()[:12]
    return CACHE_DIR / f"prices_{key}.pkl"


def download_prices(tickers, start, end, chunk=60, use_cache=True, threads=True):
    """티커별 OHLCV DataFrame(dict) 반환.

    묶음으로 두 번 받아보고, 그래도 빠진 종목은 개별 요청으로 한 번 더 시도한다.
    """
    cache_path = None
    if use_cache:
        cache_path = price_cache_path(tickers, start, end)
        if cache_path.exists() and (time.time() - cache_path.stat().st_mtime) < 6 * 3600:
            try:
                with cache_path.open("rb") as fh:
                    cached = pickle.load(fh)
                if not isinstance(cached, dict):
                    raise ValueError("invalid price cache")
                cached = {t: df for t, df in cached.items()
                          if t in tickers and isinstance(df, pd.DataFrame) and not df.empty}
                download_prices.last_failed = sorted(set(tickers) - set(cached))
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
                    threads=threads,
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
                # Retain priced sessions for the monthly screen when volume is
                # missing. Daily volume rules validate that field separately.
                df = df.dropna(subset=["Close"])
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
                df = df.dropna(subset=["Close"])
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


def missing_price_sessions(prices, window, since=None):
    """Peer-observed sessions; never manufacture weekdays or fill missing prices.

    Include the moving-average warmup, not just the signal search window. A
    session absent from every supplied security is not detectable by this check.
    """
    if not prices:
        return {}
    counts = Counter()
    dates = {}
    for ticker, frame in prices.items():
        if frame.empty:
            continue
        index = frame.index
        if index.tz is not None:
            index = index.tz_localize(None)
        dates[ticker] = set(index.normalize())
        counts.update(dates[ticker])
    quorum = max(min(5, len(dates)), int(len(dates) * 0.03), 1)
    calendar = sorted(day for day, count in counts.items() if count >= quorum)
    required = set(calendar[-window:])
    if since is not None:
        required.update(day for day in calendar if day >= pd.Timestamp(since))
    return {ticker: sorted(day for day in required - have if day >= min(have))
            for ticker, have in dates.items()
            if any(day >= min(have) for day in required - have)}


def recover_price_sessions(prices, market, start, end, window, since=None, price_basis="adj"):
    """One bounded, uncached, serial retry for nonempty but incomplete feeds.

    Replace the whole history only after all known required sessions return.
    Splicing a new adjusted-price fragment into an old adjustment basis is unsafe.
    """
    gaps = missing_price_sessions(prices, window, since)
    if not gaps:
        return prices, {"detected": 0, "repaired": 0, "remaining": 0, "missingDates": []}
    missing_dates = sorted({day.date().isoformat() for days in gaps.values() for day in days})
    print(f"[{MARKETS[market]['label']}] 누락 이력 {len(gaps)}종목 재수집 · "
          f"날짜: {', '.join(missing_dates[:8])}", flush=True)
    fresh = download_prices(list(gaps), start, end, chunk=20, use_cache=False, threads=False)
    repaired = dict(prices)
    count = 0
    for ticker, days in gaps.items():
        frame = fresh.get(ticker)
        if frame is None or frame.empty:
            continue
        frame = drop_partial_bar(frame, market).sort_index()
        index = frame.index.tz_localize(None) if frame.index.tz is not None else frame.index
        frame = frame.copy()
        frame.index = index.normalize()
        frame = frame.loc[(frame.index >= pd.Timestamp(start)) & (frame.index <= pd.Timestamp(end))]
        if frame.index.has_duplicates or frame.index.hasnans:
            continue
        old_index = prices[ticker].index
        if old_index.tz is not None:
            old_index = old_index.tz_localize(None)
        required = set(old_index.normalize()) | set(days)
        if not required.issubset(set(frame.index)):
            continue
        # A dated all-null/invalid row is not a recovered session.
        if any(column not in frame or not (
                pd.to_numeric(frame.loc[sorted(required), column], errors="coerce")
                .map(lambda value: pd.notna(value) and math.isfinite(value) and value > 0).all())
               for column in (["Open", "High", "Low", "Close"]
                              + (["Adj Close"] if price_basis == "adj" else []))):
            continue
        repaired[ticker] = frame
        count += 1
    remaining = missing_price_sessions(repaired, window, since)
    print(f"[{MARKETS[market]['label']}] 누락 이력 복구 {count}/{len(gaps)}종목 · "
          f"재검사 잔여 {len(remaining)}종목", flush=True)
    return repaired, {"detected": len(gaps), "repaired": count,
                      "remaining": len(remaining), "missingDates": missing_dates}


def fetch_fundamentals(tickers):
    """조건 통과 종목의 기본 지표와 Yahoo 현재가를 종목당 한 번 조회한다."""
    out = {}
    for i, t in enumerate(tickers, 1):
        print(f"  현재가·PER 조회 {i}/{len(tickers)}", end="\r", flush=True)
        try:
            info = yf.Ticker(t).info
            current_price = _num(info.get("currentPrice"))
            if current_price is None or current_price <= 0:
                current_price = _num(info.get("regularMarketPrice"))
            out[t] = {
                "per": info.get("trailingPE"),
                "forward_per": info.get("forwardPE"),
                "eps": info.get("trailingEps"),
                "sector": info.get("sector") or "",
                "quote_price": current_price if current_price and current_price > 0 else None,
                "quote_as_of": _quote_as_of(info),
            }
        except Exception:
            out[t] = {"per": None, "forward_per": None, "eps": None, "sector": "",
                      "quote_price": None, "quote_as_of": None}
    print(" " * 40, end="\r")
    return out


def _num(v):
    """None/NaN/무한대를 걸러 float 또는 None 을 돌려준다."""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if pd.isna(f) or f in (float("inf"), float("-inf")) else round(f, 2)


def _quote_as_of(info):
    """Yahoo regularMarketTime을 거래소 현지 시각이 담긴 ISO 문자열로 바꾼다."""
    raw = info.get("regularMarketTime")
    try:
        stamp = float(raw)
        if not math.isfinite(stamp) or stamp <= 0:
            return None
        zone_name = info.get("exchangeTimezoneName") or "UTC"
        return datetime.fromtimestamp(stamp, tz=ZoneInfo(zone_name)).isoformat(timespec="seconds")
    except (KeyError, TypeError, ValueError, OverflowError):
        return None


def _quote_return(price, base):
    """Yahoo 현재가의 신호일 실제 종가 대비 수익률. 확정 봉 수익률과 분리한다."""
    quote = _num(price)
    basis = _num(base)
    if quote is None or quote <= 0 or basis is None or basis <= 0:
        return None
    return round((quote / basis - 1) * 100, 2)


def _text(v):
    """CSV의 빈 문자열은 재로딩 시 NaN일 수 있다."""
    return "" if v is None or pd.isna(v) else str(v)


# --------------------------------------------------------------------------
# 3. 신호 계산
# --------------------------------------------------------------------------
def signal_price(df, price_basis):
    """선택한 판정 기준을 보존한다. 잘못된 가격은 봉을 삭제하지 않고 결측으로 둔다."""
    if price_basis not in ("adj", "raw"):
        raise ValueError(f"지원하지 않는 가격 기준: {price_basis}")
    column = "Adj Close" if price_basis == "adj" else "Close"
    if column not in df:
        raise ValueError(f"{column} 열이 없어 {price_basis} 기준으로 판정할 수 없습니다")
    values = pd.to_numeric(df[column], errors="coerce").astype(float)
    return values.where(values.map(math.isfinite) & (values > 0))


def find_signals(df, ma_period, vol_window, vol_mult, price_basis, require_hold=True):
    """한 종목의 시계열에서 조건을 만족하는 모든 날짜를 반환."""
    price = signal_price(df, price_basis)
    raw_close = signal_price(df, "raw")
    volume = pd.to_numeric(df["Volume"], errors="coerce").astype(float)
    volume = volume.where(volume.map(math.isfinite) & (volume >= 0))

    ma = price.rolling(ma_period, min_periods=ma_period).mean()
    # 돌파 당일을 제외한 직전 vol_window 거래일의 평균 거래량
    avg_vol = volume.shift(1).rolling(vol_window, min_periods=vol_window).mean()

    crossed_up = (price > ma) & (price.shift(1) <= ma.shift(1)) & ma.notna() & ma.shift(1).notna()
    vol_ratio = volume / avg_vol.where(avg_vol > 0)
    hit = (crossed_up & (vol_ratio >= vol_mult) & vol_ratio.map(math.isfinite)
           & raw_close.notna()).fillna(False)

    if not hit.any():
        return pd.DataFrame()

    # 돌파 이후 현재가가 이동평균선 위를 지키고 있는지
    last_price, last_ma = price.iloc[-1], ma.iloc[-1]
    if pd.isna(last_price) or pd.isna(raw_close.iloc[-1]) or pd.isna(last_ma):
        return pd.DataFrame()
    holding = bool(pd.notna(last_ma) and last_price > last_ma)
    if require_hold and not holding:
        return pd.DataFrame()

    last_i = len(df) - 1
    positions = [i for i, v in enumerate(hit.values) if v]

    return pd.DataFrame(
        {
            "date": df.index[hit],
            "close": df["Close"].astype(float)[hit].values,
            "signal_close": price[hit].values,
            "ma": ma[hit].values,
            "above_ma_pct": (price[hit].values / ma[hit].values - 1) * 100,
            "volume": volume[hit].values,
            "avg_vol": avg_vol[hit].values,
            "vol_ratio": vol_ratio[hit].values,
            "last_close": float(df["Close"].astype(float).iloc[-1]),
            "last_signal_close": float(last_price),
            "last_ma": float(last_ma) if pd.notna(last_ma) else float("nan"),
            "last_pct": (float(last_price) / float(last_ma) - 1) * 100 if pd.notna(last_ma) else float("nan"),
            "bars_since": [last_i - p for p in positions],
            "holding": holding,
        }
    )


# --------------------------------------------------------------------------
# 4. HTML 리포트 (휴대폰 브라우저용)
# --------------------------------------------------------------------------
CHART_WINDOW = 70  # 기본 차트 길이. 오래된 돌파일은 포함하도록 확장한다.


def build_series(df, sig_row, args, ticker, name, sector, market="sp500"):
    """카드 차트에 넣을 시계열 조각을 만든다."""
    price = signal_price(df, args.price_basis)
    ma = price.astype(float).rolling(args.ma, min_periods=args.ma).mean()
    # 캔들과 MA를 같은 판정 기준으로 그린다. 표시용 실제 종가는 별도 보존한다.
    factor = price / signal_price(df, "raw")

    end_pos = len(df.index)
    sig_date = pd.Timestamp(sig_row["date"])
    absolute_sig_pos = int(df.index.get_loc(sig_date))
    start_pos = min(max(0, end_pos - CHART_WINDOW), absolute_sig_pos)
    window = df.index[start_pos:end_pos]
    sig_pos = absolute_sig_pos - start_pos

    def clean(series):
        vals = series.iloc[start_pos:end_pos]
        return [None if pd.isna(v) or not math.isfinite(float(v)) else round(float(v), 4)
                for v in vals]

    return {
        "market": market,
        "ticker": _text(ticker),
        "name": _text(name),
        "sector": _text(sector),
        "date": str(pd.Timestamp(sig_row["date"]).date()),
        "close": round(float(sig_row["close"]), 2),
        "signalClose": round(float(sig_row["signal_close"]), 4),
        "priceBasis": args.price_basis,
        "ma": round(float(sig_row["ma"]), 2),
        "abovePct": round(float(sig_row["above_ma_pct"]), 2),
        "volume": int(sig_row["volume"]),
        "avgVol": int(sig_row["avg_vol"]),
        "volRatio": round(float(sig_row["vol_ratio"]), 2),
        "lastClose": round(float(sig_row["last_close"]), 2),
        "lastSignalClose": round(float(sig_row["last_signal_close"]), 4),
        "lastMa": round(float(sig_row["last_ma"]), 2),
        "lastPct": round(float(sig_row["last_pct"]), 2),
        # 돌파일 종가 대비 현재 종가 등락률 (표시되는 실제 종가끼리의 비교)
        "retPct": round((float(sig_row["last_close"]) / float(sig_row["close"]) - 1) * 100, 2),
        "barsSince": int(sig_row["bars_since"]),
        "sigIndex": sig_pos,
        "dates": [d.strftime("%Y-%m-%d") for d in window],
        "opens": clean(pd.to_numeric(df["Open"], errors="coerce") * factor),
        "highs": clean(pd.to_numeric(df["High"], errors="coerce") * factor),
        "lows": clean(pd.to_numeric(df["Low"], errors="coerce") * factor),
        "closes": clean(price),
        "mas": clean(ma),
        "volumes": [int(v) if pd.notna(v) and math.isfinite(float(v)) and v >= 0 else None
                    for v in pd.to_numeric(df["Volume"], errors="coerce").iloc[start_pos:end_pos]],
    }


PAGES_URL = "https://kimka3.github.io/breakout-screener/"


def write_summary(rows, markets, args, out_path: Path, top=8):
    """텔레그램 등으로 보낼 짧은 텍스트 요약.

    종목이 많아도 메시지가 길어지지 않도록 시장별 상위 몇 개만 싣고,
    나머지는 개수로만 알린다. 자세한 건 링크에서 본다.
    """
    when = datetime.now(ZoneInfo("Asia/Seoul")).strftime("%Y-%m-%d %H:%M")
    lines = [f"[돌파 스크리너] {when} KST", "", f"{args.ma}일선 돌파", ""]
    if getattr(args, "date", None):
        lines.extend([f"과거 가격 기준 {args.date} · 구성종목과 재무지표는 현재 조회 자료", ""])

    for m in markets:
        hits = [r for r in rows if r["market"] == m["id"]]
        head = f"{m['label']} · 기준 {m['latest']} · {len(hits)}건"
        if m.get("degraded"):
            head += "  ※ 데이터 결손으로 신뢰 불가"
        lines.append(head)

        if not hits:
            lines.append("  해당 없음")
        else:
            dec = m["decimals"]
            for r in sorted(hits, key=lambda x: -x["vol_ratio"])[:top]:
                ret = r["return_since_%"]
                rs = (f"  RS {int(r['rs_rating'])}"
                      if isinstance(r.get("rs_rating"), (int, float))
                      and math.isfinite(r["rs_rating"]) else "")
                lines.append(
                    f"  {r['ticker']} {r['name'][:14]}"
                    f"  {r['vol_ratio']:.2f}x"
                    f"{rs}"
                    f"  {r['last_close']:,.{dec}f}"
                    f"  ({ret:+.1f}%)"
                )
            if len(hits) > top:
                lines.append(f"  … 외 {len(hits) - top}건")
        lines.append("")

    monthly = getattr(args, "monthly_payload", None)
    if monthly is not None:
        lines.extend(["10개월선 장기 돌파 · 직전 6개월 이평선 아래 · 거래량 > 직전 3개월 평균", ""])
        for m in monthly["markets"]:
            hits = [h for h in monthly["hits"] if h["market"] == m["id"]]
            lines.append(f"{m['label']} · 기준월 {m['targetMonth']} · {len(hits)}건")
            if not hits:
                lines.append("  해당 없음")
            for h in sorted(hits, key=lambda h: (-h["abovePct"], h["ticker"]))[:top]:
                rs = h.get("rs") or {}
                rs_label = f"  RS {rs['rating']}" if isinstance(rs.get("rating"), int) else ""
                lines.append(f"  {h['ticker']} {h['name'][:14]}  이격 {h['abovePct']:+.2f}%"
                             f"{rs_label}")
            if len(hits) > top:
                lines.append(f"  … 외 {len(hits) - top}건")
            lines.append("")
    lines.append(PAGES_URL)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"요약: {out_path.resolve()}")


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
        "id": "daily",
        "label": f"{args.ma}일선 돌파",
        "timeframe": "day",
        "generatedAt": datetime.now(ZoneInfo("Asia/Seoul")).strftime("%Y-%m-%d %H:%M") + " KST",
        "historicalAsOf": getattr(args, "date", None),
        "fundamentalsAsOf": getattr(args, "fundamentals_as_of", None),
        "markets": markets,
        "maPeriod": args.ma,
        "volWindow": args.vol_window,
        "volMult": args.vol_mult,
        "lookback": args.lookback,
        "priceBasis": args.price_basis,
        "requireHold": not args.no_hold,
        "hits": charts,
    }
    monthly = getattr(args, "monthly_payload", None)
    if monthly is not None:
        # Retain the legacy daily keys for consumers of earlier report payloads.
        payload["screens"] = [dict(payload), monthly]
    # 외부 종목명에 </script>가 있어도 인라인 데이터 영역을 벗어나지 않게 한다.
    encoded = json.dumps(payload, ensure_ascii=False, allow_nan=False,
                         separators=(",", ":")).replace("<", "\\u003c")
    fragment = template_path.read_text(encoding="utf-8").replace("/*__DATA__*/null", encoded)

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
        '<meta name="apple-mobile-web-app-title" content="돌파 스크리너">\n'
        '<meta name="theme-color" media="(prefers-color-scheme: light)" content="#EEF1F2">\n'
        '<meta name="theme-color" media="(prefers-color-scheme: dark)" content="#0E1315">\n'
        "</head>\n<body>\n" + fragment + "\n</body>\n</html>\n"
    )
    out_path.write_text(standalone, encoding="utf-8")
    print(f"HTML: {out_path.resolve()}")
    print(f"      {artifact_path.resolve()}  (돌파 스크리너 삽입용 조각)")


# --------------------------------------------------------------------------
# 5. 메인
# --------------------------------------------------------------------------
def result_columns(args):
    """신호가 없어도 동일한 CSV 스키마를 기록한다."""
    return ["market", "date", "ticker", "name", "sector", "close", "signal_close",
            f"ma{args.ma}", "above_ma_%", "volume", f"avg_vol_{args.vol_window}d",
            "vol_ratio", "last_close", "last_signal_close", "last_above_ma_%",
            "return_since_%", "bars_since", "price_basis", *RS_CSV_COLUMNS,
            "per", "forward_per", "eps",
            "fundamentals_as_of", "quote_price", "quote_as_of",
            "quote_return_since_%"]


def main() -> int:
    p = argparse.ArgumentParser(
        description="S&P 500·KOSPI·KOSDAQ 150 이동평균선 돌파 스크리너",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--ma", type=int, default=120, help="이동평균 기간(거래일)")
    p.add_argument("--vol-window", type=int, default=5, help="직전 평균거래량 산출 기간(거래일)")
    p.add_argument("--vol-mult", type=float, default=2.0, help="거래량 배수 기준")
    p.add_argument("--lookback", type=int, default=1, help="최근 N 거래일 내의 신호를 검색")
    p.add_argument("--date", default=None, help="기준일 YYYY-MM-DD (미지정시 최신 거래일)")
    p.add_argument("--price-basis", choices=["adj", "raw"], default="adj",
                   help="신호 계산 기준가: adj=수정주가(배당/분할 반영), raw=종가 그대로")
    p.add_argument("--market", choices=["all", *MARKETS], default="all",
                   help="스캔할 시장")
    p.add_argument("--with-monthly", action="store_true",
                   help="마지막 마감 월봉의 10개월선 장기 돌파를 함께 검사")
    p.add_argument("--monthly-out", default="monthly_breakouts.csv",
                   help="장기 돌파 결과 CSV 경로 (--with-monthly 사용 시)")
    p.add_argument("--tickers", default=None, help="쉼표구분 티커 목록(테스트용)")
    p.add_argument("--refresh-list", action="store_true", help="구성종목 목록 강제 갱신")
    p.add_argument("--min-turnover", type=float, default=None,
                   help="일평균 거래대금 하한(억원). 시장 기본값을 덮어씁니다. 0 이면 필터 해제")
    p.add_argument("--out", default="breakouts.csv", help="결과 CSV 경로")
    p.add_argument("--html", nargs="?", const="report.html", default=None,
                   help="휴대폰용 HTML 리포트 생성 (경로 생략시 report.html)")
    p.add_argument("--summary", default=None,
                   help="알림용 텍스트 요약 파일 경로 (텔레그램 등)")
    p.add_argument("--no-cache", action="store_true", help="시세 캐시를 쓰지 않고 새로 받기")
    p.add_argument("--no-hold", action="store_true",
                   help="돌파 후 현재가가 이동평균선 아래로 다시 내려간 종목도 포함")
    args = p.parse_args()
    if args.with_monthly and Path(args.out).resolve() == Path(args.monthly_out).resolve():
        p.error("일봉과 월봉 결과 CSV 경로는 서로 달라야 합니다")
    for name in ("ma", "vol_window", "lookback"):
        if getattr(args, name) <= 0:
            p.error(f"--{name.replace('_', '-')}는 양의 정수여야 합니다")
    if not math.isfinite(args.vol_mult) or args.vol_mult <= 0:
        p.error("--vol-mult는 유한한 양수여야 합니다")
    if args.tickers is not None:
        if args.market == "all":
            p.error("--tickers에는 --market sp500, kospi 또는 kosdaq150을 지정하세요")
        tickers = list(dict.fromkeys(t.strip().upper() for t in args.tickers.split(",") if t.strip()))
        if not tickers:
            p.error("--tickers에 하나 이상의 티커를 입력하세요")
    try:
        end_date = (datetime.strptime(args.date, "%Y-%m-%d").date() if args.date
                    else datetime.now(ZoneInfo("Asia/Seoul")).date())
    except ValueError:
        p.error("--date는 YYYY-MM-DD 형식의 유효한 날짜여야 합니다")
    # MA 기간 + 여유분 확보 (거래일 -> 달력일 환산 약 1.55배 + 버퍼)
    # HTML 리포트는 차트 구간 내내 MA 선이 그려져야 하므로 그만큼 더 받는다.
    span = args.ma + args.vol_window + args.lookback + (CHART_WINDOW if args.html else 0)
    start_date = min(
        end_date - timedelta(days=int(span * 1.55) + 40),
        end_date - timedelta(days=400),  # 오닐식 RS의 12개월 4개 분기 계산 여유분
    )
    if args.with_monthly:
        # Full calendar months for monthly OHLCV and a useful long-term chart.
        start_date = min(start_date, (pd.Period(end_date, freq="M") - 48).start_time.date())

    print(f"기간: {start_date} ~ {end_date}")
    print(f"조건: 종가 > MA{args.ma} 상향돌파 & 거래량 >= 직전 {args.vol_window}거래일 평균 x {args.vol_mult}"
          + ("" if args.no_hold else f" & 현재가 > MA{args.ma} 유지"))

    selected = list(MARKETS) if args.market == "all" else [args.market]
    rows, charts, market_summaries = [], [], []
    monthly_rows, monthly_charts, monthly_markets = [], [], []
    rs_frames = {}
    scan_errors = []

    for mk in selected:
        spec = MARKETS[mk]
        if args.tickers is not None:
            meta = pd.DataFrame({"ticker": tickers})
            meta["name"] = ""
            meta["sector"] = ""
        else:
            meta = spec["loader"](refresh=args.refresh_list)
        meta["ticker"] = meta["ticker"].astype(str)
        for column in ("name", "sector"):
            meta[column] = meta[column].fillna("").astype(str)
        meta["yahoo"] = meta["ticker"].map(spec["to_yahoo"])

        print(f"\n[{spec['label']}] 대상 {len(meta)}종목")
        prices = download_prices(meta["yahoo"].tolist(), start_date, end_date,
                                 use_cache=not args.no_cache)
        n_failed = len(set(meta["yahoo"]) - set(prices))
        # Monthly charts also show the latest completed daily close. Remove the
        # in-progress daily bar before either strategy consumes the same feed.
        prices = {t: drop_partial_bar(df, mk) for t, df in prices.items()}
        downloaded_prices = prices
        download_tickers = meta["yahoo"].tolist()

        # 지수 편입 명단 대신 시장 전체를 받는 경우, 하루 몇 천만 원어치만
        # 거래되는 종목이 섞인다. 그런 종목은 '직전 평균 거래량의 2배' 가 몇 백
        # 주만으로 성립해 신호가 아니라 잡음이 된다. 실제로 사고팔 수 있는
        # 규모만 남긴다. 걸러낸 수는 출력에 남겨 조용히 줄어들지 않게 한다.
        floor = (args.min_turnover * 1e8 if args.min_turnover is not None
                 else spec.get("min_turnover"))
        if floor:
            liquid = {tk: df for tk, df in prices.items()
                      if average_turnover(df) >= floor}
            n_thin = len(prices) - len(liquid)
            if n_thin:
                print(f"[{spec['label']}] 거래대금 하한({floor / 1e8:.0f}억) 미달 "
                      f"{n_thin}종목 제외 — {len(liquid)}종목 스캔")
            prices = liquid
            meta = meta[meta["yahoo"].isin(prices)].reset_index(drop=True)

        # Validate and recover before monthly bars, RS, or daily signals consume
        # the data. Nonempty responses and cache hits can still omit sessions.
        monthly_as_of = end_date if args.date else datetime.now(ZoneInfo(spec["tz"]))
        monthly_since = ((pd.Period(pd.Timestamp(monthly_as_of).date(), freq="M") - 16)
                         .start_time if args.with_monthly else None)
        quality_window = max(args.ma, args.vol_window) + args.lookback + 1
        prices, recovery = recover_price_sessions(
            prices, mk, start_date, end_date, quality_window, monthly_since, args.price_basis)
        if recovery["repaired"] and not args.no_cache:
            try:
                downloaded_prices.update(prices)
                CACHE_DIR.mkdir(parents=True, exist_ok=True)
                with price_cache_path(download_tickers, start_date, end_date).open("wb") as fh:
                    pickle.dump(downloaded_prices, fh)
            except OSError as exc:
                print(f"[warn] 복구 시세 캐시 저장 실패: {exc}", file=sys.stderr)

        rs_frames.update({(mk, ticker): frame for ticker, frame in prices.items()
                          if isinstance(frame, pd.DataFrame) and not frame.empty})
        if args.with_monthly:
            from monthly_breakout import scan_monthly_market

            # On the first KST day, the US can still be trading the previous month.
            # Use each market's local calendar so that a partial month is never final.
            monthly_as_of = end_date if args.date else datetime.now(ZoneInfo(spec["tz"]))
            mr, mc, ms = scan_monthly_market(prices, meta, mk, monthly_as_of, args.price_basis)
            monthly_rows.extend(mr)
            monthly_charts.extend(mc)
            monthly_markets.append(ms)
            print(f"[{spec['label']} / 장기] 기준월 {ms['targetMonth']} · "
                  f"{ms['scanned']}종목 평가 · {ms['hits']}건")
            if not ms["scanned"] or ms.get("degraded"):
                scan_errors.append(spec["label"] + " 장기 월봉")
        # 과거 데이터가 짧으면 이동평균이 계산되지 않아 신호가 조용히 사라진다.
        # MA 는 앞선 ma 봉이 있어야 나오므로, 검색 구간 전체를 평가하려면
        # ma + lookback 봉이 필요하다. 그에 못 미치는 종목을 따로 센다.
        min_history = max(args.ma, args.vol_window)
        need_full = min_history + args.lookback
        n_no_hist = sum(1 for df in prices.values() if len(df) <= min_history)
        prices = {t: df for t, df in prices.items() if len(df) > min_history}
        invalid = []
        for t, df in prices.items():
            try:
                basis = signal_price(df, args.price_basis)
                raw_close = signal_price(df, "raw")
                if pd.isna(basis.iloc[-1]) or pd.isna(raw_close.iloc[-1]):
                    raise ValueError("마지막 봉 가격이 유효하지 않습니다")
                ma = basis.rolling(args.ma, min_periods=args.ma).mean()
                volumes = pd.to_numeric(df["Volume"], errors="coerce").astype(float)
                volumes = volumes.where(volumes.map(math.isfinite) & (volumes >= 0))
                avg_vol = volumes.shift(1).rolling(args.vol_window,
                                                   min_periods=args.vol_window).mean()
                evaluable = (basis.notna() & basis.shift(1).notna() & raw_close.notna()
                             & ma.notna() & ma.shift(1).notna() & volumes.notna()
                             & (avg_vol > 0))
                if pd.isna(ma.iloc[-1]) or not evaluable.iloc[-args.lookback:].any():
                    raise ValueError("검색 구간의 가격·거래량으로 조건을 평가할 수 없습니다")
            except (KeyError, ValueError) as exc:
                invalid.append(t)
                print(f"[{spec['label']}] {t} 판정 불가 — 제외: {exc}", file=sys.stderr)
        prices = {t: df for t, df in prices.items() if t not in set(invalid)}
        short = {t: len(df) for t, df in prices.items() if len(df) < need_full}
        print(f"[{spec['label']}] 시세 확보 {len(prices)}종목"
              + (f" (다운로드 실패 {n_failed}종목 — 스캔 제외)" if n_failed else ""))
        if n_no_hist:
            print(f"[{spec['label']}] 계산에 필요한 이력이 부족해 제외 {n_no_hist}종목")
        if short:
            sample = ", ".join(f"{t}({n}봉)" for t, n in list(short.items())[:8])
            print(f"[{spec['label']}] 이력이 짧아 검색 구간 일부만 평가됨 {len(short)}종목: {sample}"
                  + (" ..." if len(short) > 8 else ""))
        if not prices:
            scan_errors.append(spec["label"])
            continue

        # 기준 거래일은 최댓값이 아니라 최빈값으로 잡는다. 한두 종목이 남들보다
        # 하루 앞선 봉을 갖고 있는 경우가 있어, 최댓값을 쓰면 헤더에 찍히는 날짜가
        # 정작 대부분 종목의 데이터보다 하루 앞서게 된다.
        # 야후가 종목에 따라 거래일을 통째로 빠뜨리는 일이 있다(환경에 따라 다르다).
        # 빠진 날이 검색 구간 안에 있으면 "전일 아래 → 당일 위" 가 실제로는
        # 없었는데 있는 것처럼 보여 가짜 돌파가 만들어진다. 놓치는 것보다 나쁘다.
        # 전 종목의 날짜 분포로 그 시장의 거래일 달력을 만들어 대조한다.
        gaps = missing_price_sessions(prices, quality_window)
        gapped = list(gaps)

        degraded = len(gapped) > len(prices) * 0.2
        if gapped and not degraded:
            sample = ", ".join(gapped[:8]) + (" ..." if len(gapped) > 8 else "")
            print(f"[{spec['label']}] 거래일 누락으로 제외 {len(gapped)}종목: {sample}")
            prices = {t: df for t, df in prices.items() if t not in set(gapped)}
        elif degraded:
            miss = sorted({d.date().isoformat() for days in gaps.values() for d in days})
            print(f"[{spec['label']}] ★ 거래일 결손이 광범위합니다: "
                  f"{len(gapped)}/{len(prices)}종목. 빠진 날짜: {', '.join(miss[:5])}"
                  + (" ..." if len(miss) > 5 else ""), file=sys.stderr)
            print(f"[{spec['label']}] ★ 없던 교차가 신호로 잡힐 수 있어 이 결과는 "
                  f"게시하지 않습니다.", file=sys.stderr)
            scan_errors.append(spec["label"] + " 일봉")
            continue

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
                        "signal_close": round(r["signal_close"], 4),
                        f"ma{args.ma}": round(r["ma"], dec),
                        "above_ma_%": round(r["above_ma_pct"], 2),
                        "volume": int(r["volume"]),
                        f"avg_vol_{args.vol_window}d": int(r["avg_vol"]),
                        "vol_ratio": round(r["vol_ratio"], 2),
                        "last_close": round(r["last_close"], dec),
                        "last_signal_close": round(r["last_signal_close"], 4),
                        "last_above_ma_%": round(r["last_pct"], 2),
                        "return_since_%": round((r["last_close"] / r["close"] - 1) * 100, 2),
                        "bars_since": int(r["bars_since"]),
                        "price_basis": args.price_basis,
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
            "invalidData": len(invalid),
            "noHistory": n_no_hist,
            "shortHistory": len(short),
            "gapped": len(gapped),
            "recovery": recovery,
            "degraded": bool(degraded),
            "stale": n_stale,
            "hits": len(rows) - n_before,
        })

    if scan_errors:
        print("[error] 평가 불가 또는 광범위한 데이터 결손이 있는 시장: " + ", ".join(scan_errors)
              + ". 정상적인 0건 결과와 구분하기 위해 출력을 갱신하지 않습니다.", file=sys.stderr)
        return 1

    for s in market_summaries:
        print(f"\n[{s['label']}] 조건 충족 {s['hits']}건 (기준 {s['latest']}, {s['scanned']}종목 스캔)")

    if not rows:
        print("\n조건을 만족하는 종목이 없습니다.")

    rs_ratings, rs_groups = calculate_relative_strength(rs_frames, args.price_basis)
    all_rows = rows + monthly_rows
    all_charts = charts + monthly_charts
    for row in all_rows:
        row.update(rs_csv_fields(rs_ratings.get((row["market"], row["_y"]))))
    for chart in all_charts:
        chart["rs"] = rs_report_fields(
            rs_ratings.get((chart["market"], chart["_y"])))
    for summary in market_summaries + monthly_markets:
        group = rs_groups.get(MARKET_COUNTRY.get(summary["id"]), {})
        summary["rsAsOf"] = group.get("asOf")
        summary["rsUniverse"] = group.get("universe")
        summary["rsUniverseSize"] = group.get("universeSize", 0)
        summary["rsRanked"] = group.get("rankedByMarket", {}).get(summary["id"], 0)

    args.fundamentals_as_of = (datetime.now(ZoneInfo("Asia/Seoul")).strftime("%Y-%m-%d %H:%M")
                               + " KST") if all_rows else None
    funda = fetch_fundamentals(sorted({r["_y"] for r in all_rows})) if all_rows else {}
    for r in all_rows:
        f = funda.get(r.pop("_y"), {})
        r["fundamentals_as_of"] = args.fundamentals_as_of
        r["per"] = _num(f.get("per"))
        r["forward_per"] = _num(f.get("forward_per"))
        r["eps"] = _num(f.get("eps"))
        r["quote_price"] = _num(f.get("quote_price"))
        r["quote_as_of"] = _text(f.get("quote_as_of")) or None
        r["quote_return_since_%"] = _quote_return(r["quote_price"], r.get("close"))
        if not str(r.get("sector") or "").strip():
            r["sector"] = _text(f.get("sector"))
    for c in all_charts:
        f = funda.get(c.pop("_y"), {})
        c["per"] = _num(f.get("per"))
        c["forwardPer"] = _num(f.get("forward_per"))
        c["eps"] = _num(f.get("eps"))
        c["quotePrice"] = _num(f.get("quote_price"))
        c["quoteAsOf"] = _text(f.get("quote_as_of")) or None
        c["quoteReturnPct"] = _quote_return(c["quotePrice"], c.get("close"))
        if not str(c.get("sector") or "").strip():
            c["sector"] = _text(f.get("sector"))

    res = pd.DataFrame(rows, columns=result_columns(args)).sort_values(["market", "date", "vol_ratio"],
                                         ascending=[True, False, False])
    with pd.option_context("display.max_rows", None, "display.width", 240,
                           "display.max_colwidth", 22):
        print(f"\n합계 {len(res)}건\n")
        print(res.to_string(index=False))

    out_path = Path(args.out)
    res.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\n저장: {out_path.resolve()}")

    if args.with_monthly:
        from monthly_breakout import monthly_result_columns

        monthly_res = pd.DataFrame(monthly_rows, columns=monthly_result_columns())
        if not monthly_res.empty:
            monthly_res = monthly_res.sort_values(["market", "date", "above_ma_%", "ticker"],
                                                 ascending=[True, False, False, True])
        monthly_res.to_csv(Path(args.monthly_out), index=False, encoding="utf-8-sig")
        monthly_charts.sort(key=lambda c: (-pd.Timestamp(c["date"]).value, -c["abovePct"], c["ticker"]))
        args.monthly_payload = {
            "id": "monthly", "label": "10개월선 장기 돌파", "timeframe": "month",
            "generatedAt": datetime.now(ZoneInfo("Asia/Seoul")).strftime("%Y-%m-%d %H:%M") + " KST",
            "historicalAsOf": args.date, "fundamentalsAsOf": args.fundamentals_as_of,
            "markets": monthly_markets, "hits": monthly_charts,
            "maPeriod": 10, "belowMonths": 6, "volWindow": 3, "volMult": 1,
            "volumeFilter": True, "volumeComparison": "gt",
            "lookback": 1, "priceBasis": args.price_basis, "requireHold": False,
        }
        print(f"장기 CSV: {Path(args.monthly_out).resolve()} ({len(monthly_res)}건)")

    if args.html:
        charts.sort(key=lambda c: (c["date"], c["volRatio"]), reverse=True)
        write_html(charts, args, market_summaries, Path(args.html))
    if args.summary:
        write_summary(rows, market_summaries, args, Path(args.summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
