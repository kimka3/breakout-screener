"""Regressions for a nonempty feed missing a session across most tickers."""
from contextlib import redirect_stderr, redirect_stdout
from datetime import date
import io
import pickle
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

from test_screener import price_frame, report_payload, screener


def feeds(total=20, damaged=12):
    complete = price_frame([10] * 18 + [12, 13], [100] * 18 + [500, 100])
    missing = complete.index[-3]
    prices = {f'T{i}': complete.drop(index=missing) if i < damaged else complete.copy()
              for i in range(total)}
    return complete, missing, prices


class RecoveryTests(unittest.TestCase):
    def test_441_nonempty_partial_histories_are_detected_and_replaced(self):
        complete, missing, prices = feeds(501, 441)
        fresh = {ticker: complete.copy() for ticker in list(prices)[:441]}
        # A revised adjusted history must replace the old basis in its entirety.
        for frame in fresh.values():
            frame['Adj Close'] *= .5
        with patch.object(screener, 'download_prices', return_value=fresh) as retry, \
             redirect_stdout(io.StringIO()):
            result, audit = screener.recover_price_sessions(
                prices, 'sp500', date(2026, 1, 1), date(2026, 2, 1), 20)
        self.assertEqual(audit['detected'], 441)
        self.assertEqual(audit['repaired'], 441)
        self.assertEqual(audit['remaining'], 0)
        self.assertEqual(audit['missingDates'], [str(missing.date())])
        retry.assert_called_once()
        self.assertFalse(retry.call_args.kwargs['use_cache'])
        self.assertFalse(retry.call_args.kwargs['threads'])
        pd.testing.assert_frame_equal(result['T0'], fresh['T0'], check_freq=False)
        self.assertEqual(result['T0']['Adj Close'].iloc[0], 5)

    def test_unrecovered_null_or_shorter_retry_never_fabricates_a_bar(self):
        complete, missing, prices = feeds()
        for kind in ('unchanged', 'null', 'shorter'):
            with self.subTest(kind=kind):
                frame = prices['T0'].copy() if kind == 'unchanged' else complete.copy()
                if kind == 'null':
                    frame.loc[missing, 'Close'] = float('nan')
                if kind == 'shorter':
                    frame = frame.iloc[1:]
                with patch.object(screener, 'download_prices', return_value={'T0': frame}), \
                     redirect_stdout(io.StringIO()):
                    result, audit = screener.recover_price_sessions(
                        prices, 'sp500', date(2026, 1, 1), date(2026, 2, 1), 20)
                self.assertEqual(audit['repaired'], 0)
                self.assertEqual(audit['remaining'], 12)
                self.assertNotIn(missing, result['T0'].index)

    def test_gap_in_ma_warmup_is_checked_even_outside_signal_window(self):
        complete = price_frame([10] * 180)
        missing = complete.index[-100]
        prices = {f'T{i}': complete.drop(index=missing) if i < 12 else complete
                  for i in range(20)}
        self.assertEqual(screener.missing_price_sessions(prices, 27), {})
        self.assertEqual(len(screener.missing_price_sessions(prices, 141)), 12)

    def test_raw_recovery_does_not_require_adjusted_close(self):
        complete, missing, prices = feeds()
        fresh = {t: complete.drop(columns='Adj Close') for t in list(prices)[:12]}
        with patch.object(screener, 'download_prices', return_value=fresh), \
             redirect_stdout(io.StringIO()):
            result, audit = screener.recover_price_sessions(
                prices, 'sp500', date(2026, 1, 1), date(2026, 2, 1), 20, price_basis='raw')
        self.assertEqual(audit['repaired'], 12)
        self.assertIn(missing, result['T0'].index)

    def test_peer_holidays_and_prelisting_dates_do_not_trigger_retry(self):
        complete, missing, prices = feeds()
        holiday = {t: complete.drop(index=missing) for t in prices}
        holiday['IPO'] = holiday['T0'].iloc[-4:]
        with patch.object(screener, 'download_prices') as retry:
            _, audit = screener.recover_price_sessions(
                holiday, 'sp500', date(2026, 1, 1), date(2026, 2, 1), 20)
        self.assertEqual(audit['detected'], 0)
        retry.assert_not_called()


class PublicationGateTests(unittest.TestCase):
    def run_case(self, recover):
        complete, missing, prices = feeds()
        fresh = {t: complete for t in list(prices)[:12]} if recover else {}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = [root / name for name in ('daily.csv', 'report.html', 'summary.txt', 'monthly.csv')]
            for path in paths:
                path.write_text('previous successful report', encoding='utf-8')
            args = ['sp500_breakout.py', '--market', 'sp500', '--tickers', ','.join(prices),
                    '--date', '2026-02-01', '--ma', '3', '--vol-window', '2', '--lookback', '5',
                    '--out', str(paths[0]), '--html', str(paths[1]), '--summary', str(paths[2]),
                    '--monthly-out', str(paths[3])]
            with patch.object(sys, 'argv', args), \
                 patch.object(screener, 'CACHE_DIR', root / 'cache'), \
                 patch.object(screener, 'download_prices', side_effect=[prices, fresh]), \
                 patch.object(screener, 'fetch_fundamentals', return_value={}) as funda, \
                 redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                code = screener.main()
            if recover:
                self.assertEqual(code, 0)
                market = report_payload(paths[1])['markets'][0]
                self.assertEqual(market['gapped'], 0)
                self.assertFalse(market['degraded'])
                self.assertEqual(market['recovery']['repaired'], 12)
                cache_paths = list((root / 'cache').glob('prices_*.pkl'))
                self.assertEqual(len(cache_paths), 1)
                cached = pickle.loads(cache_paths[0].read_bytes())
                self.assertEqual(set(cached), set(prices))
                self.assertTrue(all(missing in frame.index for frame in cached.values()))
            else:
                self.assertEqual(code, 1)
                funda.assert_not_called()
                for path in paths:
                    self.assertEqual(path.read_text(encoding='utf-8'), 'previous successful report')

    def test_unresolved_widespread_gaps_fail_without_overwriting_any_report(self):
        self.run_case(False)

    def test_successful_recovery_recalculates_and_caches_before_publication(self):
        self.run_case(True)


if __name__ == '__main__':
    unittest.main()
