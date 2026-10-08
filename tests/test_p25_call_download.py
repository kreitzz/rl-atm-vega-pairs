"""Check that the targeted WRDS request cannot silently broaden its scope."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

from scripts import download_p25_call_transfer as target


def raw_quote(date):
    return dict(secid=123, date=date, optionid=987, exdate=pd.Timestamp('2022-01-07'),
                cp_flag='C', strike_price=50000., best_bid=2., best_offer=2.2,
                open_interest=150., impl_volatility=.25, delta=.75, vega=12.,
                ss_flag='0', cfadj=1., contract_size=100.)


def workspace_temp():
    parent = (target.ROOT / 'data').resolve()
    folder = tempfile.TemporaryDirectory(dir=parent, prefix='p25_call_test_')
    # Verify the cleanup target remains inside the intended workspace folder.
    Path(folder.name).resolve().relative_to(parent)
    return folder


class TargetedDownloadTests(unittest.TestCase):
    def test_union_dates_and_no_settlement_or_future_quotes(self):
        dates = pd.bdate_range('2022-01-03', '2022-01-10')
        mapping = pd.DataFrame([
            dict(date=dates[0], end_date=dates[3], optionid=1),
            dict(date=dates[1], end_date=dates[4], optionid=1)])
        result = target.quote_keys(mapping, dates)
        self.assertEqual(result.date.tolist(), list(dates[:4]))
        self.assertEqual(len(result), 4)
        with patch.object(target, 'MAX_QUOTE_KEYS', 3):
            with self.assertRaises(RuntimeError):
                target.quote_keys(mapping, dates)

    def test_missing_contract_lookup_then_only_missing_daily_keys(self):
        dates = pd.bdate_range('2022-01-03', '2022-01-10')
        refs = pd.DataFrame([dict(cohort='test', date=dates[0], end_date=dates[3],
                                  secid=123, strike=50., expiry=pd.Timestamp('2022-01-07'))])
        mapping = refs.assign(optionid=float('nan'))
        local = target.normalize(pd.DataFrame([raw_quote(dates[0])])).iloc[:0]
        queries = []

        class FakeDownloader:
            def __init__(self, out):
                pass

            def query(self, label, query, params, maximum):
                queries.append((label, query, params, maximum))
                if label.startswith('lookup'):
                    return pd.DataFrame([raw_quote(dates[0])])
                return pd.DataFrame([raw_quote(d) for d in params['dates']])

            def close(self):
                pass

        with workspace_temp() as tmp, patch.object(target, 'Downloader', FakeDownloader):
            out = Path(tmp)
            target.write_json(out / 'download_plan.json', {})
            target.download(out, refs, mapping, local, dates, {})
            quotes = pd.read_parquet(out / 'call_quotes.parquet')
            self.assertEqual(quotes.date.tolist(), list(dates[:3]))
            self.assertTrue(quotes.strike.eq(50).all())
            self.assertTrue(quotes.vendor_vega.eq(.12).all())
            status = json.loads((out / 'download_status.json').read_text())
            self.assertEqual(status['unmatched_legs'], 0)
            self.assertEqual(status['missing_daily_quotes'], 0)
        self.assertEqual(len(queries), 2)
        self.assertEqual(queries[0][2]['strikes'], [50000])
        self.assertEqual(queries[1][2]['dates'], dates[1:3].date.tolist())
        self.assertEqual(queries[1][2]['ids'], [987, 987])
        for _, sql, _, _ in queries:
            self.assertIn("q.cp_flag='C'", sql)
            self.assertIn('JOIN unnest', sql)
            self.assertIn('LIMIT', sql)

    def test_query_checkpoints_resume_without_connection_and_detect_tampering(self):
        class FakeDB:
            def raw_sql(self, *args, **kwargs):
                return pd.DataFrame([raw_quote(pd.Timestamp('2022-01-03'))])

        with workspace_temp() as tmp:
            downloader = target.Downloader(Path(tmp))
            downloader.db = FakeDB()
            first = downloader.query('test', 'bounded query', {'ids': [987]}, 1)
            downloader.db = None
            second = downloader.query('test', 'bounded query', {'ids': [987]}, 1)
            pd.testing.assert_frame_equal(first, second)
            cache = next((Path(tmp) / 'query_cache').glob('*.parquet'))
            cache.write_bytes(b'changed')
            with self.assertRaises(RuntimeError):
                downloader.query('test', 'bounded query', {'ids': [987]}, 1)


if __name__ == '__main__':
    unittest.main()
