import unittest
import tempfile
import json
from pathlib import Path
import pandas as pd
from scripts.download_pair_contract_history import quote_jobs, registry, sql_for
from scripts.audit_contract_history import read_contracts, iter_contract_batches, mask_bounds


class ContractDownloadTests(unittest.TestCase):
    def test_registry_preserves_all_prior_pair_ids(self):
        assets,pairs=registry()
        self.assertEqual(len(assets),317)
        self.assertEqual(len(pairs),358)
        self.assertFalse(assets.secid.duplicated().any())
        self.assertFalse(pairs.pair.duplicated().any())
        self.assertTrue(set(pairs.secid_a)|set(pairs.secid_b)<=set(assets.secid))
        self.assertTrue(assets.start.eq('2010-01-01').all())
        self.assertTrue(pairs.start.eq('2010-01-01').all())

    def test_partitions_cover_each_requested_name_year_once(self):
        assets=pd.DataFrame([dict(secid=1,start='2010-01-01'),dict(secid=2,start='2018-01-01')])
        jobs=quote_jobs(assets,1)
        counts={}
        for j in jobs:
            if j['kind']!='quotes':continue
            for sid in j['secids']:
                k=(sid,j['year']);counts[k]=counts.get(k,0)+1
        self.assertEqual(len(counts),24)
        self.assertEqual(set(counts.values()),{1})
        self.assertNotIn((2,2017),counts)
        self.assertIn((2,2018),counts)

    def test_quote_query_retains_bad_quotes_for_later_audit(self):
        query,params=sql_for(dict(kind='quotes',table='optionm.opprcd2025',secids=[1,2],start='2025-01-01',end='2025-09-30'))
        where=query.split('where')[1]
        self.assertNotIn('best_bid',where)
        self.assertNotIn('delta',where)
        self.assertIn('date + 60',where)
        self.assertEqual(params[0],[1,2])

    def test_streaming_reader_filters_and_enforces_memory_limit(self):
        with tempfile.TemporaryDirectory() as temp:
            folder=Path(temp)/'quotes/year=2025';folder.mkdir(parents=True)
            path=folder/'batch=test.parquet'
            pd.DataFrame(dict(secid=[1,1,2],date=pd.to_datetime(['2025-01-02']*3),
                              exdate=pd.to_datetime(['2025-02-07']*3),optionid=[1,2,3])).to_parquet(path,index=False)
            path.with_suffix('.json').write_text(json.dumps(dict(request=dict(secids=[1,2]))))
            batches=list(iter_contract_batches(temp,[1],'2025-01-01','2025-01-31'))
            self.assertEqual(sum(map(len,batches)),2)
            with self.assertRaises(MemoryError):
                read_contracts(temp,[1],'2025-01-01','2025-01-31',max_rows=1)
            # An expanded universe can leave superseded batches on disk. The
            # active request list must prevent double counting their quotes.
            obsolete=folder/'batch=obsolete.parquet'
            obsolete.write_bytes(path.read_bytes())
            obsolete.with_suffix('.json').write_text(path.with_suffix('.json').read_text())
            (Path(temp)/'requests.json').write_text(json.dumps([dict(kind='quotes',year=2025,
                secids=[1,2],path='quotes/year=2025/batch=test.parquet')]))
            self.assertEqual(len(read_contracts(temp,[1],'2025-01-01','2025-01-31')),2)

    def test_compact_calendar_preserves_first_and_last_date(self):
        a,b=mask_bounds((1<<0)|(1<<365))
        self.assertEqual(a,pd.Timestamp('2010-01-01'))
        self.assertEqual(b,pd.Timestamp('2011-01-01'))
        self.assertTrue(pd.isna(mask_bounds(0)[0]))


if __name__=='__main__':unittest.main()
