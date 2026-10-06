"""Small public daily FX input required for foreign-currency cash dividends."""
import json
from pathlib import Path
import urllib.request

URL='https://fred.stlouisfed.org/graph/fredgraph.csv?id=DEXCAUS&cosd=2009-12-01&coed=2025-08-29'
PATH=Path('data/raw/opposite_wing_DEXCAUS.csv')

def main():
    if PATH.exists():print('Historical CAD/USD file already cached');return
    request=urllib.request.Request(URL,headers={'User-Agent':'Mozilla/5.0'})
    with urllib.request.urlopen(request,timeout=30) as response:raw=response.read(1_000_001)
    assert len(raw)<1_000_000 and b'DEXCAUS' in raw[:200]
    PATH.write_bytes(raw)
    PATH.with_suffix('.source.json').write_text(json.dumps(dict(url=URL,
        source='Federal Reserve H.10 via FRED: CAD per USD daily noon market-rate proxy',
        purpose='Foreign dividend accounting, not an alpha feature. Prior observation for each quote/ex-date; static declaration-rate conversion in valuation, identical USD amount for stock cash accounting. Not a vintage-certified daily feed.'),indent=2))
    print(f'Cached {len(raw)} bytes of CAD/USD history')

if __name__=='__main__':main()
