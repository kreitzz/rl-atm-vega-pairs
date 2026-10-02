from functools import lru_cache
from pathlib import Path
import pandas as pd
from scripts.run_broad_contract_replay import Market
from scripts.option_flex.policy import COMBOS


class FlexMarket(Market):
    @lru_cache(maxsize=4)
    def puts(self,sid):
        root=Path('data/call_put/history')
        q=[root/str(y)/f'{sid}.parquet' for y in sorted(set(self.calendar.year))]
        s=[root/str(y)/f'{sid}_selection.parquet' for y in sorted(set(self.calendar.year))]
        q=[pd.read_parquet(p) for p in q if p.exists()];s=[pd.read_parquet(p) for p in s if p.exists()]
        return (pd.concat(q,ignore_index=True) if q else pd.DataFrame(),pd.concat(s,ignore_index=True) if s else pd.DataFrame(columns=['date','secid','optionid']))

    def flex_pair(self,secids):
        q,s,call,div=self.pair(secids)
        puts=[self.puts(sid) for sid in secids]
        frames=[q]+[p[0] for p in puts if len(p[0])];q=pd.concat(frames,ignore_index=True)
        bytype={sid:{'C':call[call.secid.eq(sid)],'P':p[1]} for sid,p in zip(secids,puts)}
        selections={c:pd.concat([bytype[sid][cp] for sid,cp in zip(secids,c)],ignore_index=True) for c in COMBOS}
        return q,s,selections,div
