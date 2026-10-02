"""Read-only WRDS catalog scan: one connection, bounded rows, query timeouts.

Uses the existing libpq credential store and never prints credentials. This
inspects entitlements and schemas; it does not download vendor datasets.
"""
import argparse
import csv
import getpass
import json
import os
from pathlib import Path
import resource
from contextlib import closing
from datetime import datetime, timezone

import psycopg2
from psycopg2 import sql

OUT = Path('reports/wrds_discovery')


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, default=str) + '\n')
    temporary.replace(path)


def rows(db, statement, parameters=None):
    with db.cursor() as cursor:
        cursor.execute(statement, parameters)
        names = [c.name for c in cursor.description]
        return [dict(zip(names, row)) for row in cursor.fetchall()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inspect', nargs='*', default=[])
    parser.add_argument('--probe', nargs='*', default=[])
    parser.add_argument('--bounds', nargs='*', default=[])
    parser.add_argument('--cached-catalog', action='store_true')
    args = parser.parse_args()
    os.nice(15)
    OUT.mkdir(parents=True, exist_ok=True)
    with closing(psycopg2.connect(host='wrds-pgdata.wharton.upenn.edu', port=9737,
                         dbname='wrds', user=os.getenv('WRDS_USERNAME', getpass.getuser()),
                         sslmode='require', connect_timeout=20,
                         application_name='bounded_research_catalog_scan')) as db:
        db.set_session(readonly=True, autocommit=True)
        with db.cursor() as cursor:
            cursor.execute("SET statement_timeout = '8s'")
            cursor.execute("SET lock_timeout = '2s'")
        safety=rows(db,"SELECT current_setting('transaction_read_only') AS readonly, current_setting('statement_timeout') AS statement_timeout")[0]
        assert safety['readonly']=='on';print(json.dumps(dict(safety=safety)),flush=True)
        catalog = (json.loads((OUT/'catalog.json').read_text())['catalog']
                   if args.cached_catalog else rows(db, """
            SELECT n.nspname AS schema, c.relname AS table, c.relkind AS kind,
                   has_schema_privilege(n.oid, 'USAGE') AS schema_usage,
                   has_table_privilege(c.oid, 'SELECT') AS select_permission,
                   c.reltuples AS estimated_rows,
                   obj_description(c.oid, 'pg_class') AS description
              FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace
             WHERE c.relkind IN ('r', 'v', 'm', 'f', 'p')
               AND n.nspname NOT IN ('pg_catalog', 'information_schema')
               AND n.nspname NOT LIKE 'pg_toast%'
             ORDER BY n.nspname, c.relname
        """))
        if not args.cached_catalog:
            write_json(OUT/'catalog.json', dict(scanned_at=datetime.now(timezone.utc).isoformat(),
                       catalog=catalog, method='PostgreSQL metadata and role SELECT/USAGE checks; data reads separately probed'))
        schemas = {}
        for r in catalog:
            s = schemas.setdefault(r['schema'], dict(tables=0, select_tables=0, usage=r['schema_usage']))
            s['tables'] += 1
            s['select_tables'] += int(r['select_permission'] and r['schema_usage'])
        with (OUT/'schemas.csv').open('w') as stream:
            writer = csv.DictWriter(stream, fieldnames=['schema', 'tables', 'select_tables', 'usage'])
            writer.writeheader()
            for name, values in schemas.items(): writer.writerow(dict(schema=name, **values))
        print(json.dumps(dict(schemas=len(schemas), tables=len(catalog),
                     schemas_with_select=sum(s['select_tables']>0 for s in schemas.values()))), flush=True)
        if args.inspect:
            wanted = set(args.inspect)
            known = {r['schema'] for r in catalog}
            assert wanted <= known, f'Unknown schemas: {sorted(wanted-known)}'
            columns = rows(db, """
                SELECT n.nspname AS schema, c.relname AS table,
                       a.attnum AS ordinal, a.attname AS column,
                       format_type(a.atttypid,a.atttypmod) AS type,
                       col_description(c.oid,a.attnum) AS description
                  FROM pg_attribute a JOIN pg_class c ON c.oid=a.attrelid
                  JOIN pg_namespace n ON n.oid=c.relnamespace
                 WHERE n.nspname=ANY(%s) AND a.attnum>0 AND NOT a.attisdropped
                   AND c.relkind IN ('r','v','m','f','p')
                 ORDER BY n.nspname,c.relname,a.attnum
            """, (sorted(wanted),))
            previous=json.loads((OUT/'columns.json').read_text()) if (OUT/'columns.json').exists() else []
            write_json(OUT/'columns.json', [r for r in previous if r['schema'] not in wanted]+columns)
            print(json.dumps(dict(inspected_schemas=sorted(wanted), columns=len(columns))), flush=True)
        known={(r['schema'],r['table']) for r in catalog}
        if args.probe:
            assert len(args.probe)<=64
            results=json.loads((OUT/'probes.json').read_text()) if (OUT/'probes.json').exists() else {}
            for name in args.probe:
                schema,table=name.split('.');assert (schema,table) in known,name
                try:
                    got=rows(db,sql.SQL('SELECT 1 AS present FROM {} LIMIT 1').format(sql.Identifier(schema,table)))
                    result=dict(status='readable',nonempty=bool(got))
                except psycopg2.Error as exc:
                    result=dict(status='timeout' if exc.pgcode=='57014' else 'error',sqlstate=exc.pgcode,
                                message=exc.diag.message_primary)
                results[name]=result;write_json(OUT/'probes.json',results)
                print(json.dumps(dict(table=name,**result)),flush=True)
        if args.bounds:
            assert len(args.bounds)<=32
            results=json.loads((OUT/'date_bounds.json').read_text()) if (OUT/'date_bounds.json').exists() else {}
            for name in args.bounds:
                schema,table,column=name.split('.');assert (schema,table) in known,name
                result={}
                try:
                    for direction,label in [('ASC','first'),('DESC','last')]:
                        query=sql.SQL('SELECT {} AS value FROM {} WHERE {} IS NOT NULL ORDER BY {} {} LIMIT 1').format(
                            sql.Identifier(column),sql.Identifier(schema,table),sql.Identifier(column),
                            sql.Identifier(column),sql.SQL(direction))
                        plan=rows(db,sql.SQL('EXPLAIN (FORMAT JSON) ')+query)[0]['QUERY PLAN'][0]['Plan']
                        def nodes(p):
                            yield p
                            for child in p.get('Plans',[]):yield from nodes(child)
                        heavy=[p for p in nodes(plan) if p['Node Type'] in ['Seq Scan','Parallel Seq Scan','Foreign Scan'] and p.get('Plan Rows',0)>100000]
                        if heavy:
                            result[label]=dict(status='skipped_large_scan',estimated_rows=max(p['Plan Rows'] for p in heavy))
                        else:
                            values=rows(db,query);result[label]=dict(status='observed',value=values[0]['value'] if values else None)
                    if any(r.get('status')=='skipped_large_scan' for r in result.values()):
                        statistics=rows(db,"SELECT histogram_bounds::text AS bounds FROM pg_stats WHERE schemaname=%s AND tablename=%s AND attname=%s",(schema,table,column))
                        if statistics and statistics[0]['bounds']:
                            bins=next(csv.reader([statistics[0]['bounds'].strip('{}')]))
                            result['metadata_statistics']=dict(status='approximate_analyze_statistics',first=bins[0],last=bins[-1])
                except psycopg2.Error as exc:
                    result['error']=dict(sqlstate=exc.pgcode,message=exc.diag.message_primary)
                results[name]=result;write_json(OUT/'date_bounds.json',results)
                print(json.dumps(dict(field=name,**result),default=str),flush=True)
        print(json.dumps(dict(peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)), flush=True)


if __name__ == '__main__':
    main()
