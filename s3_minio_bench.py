import http.server, os, socketserver, threading, time, urllib.request
import boto3, duckdb
from botocore.client import Config

MINIO_URL = 'http://127.0.0.1:9000'
BUCKET = 'swapo-bench'
GENES = [('COMT', '22', 19941772, 19970744), ('DGCR8', '22', 20080197, 20111641), ('BCR', '22', 23180509, 23318037)]

stats = {'gets': 0, 'heads': 0, 'bytes': 0}
lock = threading.Lock()

class CountingProxy(http.server.BaseHTTPRequestHandler):
    def do_HEAD(self):
        with lock: stats['heads'] += 1
        self._forward('HEAD')
    def do_GET(self):
        with lock: stats['gets'] += 1
        self._forward('GET')
    def _forward(self, method):
        req = urllib.request.Request(MINIO_URL + self.path, method=method)
        for k, v in self.headers.items():
            if k.lower() not in ('connection',): req.add_header(k, v)
        try:
            with urllib.request.urlopen(req) as resp:
                body = resp.read() if method == 'GET' else b''
                if method == 'GET':
                    with lock: stats['bytes'] += len(body)
                self.send_response(resp.status)
                for k, v in resp.headers.items():
                    if k.lower() not in ('transfer-encoding', 'connection'): self.send_header(k, v)
                self.end_headers()
                if body: self.wfile.write(body)
        except urllib.error.HTTPError as e:
            body = e.read()
            self.send_response(e.code)
            for k, v in e.headers.items():
                if k.lower() not in ('transfer-encoding', 'connection'): self.send_header(k, v)
            self.end_headers()
            if body: self.wfile.write(body)
    def log_message(self, *args): pass

class ThreadedServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True

srv = ThreadedServer(('127.0.0.1', 0), CountingProxy)
proxy_port = srv.server_address[1]
threading.Thread(target=srv.serve_forever, daemon=True).start()

s3 = boto3.client('s3', endpoint_url=MINIO_URL, aws_access_key_id='minioadmin', aws_secret_access_key='minioadmin', config=Config(signature_version='s3v4'), region_name='us-east-1')
try: s3.create_bucket(Bucket=BUCKET)
except Exception: pass

for layout in ('sorted', 'swapo'):
    folder = f'chr22_{layout}_pq'
    for root, _, files in os.walk(folder):
        for fn in files:
            full = os.path.join(root, fn)
            rel = os.path.relpath(full, folder).replace('\\', '/')
            s3.upload_file(full, BUCKET, f'{layout}/{rel}')

q, dq = chr(39), chr(34)
print('=== S3 OBJECT-STORE RANGE-GET & BYTE TRANSFER BENCHMARK (MinIO + DuckDB httpfs) ===')
for gene, chrom, qs, qe in GENES:
    row = [f'{gene:6s}']
    for layout in ('sorted', 'swapo'):
        con = duckdb.connect()
        con.execute('INSTALL httpfs; LOAD httpfs;')
        con.execute(f'SET s3_endpoint={q}127.0.0.1:{proxy_port}{q}; SET s3_access_key_id={q}minioadmin{q}; SET s3_secret_access_key={q}minioadmin{q}; SET s3_use_ssl=false; SET s3_url_style={q}path{q};')
        with lock: stats['gets'] = stats['heads'] = stats['bytes'] = 0
        t0 = time.perf_counter()
        sql = f'SELECT count(*) FROM read_parquet({q}s3://{BUCKET}/{layout}/**/*.parquet{q}, hive_partitioning=true) WHERE contig = {q}{chrom}{q} AND {dq}end{dq} >= {qs} AND start <= {qe}'
        hits = con.execute(sql).fetchone()[0]
        ms = (time.perf_counter() - t0) * 1000
        kb = stats['bytes'] / 1024.0
        row.append(f'{layout}: {ms:6.1f} ms | GETs={stats["gets"]:2d} | {kb:7.1f} KB (hits={hits})')
        con.close()
    print(' | '.join(row))
