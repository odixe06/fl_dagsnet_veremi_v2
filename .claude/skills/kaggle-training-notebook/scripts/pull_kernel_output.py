#!/usr/bin/env python
"""Parallel, resumable pull of one kernel's committed output (replacement for `kaggle kernels output`).

    python kaggle_as.py <owner> -- python pull_kernel_output.py <owner>/<slug> -p <dest> [-j 8] [--pattern REGEX]

Kaggle 2.2.5 streams/resumes output downloads and honors explicit kernel versions. Its CLI still handles one
listing page per invocation. This helper keeps automatic all-page traversal, parallel transfers and a final
missing-file exit gate. Use owner/slug/version when collecting an immutable historical run.

Here: list every file once (each entry carries its own signed URL), then a thread pool streams files to `<name>.part`
and renames only when the byte count equals Content-Length. Files already local with the remote size are skipped
(a HEAD-free check: the size comes from the GET headers before any body is read). Failed files are retried with
backoff; after a pass, the listing is refreshed (signed URLs expire) and missing files are retried, up to --passes.
Interrupted `.part` files are retained; later attempts request the remaining bytes with HTTP Range, validate
Content-Range before appending, and safely restart if the server ignores Range.
Exit status is non-zero if any file is still missing; also validate expected artifacts after the pull.
The kernel log is written to <dest>/<slug>.log as the CLI does. Credentials come from the environment (kaggle_as.py).
"""
import argparse, os, re, sys, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import requests


def listing(kernel):
    from kaggle.api.kaggle_api_extended import KaggleApi
    from kagglesdk.kernels.types.kernels_api_service import ApiListKernelSessionOutputRequest
    api = KaggleApi(); api.authenticate(); owner, slug, version = api.parse_kernel_string(kernel)
    files, log, token = {}, None, None
    with api.build_kaggle_client() as k:
        while True:
            req = ApiListKernelSessionOutputRequest(); req.user_name, req.kernel_slug, req.page_size = owner, slug, 200
            if version: req.version_label = f'v{version}' if version.isdigit() else version
            if token: req.page_token = token
            r = k.kernels.kernels_api_client.list_kernel_session_output(req)
            files.update({f.file_name: f.url for f in r.files or []}); log = log or r.log; token = r.next_page_token
            if not token: return slug, files, log


def fetch(sess, url, out, timeout):
    """-> ('skip'|'ok', bytes). Raises on any short or failed transfer."""
    part = out.with_name(out.name + '.part')
    offset = part.stat().st_size if part.is_file() else 0
    r = sess.get(url, stream=True, timeout=(30, timeout), headers={'Range': f'bytes={offset}-'} if offset else {})
    if r.status_code == 416 and offset:  # stale or fully downloaded partial; ask for the whole object
        r.close(); offset = 0
        r = sess.get(url, stream=True, timeout=(30, timeout))
    with r:
        r.raise_for_status()
        if r.status_code == 206:
            span = re.fullmatch(r'bytes (\d+)-(\d+)/(\d+)', r.headers.get('Content-Range', ''))
            if not span or int(span[1]) != offset or int(span[2]) != int(span[3]) - 1:
                raise IOError('invalid Content-Range; partial not appended')
            size = int(span[3])
        else:
            offset = 0  # server ignored Range; overwrite the partial, never append a full response
            size = int(r.headers.get('Content-Length', -1))
        if size >= 0 and out.is_file() and out.stat().st_size == size: return 'skip', 0
        out.parent.mkdir(parents=True, exist_ok=True); n = offset
        with open(part, 'ab' if offset else 'wb') as f:
            for chunk in r.iter_content(1 << 20): f.write(chunk); n += len(chunk)
        if size >= 0 and n != size: raise IOError(f'{out.name}: got {n} of {size} bytes')
        os.replace(part, out); return 'ok', n - offset


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('kernel'); ap.add_argument('-p', '--path', required=True)
    ap.add_argument('-j', '--jobs', type=int, default=8); ap.add_argument('--pattern'); ap.add_argument('--passes', type=int, default=4)
    ap.add_argument('--retries', type=int, default=3); ap.add_argument('--timeout', type=float, default=120, help='read timeout per chunk, s')
    a = ap.parse_args(); dest = Path(a.path); dest.mkdir(parents=True, exist_ok=True); pat = re.compile(a.pattern) if a.pattern else None
    sess = requests.Session(); sess.mount('https://', requests.adapters.HTTPAdapter(pool_connections=a.jobs, pool_maxsize=a.jobs))
    t0, total, done, missing = time.monotonic(), 0, set(), set()
    for p in range(a.passes):
        slug, files, log = listing(a.kernel)
        if log and p == 0: (dest / f'{slug}.log').write_text(log, encoding='utf-8')
        todo = {n: u for n, u in files.items() if (not pat or pat.search(n)) and n not in done}
        if p == 0: print(f'{a.kernel}: {len(todo)} files, {a.jobs} parallel', flush=True)
        def one(item):
            name, url = item
            for k in range(a.retries):
                try: return name, *fetch(sess, url, dest / name, a.timeout)
                except Exception as e: err = e; time.sleep(2 ** k)
            return name, 'fail', type(err).__name__  # Exceptions may contain a signed private-output URL.
        missing = set()
        with ThreadPoolExecutor(a.jobs) as ex:
            for f in as_completed([ex.submit(one, it) for it in todo.items()]):
                name, st, v = f.result()
                if st == 'fail': missing.add(name); print('FAIL', name, v, file=sys.stderr, flush=True)
                else: done.add(name); total += v if st == 'ok' else 0
        if not missing: break
        print(f'pass {p + 1}: {len(missing)} missing, refreshing listing', file=sys.stderr, flush=True)
    dt = time.monotonic() - t0
    print(f'{a.kernel}: {len(done)} files ok, {len(missing)} missing, {total / 2**20:.0f} MiB new in {dt:.0f} s '
          f'({total / 2**20 / max(dt, 1e-9):.1f} MiB/s)', flush=True)
    sys.exit(1 if missing else 0)


if __name__ == '__main__':
    main()
