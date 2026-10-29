"""Offline transfer checks against a local HTTP server; no Kaggle credentials needed."""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import tempfile
import threading
import unittest

import requests
from pull_kernel_output import fetch


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.server.last_range = self.headers.get('Range')
        if self.path == '/range' and self.server.last_range:
            start = int(self.server.last_range.removeprefix('bytes=').removesuffix('-'))
            self.send_response(206)
            self.send_header('Content-Range', f'bytes {start}-7/8')
            self.send_header('Content-Length', str(8 - start))
            self.end_headers()
            self.wfile.write(b'complete'[start:])
            self.close_connection = True
            return
        if self.path == '/bad-range':
            self.send_response(206)
            self.send_header('Content-Range', 'bytes 0-7/8')
            self.send_header('Content-Length', '8')
            self.end_headers()
            self.wfile.write(b'complete')
            self.close_connection = True
            return
        self.send_response(200)
        self.send_header('Content-Length', '8')
        self.end_headers()
        self.wfile.write(b'partial' if self.path == '/short' else b'complete')
        self.close_connection = True

    def log_message(self, *args):
        pass


class TransferTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.url = f'http://127.0.0.1:{cls.server.server_port}'

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join()

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.out = Path(tmp.name) / 'nested' / 'output.bin'
        self.session = requests.Session()
        self.addCleanup(self.session.close)

    def test_download_then_skip_completed_file(self):
        self.assertEqual(fetch(self.session, self.url, self.out, 2), ('ok', 8))
        self.assertEqual(self.out.read_bytes(), b'complete')
        self.assertFalse(self.out.with_suffix('.bin.part').exists())
        self.assertEqual(fetch(self.session, self.url, self.out, 2), ('skip', 0))

    def test_short_transfer_preserves_previous_file(self):
        self.out.parent.mkdir()
        self.out.write_bytes(b'old')
        with self.assertRaises(requests.exceptions.ChunkedEncodingError):
            fetch(self.session, self.url + '/short', self.out, 2)
        self.assertEqual(self.out.read_bytes(), b'old')
        self.assertEqual(fetch(self.session, self.url, self.out, 2), ('ok', 8))
        self.assertEqual(self.out.read_bytes(), b'complete')

    def test_retry_downloads_only_remaining_bytes(self):
        self.out.parent.mkdir()
        self.out.with_suffix('.bin.part').write_bytes(b'com')
        self.assertEqual(fetch(self.session, self.url + '/range', self.out, 2), ('ok', 5))
        self.assertEqual(self.server.last_range, 'bytes=3-')
        self.assertEqual(self.out.read_bytes(), b'complete')

    def test_ignored_range_restarts_without_duplicating_prefix(self):
        self.out.parent.mkdir()
        self.out.with_suffix('.bin.part').write_bytes(b'com')
        self.assertEqual(fetch(self.session, self.url, self.out, 2), ('ok', 8))
        self.assertEqual(self.out.read_bytes(), b'complete')

    def test_wrong_range_preserves_partial_and_previous_file(self):
        self.out.parent.mkdir()
        self.out.write_bytes(b'old')
        part = self.out.with_suffix('.bin.part')
        part.write_bytes(b'com')
        with self.assertRaisesRegex(IOError, 'invalid Content-Range'):
            fetch(self.session, self.url + '/bad-range', self.out, 2)
        self.assertEqual(part.read_bytes(), b'com')
        self.assertEqual(self.out.read_bytes(), b'old')


if __name__ == '__main__':
    unittest.main()


def test_historical_kernel_version_and_all_pages_are_preserved():
    """Use the SDK request schema; never silently fetch latest for owner/slug/7."""
    import sys
    from types import ModuleType,SimpleNamespace
    from unittest.mock import patch
    from pull_kernel_output import listing

    calls=[]
    def output(req):
        calls.append((req.version_label,req.page_token))
        if not req.page_token:
            return SimpleNamespace(files=[SimpleNamespace(file_name='first.txt',url='https://example.invalid/1')],
                                   log='Tiếng Việt',next_page_token='next')
        return SimpleNamespace(files=[SimpleNamespace(file_name='last.txt',url='https://example.invalid/2')],
                               log=None,next_page_token=None)
    backend=SimpleNamespace(kernels=SimpleNamespace(kernels_api_client=SimpleNamespace(list_kernel_session_output=output)))
    class Client:
        def __enter__(self):return backend
        def __exit__(self,*args):pass
    class Api:
        def authenticate(self):pass
        def parse_kernel_string(self,kernel):return kernel.split('/')
        def build_kaggle_client(self):return Client()
    module=ModuleType('kaggle.api.kaggle_api_extended');module.KaggleApi=Api
    with patch.dict(sys.modules,{'kaggle':ModuleType('kaggle'),'kaggle.api':ModuleType('kaggle.api'),
                                'kaggle.api.kaggle_api_extended':module}):
        slug,files,log=listing('owner/slug/7')
    assert slug=='slug' and set(files)=={'first.txt','last.txt'} and log=='Tiếng Việt'
    assert calls==[('v7',''),('v7','next')]
