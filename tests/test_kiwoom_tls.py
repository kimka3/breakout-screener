"""Offline regressions for the narrowly scoped KIWOOM TLS chain repair."""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import ssl
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import certifi

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import kiwoom_tls


URL = "https://www.kiwoometf.com/service/etf/KO02010200M?gcode=316670"
MISSING = "SSL certificate problem: unable to get local issuer certificate"


class CurlFailure(Exception):
    def __init__(self, message=MISSING, code=60):
        super().__init__(message)
        self.code = code


class KiwoomTlsTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="kiwoom-tls-test-")
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.baseline = self.root / "baseline.pem"
        self.baseline.write_bytes(Path(certifi.where()).read_bytes())
        self.environment = patch.dict(os.environ, {}, clear=True)
        self.environment.start()
        self.addCleanup(self.environment.stop)
        self.ca = patch.object(kiwoom_tls.certifi, "where", return_value=str(self.baseline))
        self.ca.start()
        self.addCleanup(self.ca.stop)
        self.openssl = patch.object(kiwoom_tls.subprocess, "run",
                                    return_value=SimpleNamespace(returncode=0))
        self.run = self.openssl.start()
        self.addCleanup(self.openssl.stop)

    def test_public_certificate_has_the_pinned_der_fingerprint(self):
        der = ssl.PEM_cert_to_DER_cert(kiwoom_tls.KIWOOM_ISSUER_PEM)
        self.assertEqual(hashlib.sha256(der).hexdigest(),
                         "72a34ac2b424aed3f6b0b04755b88cc027dccc806fddb22b4cd7c47773973ec0")

    def test_successful_first_request_is_unchanged_and_does_not_prepare_a_bundle(self):
        response = object()
        request = Mock(return_value=response)
        options = {"timeout": 17, "headers": {"Referer": URL}, "verify": True}
        self.assertIs(kiwoom_tls.request_with_kiwoom_chain(request, URL, **options), response)
        request.assert_called_once_with(URL, **options)
        self.run.assert_not_called()

    def test_missing_issuer_retries_once_preserving_options_and_cleans_up_bundle(self):
        response = object()
        options = {"timeout": 17, "headers": {"Referer": URL},
                   "data": {"schGubun1": "316670"}, "impersonate": "chrome"}
        observed = []

        def request(url, **kwargs):
            observed.append((url, kwargs))
            if len(observed) == 1:
                raise CurlFailure()
            bundle = Path(kwargs["verify"])
            self.assertEqual(bundle.read_bytes(), self.baseline.read_bytes() + b"\n"
                             + kiwoom_tls.KIWOOM_ISSUER_PEM.encode("ascii"))
            self.assertTrue(bundle.is_file())
            self.assertEqual({key: value for key, value in kwargs.items() if key != "verify"}, options)
            return response

        environment_before = dict(os.environ)
        self.assertIs(kiwoom_tls.request_with_kiwoom_chain(request, URL, **options), response)
        self.assertEqual(len(observed), 2)
        self.assertEqual(observed[0], (URL, options))
        self.assertEqual(dict(os.environ), environment_before)
        bundle = Path(observed[1][1]["verify"])
        self.assertFalse(bundle.exists())
        self.assertFalse(bundle.parent.exists())
        command = self.run.call_args.args[0]
        self.assertEqual(command[:4], ["openssl", "verify", "-CAfile", str(self.baseline)])
        self.assertEqual(command[4], str(bundle.parent / "issuer.pem"))
        self.run.assert_called_once_with(command, capture_output=True, timeout=5)

    def test_both_precise_missing_issuer_errors_and_default_https_port_are_allowed(self):
        for detail in (MISSING, "SSL certificate problem: unable to verify the first certificate"):
            for url in (URL, "https://www.kiwoometf.com:443/service/etf"):
                with self.subTest(detail=detail, url=url):
                    response = object()
                    request = Mock(side_effect=[CurlFailure(detail), response])
                    self.assertIs(kiwoom_tls.request_with_kiwoom_chain(request, url), response)
                    self.assertEqual(request.call_count, 2)

    def test_other_origins_and_invalid_ports_never_retry(self):
        for url in ("http://www.kiwoometf.com/", "https://kiwoometf.com/",
                    "https://www.kiwoometf.com:444/", "https://other.example/",
                    "https://www.kiwoometf.com.evil.example/",
                    "https://www.kiwoometf.com@other.example/",
                    "https://user@www.kiwoometf.com/", "https://www.kiwoometf.com:bad/"):
            with self.subTest(url=url):
                error = CurlFailure()
                request = Mock(side_effect=error)
                with self.assertRaises(CurlFailure) as raised:
                    kiwoom_tls.request_with_kiwoom_chain(request, url)
                self.assertIs(raised.exception, error)
                request.assert_called_once_with(url)
        self.run.assert_not_called()

    def test_wrong_error_codes_or_unrelated_exceptions_never_retry(self):
        for error in (CurlFailure(MISSING, 35), CurlFailure(MISSING, "60"),
                      RuntimeError(MISSING), CurlFailure("TLS handshake failed")):
            with self.subTest(error=repr(error)):
                request = Mock(side_effect=error)
                with self.assertRaises(type(error)) as raised:
                    kiwoom_tls.request_with_kiwoom_chain(request, URL)
                self.assertIs(raised.exception, error)
                request.assert_called_once_with(URL)
        self.run.assert_not_called()

    def test_expiry_hostname_self_signed_and_signature_errors_never_retry(self):
        for detail in ("certificate has expired", "certificate is not yet valid",
                       "hostname mismatch", "no alternative certificate subject name matches",
                       "self-signed certificate in certificate chain", "self signed certificate",
                       "IP address mismatch", "certificate signature failure"):
            for message in (detail, MISSING + "; " + detail):
                with self.subTest(message=message):
                    error = CurlFailure(message)
                    request = Mock(side_effect=error)
                    with self.assertRaises(CurlFailure) as raised:
                        kiwoom_tls.request_with_kiwoom_chain(request, URL)
                    self.assertIs(raised.exception, error)
                    request.assert_called_once_with(URL)
        self.run.assert_not_called()

    def test_retry_failure_propagates_without_a_third_request(self):
        error = CurlFailure()
        request = Mock(side_effect=[CurlFailure(), error])
        with self.assertRaises(CurlFailure) as raised:
            kiwoom_tls.request_with_kiwoom_chain(request, URL)
        self.assertIs(raised.exception, error)
        self.assertEqual(request.call_count, 2)
        self.assertFalse(Path(request.call_args.kwargs["verify"]).exists())

    def test_wrong_fingerprint_fails_closed_before_openssl_or_retry(self):
        request = Mock(side_effect=CurlFailure())
        with patch.object(kiwoom_tls, "KIWOOM_ISSUER_SHA256", "0" * 64):
            with self.assertRaisesRegex(ValueError, "fingerprint"):
                kiwoom_tls.request_with_kiwoom_chain(request, URL)
        request.assert_called_once_with(URL)
        self.run.assert_not_called()

    def test_untrusted_or_expired_issuer_fails_closed_before_retry(self):
        request = Mock(side_effect=CurlFailure())
        self.run.return_value.returncode = 2
        with self.assertRaisesRegex(ValueError, "trust or validity"):
            kiwoom_tls.request_with_kiwoom_chain(request, URL)
        request.assert_called_once_with(URL)
        issuer = Path(self.run.call_args.args[0][4])
        self.assertFalse(issuer.parent.exists())

    def test_missing_or_timed_out_openssl_fails_closed_before_retry(self):
        for error in (FileNotFoundError("openssl"), subprocess.TimeoutExpired("openssl", 5)):
            with self.subTest(error=error):
                request = Mock(side_effect=CurlFailure())
                self.run.side_effect = error
                with self.assertRaises(type(error)):
                    kiwoom_tls.request_with_kiwoom_chain(request, URL)
                request.assert_called_once_with(URL)
                self.assertFalse(Path(self.run.call_args.args[0][4]).parent.exists())

    def test_explicit_ca_path_takes_precedence_over_all_ca_environment_variables(self):
        explicit = self.root / "explicit.pem"
        explicit.write_bytes(self.baseline.read_bytes())
        with patch.dict(os.environ, {key: "unused" for key in
                                    ("CURL_CA_BUNDLE", "REQUESTS_CA_BUNDLE", "SSL_CERT_FILE")}):
            request = Mock(side_effect=[CurlFailure(), object()])
            kiwoom_tls.request_with_kiwoom_chain(request, URL, verify=str(explicit))
        self.assertEqual(self.run.call_args.args[0][3], str(explicit))
        self.assertEqual(request.call_args_list[0].kwargs["verify"], str(explicit))

    def test_ca_environment_variable_precedence_and_certifi_default(self):
        keys = ("CURL_CA_BUNDLE", "REQUESTS_CA_BUNDLE", "SSL_CERT_FILE")
        for index in range(len(keys) + 1):
            paths = {}
            for key in keys[index:]:
                ca = self.root / (key + ".pem")
                ca.write_bytes(self.baseline.read_bytes())
                paths[key] = str(ca)
            with self.subTest(present=list(paths)), patch.dict(os.environ, paths, clear=True):
                request = Mock(side_effect=[CurlFailure(), object()])
                kiwoom_tls.request_with_kiwoom_chain(request, URL, verify=True)
                expected = paths[keys[index]] if paths else str(self.baseline)
                self.assertEqual(self.run.call_args.args[0][3], expected)

    def test_disabled_verification_is_never_used_for_a_repair(self):
        request = Mock(side_effect=CurlFailure())
        with self.assertRaisesRegex(ValueError, "requires TLS verification"):
            kiwoom_tls.request_with_kiwoom_chain(request, URL, verify=False)
        request.assert_called_once_with(URL, verify=False)
        self.run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
