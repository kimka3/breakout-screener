"""Repair only KIWOOM's missing public intermediate, without weakening TLS."""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import ssl
import subprocess
import tempfile
from urllib.parse import urlsplit

import certifi


# Public issuer from KIWOOM's authenticated leaf certificate's issuer URL:
# http://crt.sectigo.com/SectigoRSAOrganizationValidationSecureServerCA.crt
# Copied from RIM Lab's pinned public certificate; this is not a private key.
KIWOOM_ISSUER_PEM = """-----BEGIN CERTIFICATE-----
MIIGGTCCBAGgAwIBAgIQE31TnKp8MamkM3AZaIR6jTANBgkqhkiG9w0BAQwFADCB
iDELMAkGA1UEBhMCVVMxEzARBgNVBAgTCk5ldyBKZXJzZXkxFDASBgNVBAcTC0pl
cnNleSBDaXR5MR4wHAYDVQQKExVUaGUgVVNFUlRSVVNUIE5ldHdvcmsxLjAsBgNV
BAMTJVVTRVJUcnVzdCBSU0EgQ2VydGlmaWNhdGlvbiBBdXRob3JpdHkwHhcNMTgx
MTAyMDAwMDAwWhcNMzAxMjMxMjM1OTU5WjCBlTELMAkGA1UEBhMCR0IxGzAZBgNV
BAgTEkdyZWF0ZXIgTWFuY2hlc3RlcjEQMA4GA1UEBxMHU2FsZm9yZDEYMBYGA1UE
ChMPU2VjdGlnbyBMaW1pdGVkMT0wOwYDVQQDEzRTZWN0aWdvIFJTQSBPcmdhbml6
YXRpb24gVmFsaWRhdGlvbiBTZWN1cmUgU2VydmVyIENBMIIBIjANBgkqhkiG9w0B
AQEFAAOCAQ8AMIIBCgKCAQEAnJMCRkVKUkiS/FeN+S3qU76zLNXYqKXsW2kDwB0Q
9lkz3v4HSKjojHpnSvH1jcM3ZtAykffEnQRgxLVK4oOLp64m1F06XvjRFnG7ir1x
on3IzqJgJLBSoDpFUd54k2xiYPHkVpy3O/c8Vdjf1XoxfDV/ElFw4Sy+BKzL+k/h
fGVqwECn2XylY4QZ4ffK76q06Fha2ZnjJt+OErK43DOyNtoUHZZYQkBuCyKFHFEi
rsTIBkVtkuZntxkj5Ng2a4XQf8dS48+wdQHgibSov4o2TqPgbOuEQc6lL0giE5dQ
YkUeCaXMn2xXcEAG2yDoG9bzk4unMp63RBUJ16/9fAEc2wIDAQABo4IBbjCCAWow
HwYDVR0jBBgwFoAUU3m/WqorSs9UgOHYm8Cd8rIDZsswHQYDVR0OBBYEFBfZ1iUn
Z/kxwklD2TA2RIxsqU/rMA4GA1UdDwEB/wQEAwIBhjASBgNVHRMBAf8ECDAGAQH/
AgEAMB0GA1UdJQQWMBQGCCsGAQUFBwMBBggrBgEFBQcDAjAbBgNVHSAEFDASMAYG
BFUdIAAwCAYGZ4EMAQICMFAGA1UdHwRJMEcwRaBDoEGGP2h0dHA6Ly9jcmwudXNl
cnRydXN0LmNvbS9VU0VSVHJ1c3RSU0FDZXJ0aWZpY2F0aW9uQXV0aG9yaXR5LmNy
bDB2BggrBgEFBQcBAQRqMGgwPwYIKwYBBQUHMAKGM2h0dHA6Ly9jcnQudXNlcnRy
dXN0LmNvbS9VU0VSVHJ1c3RSU0FBZGRUcnVzdENBLmNydDAlBggrBgEFBQcwAYYZ
aHR0cDovL29jc3AudXNlcnRydXN0LmNvbTANBgkqhkiG9w0BAQwFAAOCAgEAThNA
lsnD5m5bwOO69Bfhrgkfyb/LDCUW8nNTs3Yat6tIBtbNAHwgRUNFbBZaGxNh10m6
pAKkrOjOzi3JKnSj3N6uq9BoNviRrzwB93fVC8+Xq+uH5xWo+jBaYXEgscBDxLmP
bYox6xU2JPti1Qucj+lmveZhUZeTth2HvbC1bP6mESkGYTQxMD0gJ3NR0N6Fg9N3
OSBGltqnxloWJ4Wyz04PToxcvr44APhL+XJ71PJ616IphdAEutNCLFGIUi7RPSRn
R+xVzBv0yjTqJsHe3cQhifa6ezIejpZehEU4z4CqN2mLYBd0FUiRnG3wTqN3yhsc
SPr5z0noX0+FCuKPkBurcEya67emP7SsXaRfz+bYipaQ908mgWB2XQ8kd5GzKjGf
FlqyXYwcKapInI5v03hAcNt37N3j0VcFcC3mSZiIBYRiBXBWdoY5TtMibx3+bfEO
s2LEPMvAhblhHrrhFYBZlAyuBbuMf1a+HNJav5fyakywxnB2sJCNwQs2uRHY1ihc
6k/+JLcYCpsM0MF8XPtpvcyiTcaQvKZN8rG61ppnW5YCUtCC+cQKXA0o4D/I+pWV
idWkvklsQLI+qGu41SWyxP7x09fn1txDAXYw+zuLXfdKiXyaNb78yvBXAfCNP6CH
MntHWpdLgtJmwsQt6j8k9Kf5qLnjatkYYaA7jBU=
-----END CERTIFICATE-----
"""
KIWOOM_ISSUER_SHA256 = "72a34ac2b424aed3f6b0b04755b88cc027dccc806fddb22b4cd7c47773973ec0"
_MISSING_ISSUER_ERRORS = (
    "unable to get local issuer certificate", "unable to verify the first certificate",
)
_OTHER_CERTIFICATE_ERRORS = (
    "expired", "not yet valid", "hostname", "host name", "subject name",
    "self-signed", "self signed", "ip address mismatch", "signature failure",
)


def _can_repair(url: str, error: Exception) -> bool:
    try:
        target = urlsplit(url)
        allowed = (target.scheme == "https" and target.hostname == "www.kiwoometf.com"
                   and target.port in (None, 443)
                   and target.username is None and target.password is None)
    except ValueError:
        return False
    detail = str(error).lower()
    return (allowed and getattr(error, "code", None) == 60
            and any(text in detail for text in _MISSING_ISSUER_ERRORS)
            and not any(text in detail for text in _OTHER_CERTIFICATE_ERRORS))


def _baseline_ca(verify) -> str:
    if isinstance(verify, (str, os.PathLike)):
        return os.fspath(verify)
    if verify is False:
        raise ValueError("KIWOOM chain repair requires TLS verification")
    return next((os.environ[key] for key in
                 ("CURL_CA_BUNDLE", "REQUESTS_CA_BUNDLE", "SSL_CERT_FILE")
                 if os.environ.get(key)), certifi.where())


def request_with_kiwoom_chain(request, url: str, **kwargs):
    """Try normally; retry one missing-issuer error with a verified pinned issuer.

    OpenSSL verifies the issuer's signature, trust chain and current validity
    against the caller's CA file before it can enter a temporary request-local
    bundle. The retried request still verifies the server chain and hostname.
    """
    try:
        return request(url, **kwargs)
    except Exception as error:
        if not _can_repair(url, error):
            raise

    der = ssl.PEM_cert_to_DER_cert(KIWOOM_ISSUER_PEM)
    if hashlib.sha256(der).hexdigest() != KIWOOM_ISSUER_SHA256:
        raise ValueError("KIWOOM public intermediate fingerprint does not match")
    baseline = _baseline_ca(kwargs.get("verify"))
    with tempfile.TemporaryDirectory(prefix="kiwoom-ca-") as directory:
        issuer = Path(directory) / "issuer.pem"
        issuer.write_text(KIWOOM_ISSUER_PEM, encoding="ascii")
        result = subprocess.run(
            ["openssl", "verify", "-CAfile", baseline, str(issuer)],
            capture_output=True, timeout=5,
        )
        if result.returncode != 0:
            raise ValueError("KIWOOM public intermediate trust or validity check failed")
        bundle = Path(directory) / "ca-bundle.pem"
        bundle.write_bytes(Path(baseline).read_bytes() + b"\n" + KIWOOM_ISSUER_PEM.encode("ascii"))
        return request(url, **{**kwargs, "verify": str(bundle)})
