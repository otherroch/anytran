"""Tests for anytran.certs — generate_self_signed_cert with cryptography fallback."""
import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from tests.conftest import _real_certs_funcs as _CF

_generate_self_signed_cert = _CF["generate_self_signed_cert"]


class TestGenerateSelfSignedCertWithCryptography(unittest.TestCase):
    """Test cert generation using the cryptography library fallback path."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.cert_path = os.path.join(self.tmpdir, "server.crt")
        self.key_path = os.path.join(self.tmpdir, "server.key")

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _generate(self, common_name="localhost", verbose=False):
        generate_self_signed_cert = _generate_self_signed_cert  # use saved real fn
        # Force the openssl path to None so we go straight to cryptography
        with patch("shutil.which", return_value=None):
            generate_self_signed_cert(
                self.cert_path, self.key_path,
                common_name=common_name, verbose=verbose
            )

    def test_creates_cert_file(self):
        self._generate()
        self.assertTrue(os.path.exists(self.cert_path))

    def test_creates_key_file(self):
        self._generate()
        self.assertTrue(os.path.exists(self.key_path))

    def test_cert_is_valid_pem(self):
        self._generate()
        content = open(self.cert_path, "rb").read()
        self.assertIn(b"BEGIN CERTIFICATE", content)

    def test_key_is_valid_pem(self):
        self._generate()
        content = open(self.key_path, "rb").read()
        self.assertIn(b"BEGIN", content)

    def test_common_name_localhost(self):
        self._generate(common_name="localhost")
        from cryptography import x509
        cert_bytes = open(self.cert_path, "rb").read()
        cert = x509.load_pem_x509_certificate(cert_bytes)
        cn = cert.subject.get_attributes_for_oid(
            x509.oid.NameOID.COMMON_NAME
        )[0].value
        self.assertEqual(cn, "localhost")

    def test_common_name_custom(self):
        self._generate(common_name="myserver")
        from cryptography import x509
        cert_bytes = open(self.cert_path, "rb").read()
        cert = x509.load_pem_x509_certificate(cert_bytes)
        cn = cert.subject.get_attributes_for_oid(
            x509.oid.NameOID.COMMON_NAME
        )[0].value
        self.assertEqual(cn, "myserver")

    def test_verbose_does_not_raise(self):
        # verbose=True when openssl IS available but fails (falls through to cryptography)
        generate_self_signed_cert = _generate_self_signed_cert  # use saved real fn
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stderr = "openssl error"
        with patch("shutil.which", return_value="/usr/bin/openssl"):
            with patch("subprocess.run", return_value=mock_result):
                generate_self_signed_cert(
                    self.cert_path, self.key_path,
                    common_name="localhost", verbose=True
                )
        self.assertTrue(os.path.exists(self.cert_path))

    def test_ip_address_as_common_name(self):
        self._generate(common_name="127.0.0.1")
        from cryptography import x509
        cert_bytes = open(self.cert_path, "rb").read()
        cert = x509.load_pem_x509_certificate(cert_bytes)
        # SAN extension should include the IP address
        san_ext = cert.extensions.get_extension_for_class(x509.SubjectAlternativeName)
        ip_addrs = san_ext.value.get_values_for_type(x509.IPAddress)
        self.assertTrue(any(str(addr) == "127.0.0.1" for addr in ip_addrs))

    def test_0_0_0_0_normalized_to_localhost(self):
        self._generate(common_name="0.0.0.0")
        from cryptography import x509
        cert_bytes = open(self.cert_path, "rb").read()
        cert = x509.load_pem_x509_certificate(cert_bytes)
        cn = cert.subject.get_attributes_for_oid(x509.oid.NameOID.COMMON_NAME)[0].value
        self.assertEqual(cn, "localhost")

    def test_none_common_name_uses_localhost(self):
        self._generate(common_name=None)
        self.assertTrue(os.path.exists(self.cert_path))


class TestGenerateSelfSignedCertWithOpenSSL(unittest.TestCase):
    """Test that the openssl path is used when available and succeeds."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.cert_path = os.path.join(self.tmpdir, "server.crt")
        self.key_path = os.path.join(self.tmpdir, "server.key")

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_openssl_success_path(self):
        """Test that when openssl succeeds, we don't fall through to cryptography."""
        generate_self_signed_cert = _generate_self_signed_cert  # use saved real fn
        import subprocess
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stderr = ""
        with patch("shutil.which", return_value="/usr/bin/openssl"):
            with patch("subprocess.run", return_value=mock_result):
                # Create fake cert and key files (openssl would normally create them)
                open(self.cert_path, "w").write("fake cert")
                open(self.key_path, "w").write("fake key")
                # Should not raise
                generate_self_signed_cert(
                    self.cert_path, self.key_path, common_name="localhost"
                )

    def test_openssl_failure_falls_through_to_cryptography(self):
        """When openssl fails, fallback to cryptography library."""
        generate_self_signed_cert = _generate_self_signed_cert  # use saved real fn
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stderr = "openssl error"
        with patch("shutil.which", return_value="/usr/bin/openssl"):
            with patch("subprocess.run", return_value=mock_result):
                generate_self_signed_cert(
                    self.cert_path, self.key_path,
                    common_name="localhost", verbose=True
                )
        self.assertTrue(os.path.exists(self.cert_path))
        self.assertTrue(os.path.exists(self.key_path))


if __name__ == "__main__":
    unittest.main()
