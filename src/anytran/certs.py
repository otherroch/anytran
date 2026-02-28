import os
import shutil
import subprocess
import tempfile


def generate_self_signed_cert(cert_path, key_path, common_name="localhost", verbose=False):
    openssl_path = shutil.which("openssl")
    if not openssl_path:
        openssl_path = None

    cn = common_name or "localhost"
    if cn in {"0.0.0.0", "::"}:
        cn = "localhost"

    try:
        import ipaddress
    except Exception:
        ipaddress = None

    def is_ip(value):
        if not ipaddress:
            return False
        try:
            ipaddress.ip_address(value)
            return True
        except ValueError:
            return False

    san_values = []
    if cn:
        san_values.append(f"IP:{cn}" if is_ip(cn) else f"DNS:{cn}")

    if cn != "localhost":
        san_values.append("DNS:localhost")

    san_ext = None
    if san_values:
        san_ext = "subjectAltName = " + ",".join(san_values)

    env = os.environ.copy()
    config_arg = []
    temp_conf_path = None  # Track temp file for cleanup
    if not env.get("OPENSSL_CONF"):
        conda_prefix = env.get("CONDA_PREFIX")
        if conda_prefix:
            conda_conf = os.path.join(conda_prefix, "Library", "ssl", "openssl.cnf")
            if os.path.exists(conda_conf):
                env["OPENSSL_CONF"] = conda_conf
        if not env.get("OPENSSL_CONF"):
            local_conf = os.path.join(os.getcwd(), "openssl.cnf")
            if os.path.exists(local_conf):
                env["OPENSSL_CONF"] = local_conf
        if not env.get("OPENSSL_CONF") and openssl_path:
            minimal_conf = f"""[ req ]
distinguished_name = req_distinguished_name
prompt = no

[ req_distinguished_name ]
CN = {cn}
"""
            with tempfile.NamedTemporaryFile(delete=False, suffix=".cnf") as conf_fp:
                conf_fp.write(minimal_conf.encode("utf-8"))
                temp_conf_path = conf_fp.name
                env["OPENSSL_CONF"] = temp_conf_path
        if env.get("OPENSSL_CONF"):
            config_arg = ["-config", env["OPENSSL_CONF"]]

    cmd = [
        openssl_path,
        "req",
        "-x509",
        *config_arg,
        "-newkey",
        "rsa:2048",
        "-nodes",
        "-keyout",
        cert_path.replace(".crt", ".key") if key_path.endswith(".crt") else key_path,
        "-out",
        cert_path,
        "-days",
        "365",
        "-subj",
        f"/CN={cn}",
    ]

    if san_ext:
        cmd.extend(["-addext", san_ext])

    if verbose:
        print(f"Generating self-signed cert: {' '.join(cmd)}")

    if openssl_path:
        result = subprocess.run(cmd, capture_output=True, text=True, env=env)
        if result.returncode == 0:
            if temp_conf_path and os.path.exists(temp_conf_path):
                os.remove(temp_conf_path)  # Clean up temp config file
            return
        openssl_error = result.stderr.strip()
    else:
        openssl_error = "OpenSSL not found in PATH"

    if verbose and openssl_error:
        print(f"OpenSSL error: {openssl_error}")

    try:
        from datetime import datetime, timedelta, timezone
        from cryptography import x509
        from cryptography.hazmat.primitives import hashes, serialization
        from cryptography.hazmat.primitives.asymmetric import rsa
        from cryptography.x509.oid import NameOID
    except Exception as exc:
        # Clean up temp file before raising
        if temp_conf_path and os.path.exists(temp_conf_path):
            os.remove(temp_conf_path)
        raise RuntimeError(
            f"OpenSSL failed ({openssl_error}). Install OpenSSL or cryptography, "
            "or provide --web-ssl-cert/--web-ssl-key."
        ) from exc

    if verbose:
        print("OpenSSL failed; falling back to cryptography for self-signed cert.")

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    san_items = []
    if cn:
        san_items.append(("ip" if is_ip(cn) else "dns", cn))
    if cn != "localhost":
        san_items.append(("dns", "localhost"))

    san_entries = []
    for kind, value in san_items:
        if kind == "ip" and ipaddress:
            san_entries.append(x509.IPAddress(ipaddress.ip_address(value)))
        else:
            san_entries.append(x509.DNSName(value))

    subject = issuer = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, cn)])
    cert = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(issuer)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(datetime.now(timezone.utc) - timedelta(days=1))
        .not_valid_after(datetime.now(timezone.utc) + timedelta(days=365))
        .add_extension(x509.SubjectAlternativeName(san_entries), critical=False)
        .sign(key, hashes.SHA256())
    )

    with open(cert_path, "wb") as cert_file:
        cert_file.write(cert.public_bytes(serialization.Encoding.PEM))
    with open(key_path, "wb") as key_file:
        key_file.write(
            key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.TraditionalOpenSSL,
                encryption_algorithm=serialization.NoEncryption(),
            )
        )
    
    # Clean up temp config file after successful cryptography fallback
    if temp_conf_path and os.path.exists(temp_conf_path):
        os.remove(temp_conf_path)
