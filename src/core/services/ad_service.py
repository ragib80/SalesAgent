# core/services/ad_service.py

import ssl
from dataclasses import dataclass
from urllib.parse import urlparse

from django.conf import settings
from ldap3 import (
    Server,
    Connection,
    ALL,
    NTLM,
    SUBTREE,
    ALL_ATTRIBUTES,
    MODIFY_ADD,
    MODIFY_DELETE,
    Tls,
)
from ldap3.core.exceptions import (
    LDAPSocketOpenError,
    LDAPSSLConfigurationError,
    LDAPStartTLSError,
)
from ldap3.utils.conv import escape_filter_chars

import logging,ssl,hashlib

logger = logging.getLogger(__name__)

@dataclass
class ADModel:
    first_name: str = ""
    middle_name: str = ""
    last_name: str = ""
    display_name: str = ""
    login_name: str = ""
    login_name_with_domain: str = ""
    street_address: str = ""
    city: str = ""
    state: str = ""
    postal_code: str = ""
    country: str = ""
    company: str = ""
    department: str = ""
    home_phone: str = ""
    extension: str = ""
    mobile: str = ""
    fax: str = ""
    email_address: str = ""
    title: str = ""
    manager: str = ""
    employee_id: str = ""
    manager_name: str = ""
    manager_id: str = ""


class ActiveDirectoryService:
    """
    Required settings in settings.py / .env:
      AUTH_LDAP_SERVER_URI         (e.g. ldap://AD1.bergerbd.com:389 or ldaps://AD1.bergerbd.com:636)
      AUTH_LDAP_BIND_DN            (e.g. CN=msfaapp,OU=ServiceAccounts,DC=bergerbd,DC=com)
      AUTH_LDAP_BIND_PASSWORD
      AUTH_LDAP_USER_SEARCH_BASE   (e.g. DC=bergerbd,DC=com)
      AUTH_LDAP_DOMAIN             (optional NetBIOS, e.g. BERGERBD)
      AUTH_LDAP_TLS_STRICT         (True/False)
      AUTH_LDAP_CA_CERT_FILE       (required when TLS_STRICT=True; Base-64 .cer/.pem path)
    """

    def __init__(self):
        raw = settings.AUTH_LDAP_SERVER_URI.strip()

        # Parse URI
        if "://" in raw:
            p = urlparse(raw)
            host = p.hostname
            use_ssl = p.scheme.lower() == "ldaps"
            port = p.port or (636 if use_ssl else 389)
        else:
            host, port = (raw.split(":", 1) + ["389"])[0], int((raw.split(":", 1) + ["389"])[1])
            use_ssl = False

        STRICT_TLS = bool(getattr(settings, "AUTH_LDAP_TLS_STRICT", True))
        CA_FILE    = getattr(settings, "AUTH_LDAP_CA_CERT_FILE", None)

        # Expose for other methods (SET THESE EARLY)
        self.domain       = getattr(settings, "AUTH_LDAP_DOMAIN", None)
        self.search_base  = settings.AUTH_LDAP_USER_SEARCH_BASE
        self.upn_suffix   = getattr(settings, "AUTH_LDAP_UPN_SUFFIX",
                                    self._guess_upn_suffix_from_settings())
        self.disable_ntlm = bool(getattr(settings, "AUTH_LDAP_DISABLE_NTLM", False))
        self.strict_tls   = STRICT_TLS
        self.use_ssl      = use_ssl
        self.port         = port

        tls = Tls(
            validate=ssl.CERT_REQUIRED if STRICT_TLS else ssl.CERT_NONE,
            ca_certs_file=CA_FILE,
        )

        self.server = Server(host, port=port, use_ssl=use_ssl, tls=tls, get_info=ALL, connect_timeout=10)

        bind_user = settings.AUTH_LDAP_BIND_DN
        bind_pass = settings.AUTH_LDAP_BIND_PASSWORD
        self.conn = Connection(self.server, user=bind_user, password=bind_pass, authentication="SIMPLE")

        self.conn.open()
        if STRICT_TLS and not use_ssl and port == 389:
            self.conn.start_tls()
        if not self.conn.bind():
            raise RuntimeError(f"LDAP service bind failed: {self.conn.result}")

    def _guess_upn_suffix_from_settings(self) -> str:
        """
        Build UPN suffix from AUTH_LDAP_USER_SEARCH_BASE (e.g., DC=bergerbd,DC=com -> bergerbd.com).
        Never depends on attributes set later.
        """
        base = getattr(settings, "AUTH_LDAP_USER_SEARCH_BASE", "")
        parts = [p.split("=", 1)[1] for p in base.split(",") if p.strip().upper().startswith("DC=")]
        return ".".join(parts) if parts else "local"

    # ---------- Auth & lookup helpers ----------

    # def authenticate_user(self, username: str, password: str) -> bool:
    #     """
    #     Verify a user's credentials. Uses NTLM if AUTH_LDAP_DOMAIN is set,
    #     otherwise tries SIMPLE with UPN.
    #     """
    #     try:
    #         if self.domain:
    #             user = f"{self.domain}\\{username}"
    #             auth = NTLM
    #         else:
    #             user = f"{username}@{self._guess_upn_suffix()}"
    #             auth = "SIMPLE"

    #         test_conn = Connection(
    #             self.server, user=user, password=password, authentication=auth, auto_bind=True
    #         )
    #         test_conn.unbind()
    #         return True
    #     except Exception:
    #         return False
   

    def authenticate_user(self, username: str, password: str) -> bool:
        r"""
        Try UPN (SIMPLE) first, then DN (SIMPLE).
        Optionally try DOMAIN\user (NTLM) if enabled and MD4 is available.
        """
        un = (username or "").strip()
        if not un or not password:
            return False

        candidates: list[tuple[str, str]] = []
        # Respect explicit UPN/DN passed in
        if "@" in un or "," in un:
            candidates.append(("SIMPLE", un))

        # UPN (SIMPLE)
        candidates.append(("SIMPLE", f"{un}@{self.upn_suffix}"))

        # DN (SIMPLE) – resolve if possible
        try:
            dn = self._get_user_dn(un)
            if dn:
                candidates.append(("SIMPLE", dn))
        except Exception:
            pass

        # DOMAIN\user (NTLM) – only if enabled AND MD4 exists
        md4_ok = "md4" in hashlib.algorithms_available
        if not self.disable_ntlm and md4_ok and self.domain:
            candidates.append(("NTLM", f"{self.domain}\\{un}"))
        else:
            if self.domain:
                logger.debug("Skipping NTLM (disable_ntlm=%s, md4_ok=%s)", self.disable_ntlm, md4_ok)

        # Deduplicate in order (case-insensitive)
        seen = set(); ordered = []
        for auth, principal in candidates:
            key = (auth, principal.lower())
            if key not in seen:
                seen.add(key); ordered.append((auth, principal))

        for auth, principal in ordered:
            try:
                c = Connection(self.server, user=principal, password=password, authentication=auth)
                c.open()
                if self.strict_tls and not self.use_ssl and self.port == 389:
                    c.start_tls()
                ok = c.bind()
                if not ok:
                    res = c.result
                    msg = res.get("message", "")
                    if " 775" in msg:
                        logger.warning("AD bind failed: account LOCKED for %s (%s)", principal, auth)
                    elif " 773" in msg:
                        logger.warning("AD bind failed: password must be reset for %s (%s)", principal, auth)
                    elif " 532" in msg:
                        logger.warning("AD bind failed: password EXPIRED for %s (%s)", principal, auth)
                    elif " 52e" in msg or " 775" not in msg:
                        logger.debug("AD bind failed (%s) for %s: %s", auth, principal, res)
                c.unbind()
                if ok:
                    return True
            except Exception as e:
                logger.debug("AD bind error (%s) for %s: %r", auth, principal, e)

        return False


    def find_user(self, query: str) -> ADModel | None:
        """
        Flexible lookup by sAMAccountName, UPN, mail, cn, or displayName.
        Returns ADModel or None.
        """
        q = escape_filter_chars(query)
        flt = (
            f"(|"
            f"(sAMAccountName={q})"
            f"(userPrincipalName={q})"
            f"(mail={q})"
            f"(cn={q})"
            f"(displayName={q})"
            f")"
        )
        self.conn.search(self.search_base, flt, SUBTREE, attributes=ALL_ATTRIBUTES, size_limit=1)
        return self._map(self.conn.entries[0]) if self.conn.entries else None

    def get_user_by_username(self, username: str) -> ADModel | None:
        # Keep for compatibility; now uses the flexible finder
        return self.find_user(username)

    def get_user_by_login_name(self, username: str) -> ADModel | None:
        return self.find_user(username)

    # ---------- Group ops & utilities (unchanged) ----------

    def get_users_from_group(self, group_name: str) -> list[ADModel]:
        flt = f"(&(objectClass=group)(sAMAccountName={escape_filter_chars(group_name)}))"
        self.conn.search(self.search_base, flt, SUBTREE, attributes=["member"])
        if not self.conn.entries:
            return []
        members = self.conn.entries[0]["member"].values
        users = []
        for dn in members:
            self.conn.search(dn, "(objectClass=user)", attributes=ALL_ATTRIBUTES, search_scope="BASE")
            if self.conn.entries:
                users.append(self._map(self.conn.entries[0]))
        return users

    def add_user_to_group(self, username: str, group_name: str) -> bool:
        group_dn = self._get_group_dn(group_name)
        user_dn = self._get_user_dn(username)
        if not group_dn or not user_dn:
            return False
        self.conn.modify(group_dn, {"member": [(MODIFY_ADD, [user_dn])]})
        return self.conn.result.get("description") == "success"

    def remove_user_from_group(self, username: str, group_name: str) -> bool:
        group_dn = self._get_group_dn(group_name)
        user_dn = self._get_user_dn(username)
        if not group_dn or not user_dn:
            return False
        self.conn.modify(group_dn, {"member": [(MODIFY_DELETE, [user_dn])]})
        return self.conn.result.get("description") == "success"

    def _get_user_dn(self, username: str) -> str | None:
        q = escape_filter_chars(username)
        self.conn.search(
            self.search_base,
            f"(&(objectClass=user)(|(sAMAccountName={q})(userPrincipalName={q})(mail={q})))",
            SUBTREE,
            attributes=["distinguishedName"],
            size_limit=1,
        )
        return self.conn.entries[0]["distinguishedName"].value if self.conn.entries else None

    def _get_group_dn(self, group_name: str) -> str | None:
        q = escape_filter_chars(group_name)
        self.conn.search(
            self.search_base,
            f"(&(objectClass=group)(sAMAccountName={q}))",
            SUBTREE,
            attributes=["distinguishedName"],
            size_limit=1,
        )
        return self.conn.entries[0]["distinguishedName"].value if self.conn.entries else None

    def _guess_upn_suffix(self) -> str:
        # Best-effort, adjust if your UPN suffix differs from DNS domain
        # e.g., "bergerbd.com"
        base_parts = [p.split("=")[1] for p in self.search_base.split(",") if p.strip().upper().startswith("DC=")]
        return ".".join(base_parts) if base_parts else "local"

    def _map(self, entry) -> ADModel:
        attrs = entry.entry_attributes_as_dict
        m = ADModel()
        m.first_name = (attrs.get("givenName") or [""])[0]
        m.middle_name = (attrs.get("initials") or [""])[0]
        m.last_name = (attrs.get("sn") or [""])[0]
        m.display_name = (attrs.get("displayName") or [""])[0]
        m.login_name = (attrs.get("sAMAccountName") or [""])[0]
        upn = (attrs.get("userPrincipalName") or [""])[0]
        if upn and "@" in upn:
            d = upn.split("@")[1].split(".")[0]
            m.login_name_with_domain = f"{d}\\{m.login_name}"
        m.street_address = (attrs.get("streetAddress") or [""])[0]
        m.city = (attrs.get("l") or [""])[0]
        m.state = (attrs.get("st") or [""])[0]
        m.postal_code = (attrs.get("postalCode") or [""])[0]
        m.country = (attrs.get("co") or [""])[0]
        m.company = (attrs.get("company") or [""])[0]
        m.department = (attrs.get("department") or [""])[0]
        m.home_phone = (attrs.get("homePhone") or [""])[0]
        m.extension = (attrs.get("telephoneNumber") or [""])[0]
        m.mobile = (attrs.get("mobile") or [""])[0]
        m.fax = (attrs.get("facsimileTelephoneNumber") or [""])[0]
        m.email_address = (attrs.get("mail") or [""])[0]
        m.title = (attrs.get("title") or [""])[0]
        m.manager = (attrs.get("manager") or [""])[0]
        m.employee_id = (attrs.get("employeeID") or [""])[0]

        if m.manager:
            cn = m.manager.split(",")[0].replace("CN=", "")
            parts = cn.rsplit(" ", 1)
            if len(parts) == 2:
                m.manager_name, m.manager_id = parts
        return m
