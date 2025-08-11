import re
from urllib.parse import urlparse
from dataclasses import dataclass
from ldap3 import (
    Server,
    Connection,
    ALL,
    NTLM,
    Tls,
    SUBTREE,
    ALL_ATTRIBUTES,
    MODIFY_ADD,
    MODIFY_DELETE,
)
from django.conf import settings
from ldap3.core.exceptions import LDAPSocketOpenError, LDAPSSLConfigurationError, LDAPStartTLSError
import ssl

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
    def __init__(self):
        raw = settings.AUTH_LDAP_SERVER_URI.strip()  # e.g. ldap://AD1.bergerbd.com:389

        # Normalize URI -> (host, port, use_ssl)
        if "://" in raw:
            p = urlparse(raw)
            host = p.hostname
            use_ssl = p.scheme.lower() == "ldaps"
            port = p.port or (636 if use_ssl else 389)
        else:
            if ":" in raw:
                host, port_str = raw.rsplit(":", 1)
                host, port = host.strip(), int(port_str)
            else:
                host, port = raw, 389
            use_ssl = False

        STRICT_TLS = bool(getattr(settings, "AUTH_LDAP_TLS_STRICT", True))
        CA_FILE    = getattr(settings, "AUTH_LDAP_CA_CERT_FILE", None)
        bind_user  = settings.AUTH_LDAP_BIND_DN
        bind_pass  = settings.AUTH_LDAP_BIND_PASSWORD

        # Make these available to other methods
        self.domain      = getattr(settings, "AUTH_LDAP_DOMAIN", None)
        self.search_base = settings.AUTH_LDAP_USER_SEARCH_BASE

        # Build TLS context (respects STRICT flag)
        tls = Tls(
            validate=ssl.CERT_REQUIRED if STRICT_TLS else ssl.CERT_NONE,
            ca_certs_file=CA_FILE,
        )

        self.server = Server(host, port=port, use_ssl=use_ssl, tls=tls, get_info=ALL, connect_timeout=10)

        # DN -> SIMPLE bind for service account
        self.conn = Connection(self.server, user=bind_user, password=bind_pass, authentication="SIMPLE")

        self.conn.open()

        # DEV ONLY: if STRICT is False, skip StartTLS entirely (no handshake, no cert checks)
        if STRICT_TLS and not use_ssl and port == 389:
            try:
                self.conn.start_tls()
            except (LDAPStartTLSError, LDAPSSLConfigurationError, LDAPSocketOpenError):
                # In strict mode, fail fast so you know to supply the CA or fix hostname
                raise

        # Finally bind
        self.conn.bind()


    def authenticate_user(self, username: str, password: str) -> bool:
        """Try an NTLM bind using the supplied credentials."""
        user = f"{self.domain}\\{username}"
        try:
            user_conn = Connection(
                self.server,
                user=user,
                password=password,
                authentication=NTLM,
                auto_bind=True,
            )
            user_conn.unbind()
            return True
        except Exception:
            return False

    def get_user_by_username(self, username: str) -> ADModel | None:
        """Search for a single user by sAMAccountName."""
        flt = f"(&(objectClass=user)(sAMAccountName={username}))"
        self.conn.search(self.search_base, flt, SUBTREE, attributes=ALL_ATTRIBUTES)
        if not self.conn.entries:
            return None
        return self._map(self.conn.entries[0])

    def get_user_by_login_name(self, username: str) -> ADModel | None:
        """Alias of get_user_by_username."""
        return self.get_user_by_username(username)

    def get_user_details_by_full_name(
        self, first_name: str, middle_name: str, last_name: str
    ) -> ADModel | None:
        """Search by givenName, initials, and/or sn."""
        parts = []
        if first_name:
            parts.append(f"(givenName={first_name})")
        if middle_name:
            parts.append(f"(initials={middle_name})")
        if last_name:
            parts.append(f"(sn={last_name})")
        if not parts:
            return None
        flt = f"(&(objectClass=user){''.join(parts)})"
        self.conn.search(self.search_base, flt, SUBTREE, attributes=ALL_ATTRIBUTES)
        if not self.conn.entries:
            return None
        return self._map(self.conn.entries[0])

    def get_users_from_group(self, group_name: str) -> list[ADModel]:
        """Return all members of a given group."""
        flt = f"(&(objectClass=group)(sAMAccountName={group_name}))"
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

    def get_users_by_first_name(self, first_name: str) -> list[ADModel]:
        """Find users whose givenName starts with the supplied string."""
        flt = f"(&(objectClass=user)(givenName={first_name}*))"
        self.conn.search(self.search_base, flt, SUBTREE, attributes=ALL_ATTRIBUTES)
        return [self._map(e) for e in self.conn.entries]

    def add_user_to_group(self, username: str, group_name: str) -> bool:
        """Add a user to an AD group."""
        group_dn = self._get_group_dn(group_name)
        user_dn = self._get_user_dn(username)
        if not group_dn or not user_dn:
            return False
        self.conn.modify(group_dn, {"member": [(MODIFY_ADD, [user_dn])]})
        return self.conn.result["description"] == "success"

    def remove_user_from_group(self, username: str, group_name: str) -> bool:
        """Remove a user from an AD group."""
        group_dn = self._get_group_dn(group_name)
        user_dn = self._get_user_dn(username)
        if not group_dn or not user_dn:
            return False
        self.conn.modify(group_dn, {"member": [(MODIFY_DELETE, [user_dn])]})
        return self.conn.result["description"] == "success"

    def _get_user_dn(self, username: str) -> str | None:
        self.conn.search(
            self.search_base,
            f"(&(objectClass=user)(sAMAccountName={username}))",
            SUBTREE,
            attributes=["distinguishedName"],
        )
        return self.conn.entries[0]["distinguishedName"].value if self.conn.entries else None

    def _get_group_dn(self, group_name: str) -> str | None:
        self.conn.search(
            self.search_base,
            f"(&(objectClass=group)(sAMAccountName={group_name}))",
            SUBTREE,
            attributes=["distinguishedName"],
        )
        return self.conn.entries[0]["distinguishedName"].value if self.conn.entries else None

    def _map(self, entry) -> ADModel:
        """Map an ldap3 Entry into our ADModel dataclass."""
        m = ADModel()
        attrs = entry.entry_attributes_as_dict
        m.first_name = attrs.get("givenName", [""])[0]
        m.middle_name = attrs.get("initials", [""])[0]
        m.last_name = attrs.get("sn", [""])[0]
        m.display_name = attrs.get("displayName", [""])[0]
        m.login_name = attrs.get("sAMAccountName", [""])[0]
        upn = attrs.get("userPrincipalName", [""])[0]
        if upn and "@" in upn:
            d = upn.split("@")[1].split(".")[0]
            m.login_name_with_domain = f"{d}\\{m.login_name}"
        m.street_address = attrs.get("streetAddress", [""])[0]
        m.city = attrs.get("l", [""])[0]
        m.state = attrs.get("st", [""])[0]
        m.postal_code = attrs.get("postalCode", [""])[0]
        m.country = attrs.get("co", [""])[0]
        m.company = attrs.get("company", [""])[0]
        m.department = attrs.get("department", [""])[0]
        m.home_phone = attrs.get("homePhone", [""])[0]
        m.extension = attrs.get("telephoneNumber", [""])[0]
        m.mobile = attrs.get("mobile", [""])[0]
        m.fax = attrs.get("facsimileTelephoneNumber", [""])[0]
        m.email_address = attrs.get("mail", [""])[0]
        m.title = attrs.get("title", [""])[0]
        m.manager = attrs.get("manager", [""])[0]
        m.employee_id = attrs.get("employeeID", [""])[0]

        if m.manager:
            cn = m.manager.split(",")[0].replace("CN=", "")
            parts = cn.rsplit(" ", 1)
            if len(parts) == 2:
                m.manager_name, m.manager_id = parts
        return m
