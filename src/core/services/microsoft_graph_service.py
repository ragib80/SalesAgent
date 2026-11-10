import logging
from dataclasses import dataclass
from urllib.parse import quote

import msal
import requests
from django.conf import settings
from django.core.exceptions import ImproperlyConfigured

logger = logging.getLogger(__name__)


class MicrosoftGraphServiceError(RuntimeError):
    pass


@dataclass
class GraphDirectoryUser:
    tenant_id: str = ""
    object_id: str = ""
    first_name: str = ""
    last_name: str = ""
    display_name: str = ""
    login_name: str = ""
    mail_nickname: str = ""
    user_principal_name: str = ""
    email_address: str = ""


class MicrosoftGraphDirectoryService:
    GRAPH_BASE_URL = "https://graph.microsoft.com/v1.0"
    GRAPH_SCOPE = ["https://graph.microsoft.com/.default"]
    USER_SELECT = ",".join(
        [
            "id",
            "displayName",
            "givenName",
            "mail",
            "mailNickname",
            "onPremisesSamAccountName",
            "surname",
            "userPrincipalName",
        ]
    )

    def __init__(self):
        self.tenant_id = (getattr(settings, "MICROSOFT_AUTH_TENANT_ID", "") or "").strip().strip("/")
        self.client_id = (getattr(settings, "MICROSOFT_AUTH_CLIENT_ID", "") or "").strip()
        self.client_secret = getattr(settings, "MICROSOFT_AUTH_CLIENT_SECRET", "") or ""
        self.authority = (
            getattr(settings, "MICROSOFT_AUTH_AUTHORITY", "") or ""
        ).strip() or f"https://login.microsoftonline.com/{self.tenant_id}"
        self.default_upn_suffix = (
            getattr(settings, "MICROSOFT_GRAPH_DEFAULT_UPN_SUFFIX", "") or ""
        ).strip().lstrip("@")

        missing = [
            name
            for name, value in (
                ("MICROSOFT_AUTH_TENANT_ID", self.tenant_id),
                ("MICROSOFT_AUTH_CLIENT_ID", self.client_id),
                ("MICROSOFT_AUTH_CLIENT_SECRET", self.client_secret),
            )
            if not value
        ]
        if missing:
            raise ImproperlyConfigured(
                "Missing Microsoft Graph setting(s): " + ", ".join(missing)
            )

        self._app = msal.ConfidentialClientApplication(
            client_id=self.client_id,
            authority=self.authority,
            client_credential=self.client_secret,
        )

    def _acquire_access_token(self) -> str:
        result = self._app.acquire_token_silent(self.GRAPH_SCOPE, account=None)
        if not result:
            result = self._app.acquire_token_for_client(scopes=self.GRAPH_SCOPE)

        token = (result or {}).get("access_token")
        if token:
            return token

        detail = (
            (result or {}).get("error_description")
            or (result or {}).get("error")
            or "Token acquisition failed."
        )
        raise MicrosoftGraphServiceError(detail)

    def _request(self, method: str, path: str, *, params=None, headers=None):
        response = requests.request(
            method,
            f"{self.GRAPH_BASE_URL}{path}",
            headers={
                "Accept": "application/json",
                "Authorization": f"Bearer {self._acquire_access_token()}",
                **(headers or {}),
            },
            params=params,
            timeout=(5, 20),
        )
        if response.status_code == 404:
            return None
        if response.status_code >= 400:
            detail = response.text
            try:
                payload = response.json()
                detail = payload.get("error", {}).get("message") or detail
            except ValueError:
                pass
            raise MicrosoftGraphServiceError(
                f"Graph API request failed ({response.status_code}): {detail}"
            )
        if response.status_code == 204:
            return {}
        return response.json()

    def _get_user(self, user_id_or_upn: str) -> GraphDirectoryUser | None:
        if not user_id_or_upn:
            return None

        try:
            data = self._request(
                "GET",
                f"/users/{quote(user_id_or_upn, safe='@._-')}",
                params={"$select": self.USER_SELECT},
            )
        except MicrosoftGraphServiceError as exc:
            if "(400)" in str(exc) or "(404)" in str(exc):
                return None
            raise
        return self._map_user(data) if data else None

    def _list_users(self, filter_expr: str) -> list[GraphDirectoryUser]:
        data = self._request(
            "GET",
            "/users",
            params={
                "$filter": filter_expr,
                "$select": self.USER_SELECT,
                "$top": 10,
            },
        )
        return [self._map_user(item) for item in (data or {}).get("value", [])]

    @staticmethod
    def _odata_string(value: str) -> str:
        return (value or "").replace("'", "''")

    def _map_user(self, payload: dict) -> GraphDirectoryUser:
        user_principal_name = (payload.get("userPrincipalName") or "").strip()
        email = (payload.get("mail") or "").strip()
        mail_nickname = (payload.get("mailNickname") or "").strip()
        on_prem_login = (payload.get("onPremisesSamAccountName") or "").strip()

        login_name = on_prem_login or mail_nickname
        if not login_name and user_principal_name and "@" in user_principal_name:
            login_name = user_principal_name.split("@", 1)[0]
        if not login_name and email and "@" in email:
            login_name = email.split("@", 1)[0]

        return GraphDirectoryUser(
            tenant_id=self.tenant_id,
            object_id=(payload.get("id") or "").strip(),
            first_name=(payload.get("givenName") or "").strip(),
            last_name=(payload.get("surname") or "").strip(),
            display_name=(payload.get("displayName") or "").strip(),
            login_name=login_name.strip(),
            mail_nickname=mail_nickname,
            user_principal_name=user_principal_name,
            email_address=email or user_principal_name,
        )

    def _pick_exact_match(
        self,
        candidates: list[GraphDirectoryUser],
        identifier: str,
    ) -> GraphDirectoryUser | None:
        ident = (identifier or "").strip().lower()
        if not ident:
            return None

        exact = []
        upn_prefix_matches = []

        for user in candidates:
            values = {
                (user.login_name or "").lower(),
                (user.mail_nickname or "").lower(),
                (user.user_principal_name or "").lower(),
                (user.email_address or "").lower(),
            }
            values.discard("")

            if ident in values:
                exact.append(user)
                continue

            if "@" not in ident:
                upn = (user.user_principal_name or "").lower()
                if upn.startswith(f"{ident}@"):
                    upn_prefix_matches.append(user)

        if len(exact) == 1:
            return exact[0]
        if exact:
            return exact[0]
        if len(upn_prefix_matches) == 1:
            return upn_prefix_matches[0]
        if upn_prefix_matches:
            return upn_prefix_matches[0]
        return None

    def find_user(self, query: str) -> GraphDirectoryUser | None:
        identifier = (query or "").strip()
        if not identifier:
            return None

        direct_candidates = [identifier]
        if "@" not in identifier and self.default_upn_suffix:
            direct_candidates.append(f"{identifier}@{self.default_upn_suffix}")

        for candidate in direct_candidates:
            user = self._get_user(candidate)
            if user:
                return user

        q = self._odata_string(identifier)
        filter_attempts = []

        if "@" in identifier:
            filter_attempts.append(f"userPrincipalName eq '{q}' or mail eq '{q}'")
        else:
            filter_attempts.extend(
                [
                    f"mailNickname eq '{q}'",
                    f"startswith(userPrincipalName,'{q}@')",
                    f"onPremisesSamAccountName eq '{q}'",
                ]
            )

        for filter_expr in filter_attempts:
            try:
                matches = self._list_users(filter_expr)
            except MicrosoftGraphServiceError as exc:
                error_text = str(exc).lower()
                if "unsupported" in error_text or "invalid query filter" in error_text:
                    logger.debug("Skipping unsupported Graph filter: %s", filter_expr)
                    continue
                raise

            user = self._pick_exact_match(matches, identifier)
            if user:
                return user

        return None
