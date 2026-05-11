from django.test import SimpleTestCase

from user_auth.views import (
    microsoft_authorization_bridge_response,
    microsoft_authorization_url,
)


class MicrosoftAuthorizationUrlTests(SimpleTestCase):
    def test_wrong_host_authorize_url_is_forced_to_microsoft(self):
        fixed_url = microsoft_authorization_url(
            "https://financeexpiry.bergerbd.com/tenant-id/oauth2/v2.0/authorize?client_id=abc"
        )

        self.assertEqual(
            fixed_url,
            "https://login.microsoftonline.com/tenant-id/oauth2/v2.0/authorize?client_id=abc",
        )

    def test_relative_authorize_url_is_forced_to_microsoft(self):
        fixed_url = microsoft_authorization_url(
            "/tenant-id/oauth2/v2.0/authorize?state=xyz"
        )

        self.assertEqual(
            fixed_url,
            "https://login.microsoftonline.com/tenant-id/oauth2/v2.0/authorize?state=xyz",
        )

    def test_bridge_response_does_not_emit_location_header(self):
        response = microsoft_authorization_bridge_response(
            "/tenant-id/oauth2/v2.0/authorize?state=xyz"
        )

        self.assertEqual(response.status_code, 200)
        self.assertNotIn("Location", response)
        self.assertIn(b"window.location.replace", response.content)

    def test_bridge_response_escapes_script_breakout_characters(self):
        response = microsoft_authorization_bridge_response(
            "/tenant-id/oauth2/v2.0/authorize?state=</script>"
        )

        self.assertIn(b"\\u003c/script\\u003e", response.content)
