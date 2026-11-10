from unittest.mock import MagicMock, patch

from django.test import SimpleTestCase, override_settings

from custom_admin.forms import SalesAuthUserCreateFromADForm
from core.services.microsoft_graph_service import (
    GraphDirectoryUser,
    MicrosoftGraphDirectoryService,
)


@override_settings(
    MICROSOFT_AUTH_CLIENT_ID="client-id",
    MICROSOFT_AUTH_TENANT_ID="tenant-id",
    MICROSOFT_AUTH_CLIENT_SECRET="client-secret",
    MICROSOFT_GRAPH_DEFAULT_UPN_SUFFIX="example.com",
)
class MicrosoftGraphDirectoryServiceTests(SimpleTestCase):
    @patch.object(MicrosoftGraphDirectoryService, "_get_user")
    def test_find_user_appends_default_upn_suffix_for_bare_username(self, mock_get_user):
        expected = GraphDirectoryUser(
            tenant_id="tenant-id",
            object_id="object-id",
            login_name="jdoe",
            user_principal_name="jdoe@example.com",
            email_address="jdoe@example.com",
        )
        mock_get_user.side_effect = [None, expected]

        service = MicrosoftGraphDirectoryService()
        result = service.find_user("jdoe")

        self.assertEqual(result, expected)
        self.assertEqual(
            [call.args[0] for call in mock_get_user.call_args_list],
            ["jdoe", "jdoe@example.com"],
        )

    @patch.object(MicrosoftGraphDirectoryService, "_list_users")
    @patch.object(MicrosoftGraphDirectoryService, "_get_user", return_value=None)
    def test_find_user_falls_back_to_graph_filter_lookup(self, _mock_get_user, mock_list_users):
        expected = GraphDirectoryUser(
            tenant_id="tenant-id",
            object_id="object-id",
            login_name="jdoe",
            mail_nickname="jdoe",
            user_principal_name="jdoe@example.com",
            email_address="jdoe@example.com",
        )
        mock_list_users.side_effect = [[expected]]

        service = MicrosoftGraphDirectoryService()
        result = service.find_user("jdoe")

        self.assertEqual(result, expected)
        self.assertEqual(mock_list_users.call_args.args[0], "mailNickname eq 'jdoe'")


class SalesAuthUserCreateFromADFormTests(SimpleTestCase):
    @patch("custom_admin.forms.MicrosoftGraphDirectoryService")
    @patch("custom_admin.forms.User.objects.filter")
    def test_form_uses_graph_profile_for_new_user(self, mock_filter, mock_graph_service):
        username_qs = MagicMock()
        username_qs.exists.return_value = False

        identity_qs = MagicMock()
        identity_qs.first.return_value = None

        mock_filter.side_effect = [username_qs, identity_qs]
        mock_graph_service.return_value.find_user.return_value = GraphDirectoryUser(
            tenant_id="tenant-id",
            object_id="object-id",
            first_name="Jane",
            last_name="Doe",
            login_name="jdoe",
            user_principal_name="jdoe@example.com",
            email_address="jdoe@example.com",
        )

        form = SalesAuthUserCreateFromADForm(data={"ad_identifier": "jdoe"})

        self.assertTrue(form.is_valid(), form.errors.as_json())

        user = form.save(commit=False)
        self.assertEqual(user.username, "jdoe")
        self.assertEqual(user.email, "jdoe@example.com")
        self.assertEqual(user.first_name, "Jane")
        self.assertEqual(user.last_name, "Doe")
        self.assertEqual(user.identity_provider, user.IDENTITY_PROVIDER_MICROSOFT)
        self.assertEqual(user.azure_ad_tenant_id, "tenant-id")
        self.assertEqual(user.azure_ad_object_id, "object-id")
        self.assertFalse(user.has_usable_password())
