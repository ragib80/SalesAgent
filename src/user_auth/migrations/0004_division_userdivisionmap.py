import django.db.models.deletion
from django.conf import settings
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('user_auth', '0003_merge_20260308_1307'),
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
    ]

    operations = [
        migrations.CreateModel(
            name='Division',
            fields=[
                ('id', models.BigIntegerField(db_column='id', primary_key=True, serialize=False)),
                ('code', models.CharField(db_column='Code', max_length=255)),
                ('name', models.CharField(db_column='Name', max_length=255)),
            ],
            options={
                'db_table': 'Division',
                'managed': False,
            },
        ),
        migrations.CreateModel(
            name='UserDivisionMap',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('created_at', models.DateTimeField(auto_now_add=True)),
                ('division', models.ForeignKey(db_column='division_id', on_delete=django.db.models.deletion.CASCADE, related_name='user_links', to='user_auth.division')),
                ('user', models.ForeignKey(db_column='user_id', on_delete=django.db.models.deletion.CASCADE, related_name='division_links', to=settings.AUTH_USER_MODEL)),
            ],
            options={
                'db_table': 'UserDivisionMap',
            },
        ),
        migrations.AddIndex(
            model_name='userdivisionmap',
            index=models.Index(fields=['user'], name='ix_udivmap_user'),
        ),
        migrations.AddIndex(
            model_name='userdivisionmap',
            index=models.Index(fields=['division'], name='ix_udivmap_div'),
        ),
        migrations.AddConstraint(
            model_name='userdivisionmap',
            constraint=models.UniqueConstraint(fields=('user', 'division'), name='uq_user_division'),
        ),
    ]
