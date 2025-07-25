# Generated by Django 4.2.21 on 2025-05-24 15:38

from django.db import migrations, models


class Migration(migrations.Migration):
    initial = True

    dependencies = []

    operations = [
        migrations.CreateModel(
            name="GoogleDocument",
            fields=[
                (
                    "id",
                    models.BigAutoField(
                        auto_created=True,
                        primary_key=True,
                        serialize=False,
                        verbose_name="ID",
                    ),
                ),
                ("doc_id", models.CharField(max_length=255, unique=True)),
                ("title", models.CharField(max_length=500)),
                ("content", models.TextField()),
                ("last_modified", models.DateTimeField()),
                ("processed_at", models.DateTimeField(auto_now_add=True)),
            ],
            options={
                "ordering": ["-processed_at"],
            },
        ),
        migrations.CreateModel(
            name="ProcessingStatus",
            fields=[
                (
                    "id",
                    models.BigAutoField(
                        auto_created=True,
                        primary_key=True,
                        serialize=False,
                        verbose_name="ID",
                    ),
                ),
                (
                    "status",
                    models.CharField(
                        choices=[
                            ("idle", "Idle"),
                            ("processing", "Processing"),
                            ("completed", "Completed"),
                            ("error", "Error"),
                        ],
                        default="idle",
                        max_length=20,
                    ),
                ),
                ("message", models.TextField(blank=True)),
                ("last_check", models.DateTimeField(auto_now=True)),
            ],
            options={
                "verbose_name_plural": "Processing Status",
            },
        ),
    ]
