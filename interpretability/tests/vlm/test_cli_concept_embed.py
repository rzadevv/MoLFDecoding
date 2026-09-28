"""CLI tests for the VLM sub-app (info, encode-texts, concept-embed)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import typer.testing


@pytest.fixture
def cli_runner() -> typer.testing.CliRunner:
    return typer.testing.CliRunner()


class TestInfoCommand:
    def test_info_fails_without_config(self, cli_runner: typer.testing.CliRunner) -> None:
        from molf_interp.cli.vlm import vlm_app

        result = cli_runner.invoke(vlm_app, ["info", "--config", "nonexistent.yaml"])
        assert result.exit_code != 0

    @patch("molf_interp.cli.vlm.load_vlm")
    def test_info_logs_model_name(
        self,
        mock_load: MagicMock,
        cli_runner: typer.testing.CliRunner,
        tmp_path: Path,
    ) -> None:
        from molf_interp.cli.vlm import vlm_app

        mock_vlm = MagicMock()
        mock_vlm.__class__ = type("V", (), {"device": __import__("torch").device("cpu")})
        mock_vlm.__class__.__name__ = "MockVLM"
        mock_vlm.encode_texts.return_value = __import__("torch").zeros(1, 512)
        mock_load.return_value = mock_vlm

        config_path = tmp_path / "config.yaml"
        config_path.write_text("model_name: conch_v1\ndevice: cpu\nhf_token_env_var: HF_TOKEN\n")
        result = cli_runner.invoke(vlm_app, ["info", "--config", str(config_path)])
        assert result.exit_code == 0
        assert "MockVLM" in result.output or "conch_v1" in result.output


class TestEncodeTextsCommand:
    def test_encode_texts_no_texts_warns(self, cli_runner: typer.testing.CliRunner) -> None:
        from molf_interp.cli.vlm import vlm_app

        with patch("molf_interp.cli.vlm.load_vlm") as mock_load:
            mock_vlm = MagicMock()
            mock_vlm.encode_texts.return_value = __import__("torch").zeros(0, 512)
            mock_load.return_value = mock_vlm

            config_path = Path("configs/vlm/conch_v1.yaml")
            result = cli_runner.invoke(vlm_app, ["encode-texts", "--config", str(config_path)])
            assert "No texts provided" in result.output or "warning" in result.output.lower()


class TestConceptEmbedCommand:
    def test_concept_embed_missing_config(self, cli_runner: typer.testing.CliRunner) -> None:
        from molf_interp.cli.vlm import vlm_app

        result = cli_runner.invoke(vlm_app, ["concept-embed", "--config", "nonexistent.yaml"])
        assert result.exit_code != 0

    def test_concept_embed_force_required_for_existing_output(
        self,
        cli_runner: typer.testing.CliRunner,
        tmp_path: Path,
    ) -> None:
        from molf_interp.cli.vlm import vlm_app

        # Write a stub config
        config = {
            "output_dir": str(tmp_path),
            "output_name": "exists.parquet",
            "concepts_file": str(tmp_path / "c.parquet"),
            "prompts_file": str(tmp_path / "p.parquet"),
            "bank_dir": str(tmp_path),
            "model_config": str(tmp_path / "model.yaml"),
            "vlm_name": "conch_v1",
            "text_batch_size": 32,
            "exclude_human_only_for_mouse": True,
        }
        import yaml

        (tmp_path / "exists.parquet").touch()
        (tmp_path / "c.parquet").write_bytes(b"")
        (tmp_path / "p.parquet").write_bytes(b"")
        (tmp_path / "model.yaml").write_text(
            "model_name: conch_v1\ndevice: cpu\nhf_token_env_var: HF_TOKEN\n"
        )
        (tmp_path / "config.yaml").write_text(yaml.dump(config))

        result = cli_runner.invoke(
            vlm_app,
            ["concept-embed", "--config", str(tmp_path / "config.yaml")],
        )
        assert result.exit_code != 0
        assert "already exists" in result.stdout or "already exists" in result.stderr

    def test_concept_embed_runs_with_valid_config(
        self,
        cli_runner: typer.testing.CliRunner,
        tmp_path: Path,
    ) -> None:
        from molf_interp.cli.vlm import vlm_app

        concepts_df = __import__("pandas").DataFrame(
            {
                "concept_id": ["MOR-0001", "CTY-0001"],
                "concept_name": ["Fibrosis", "T cell"],
                "tier": ["H1_morphology", "H2_cell_type"],
                "category": ["a", "b"],
                "subcategory": ["c", "d"],
                "source": ["e", "f"],
                "provenance": ["reference", "reference"],
                "organ": ["lung", "lung"],
                "level": ["tissue", None],
                "concept_type": [None, "cell_type"],
                "cross_bank_status": ["present", "present"],
                "in_primary_lexicon": [True, True],
            }
        )
        prompts_df = __import__("pandas").DataFrame(
            {
                "concept_id": ["MOR-0001", "MOR-0001", "CTY-0001"],
                "concept_name": ["Fibrosis", "Fibrosis", "T cell"],
                "tier": ["H1_morphology", "H1_morphology", "H2_cell_type"],
                "species": ["Homo_sapiens", "Mus_musculus", "Homo_sapiens"],
                "prompt_text": ["Fibrosis text", "Mouse fibrosis text", "T cell text"],
                "species_status": ["shared", "shared", "shared"],
            }
        )
        concepts_df.to_parquet(tmp_path / "merged_concepts.parquet")
        prompts_df.to_parquet(tmp_path / "merged_prompts.parquet")

        config = {
            "output_dir": str(tmp_path),
            "output_name": "embeddings.parquet",
            "concepts_file": str(tmp_path / "merged_concepts.parquet"),
            "prompts_file": str(tmp_path / "merged_prompts.parquet"),
            "bank_dir": str(tmp_path),
            "model_config": str(tmp_path / "model.yaml"),
            "vlm_name": "conch_v1",
            "text_batch_size": 32,
            "exclude_human_only_for_mouse": True,
        }
        import yaml

        (tmp_path / "model.yaml").write_text(
            "model_name: conch_v1\ndevice: cpu\nhf_token_env_var: HF_TOKEN\n"
        )
        (tmp_path / "config.yaml").write_text(yaml.dump(config))

        with patch("molf_interp.cli.vlm.load_vlm") as mock_load:
            mock_vlm = MagicMock()
            mock_vlm.encode_texts.return_value = __import__("torch").zeros(3, 512)
            mock_load.return_value = mock_vlm

            result = cli_runner.invoke(
                vlm_app,
                ["concept-embed", "--config", str(tmp_path / "config.yaml"), "--force"],
            )
            assert result.exit_code == 0, f"stderr: {result.stderr}\nexception: {result.exception}"
            assert (tmp_path / "embeddings.parquet").exists()
