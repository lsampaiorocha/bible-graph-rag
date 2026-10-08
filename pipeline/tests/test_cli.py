from typer.testing import CliRunner

from kgrag.cli import app


def test_cli_help_describes_the_pipeline():
    result = CliRunner().invoke(app, ["--help"])

    assert result.exit_code == 0
    assert "Bible Graph RAG batch pipeline." in result.output
