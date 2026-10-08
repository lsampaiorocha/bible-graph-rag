import ast
from pathlib import Path

import kgrag


def test_pipeline_package_never_imports_eval_package():
    package_dir = Path(kgrag.__file__).parent
    offending_imports = []
    for source_file in sorted(package_dir.rglob("*.py")):
        for node in ast.walk(ast.parse(source_file.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                modules = [node.module or ""]
            else:
                continue
            offending_imports += [
                f"{source_file.name}: {module}"
                for module in modules
                if module.split(".")[0] == "kgrag_eval"
            ]

    assert offending_imports == []
