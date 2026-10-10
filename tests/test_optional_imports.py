from __future__ import annotations

import subprocess
import sys
import textwrap
import unittest


class OptionalImportTests(unittest.TestCase):
    def test_diffusers_adapter_import_does_not_load_optional_dependencies(self) -> None:
        code = textwrap.dedent(
            """
            import builtins
            real_import = builtins.__import__

            def guarded_import(name, *args, **kwargs):
                if name.split('.', 1)[0] in {'torch', 'diffusers'}:
                    raise AssertionError(f'optional dependency imported eagerly: {name}')
                return real_import(name, *args, **kwargs)

            builtins.__import__ = guarded_import
            import peek.diffusers
            """
        )

        completed = subprocess.run(
            [sys.executable, "-c", code],
            cwd=".",
            text=True,
            capture_output=True,
            check=False,
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)


if __name__ == "__main__":
    unittest.main()
