"""
Dual-sink logger: escribe en consola Y en archivo simultáneamente.
Uso:
    log = Logger(exp_dir / "run.log")
    log("texto")          # INFO
    log.section("título") # bloque separado
    log.close()           # flush final
"""

import io
import sys
from datetime import datetime
from pathlib import Path

# Forzar UTF-8 en stdout de Windows (evita UnicodeEncodeError con caracteres especiales)
if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")


class Logger:
    SEP_THICK = "=" * 72
    SEP_THIN  = "-" * 72

    def __init__(self, log_path: Path):
        log_path = Path(log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(log_path, "w", encoding="utf-8")
        self.path = log_path

    # ── escritura base ───────────────────────────────────────────────────────

    def _write(self, text: str, newline: bool = True):
        end = "\n" if newline else ""
        print(text, end=end, flush=True)
        self._file.write(text + end)
        self._file.flush()

    def __call__(self, text: str = ""):
        self._write(text)

    # ── helpers de formato ───────────────────────────────────────────────────

    def section(self, title: str):
        self._write("")
        self._write(self.SEP_THICK)
        self._write(f"  {title}")
        self._write(self.SEP_THICK)

    def subsection(self, title: str):
        self._write(f"\n{self.SEP_THIN}")
        self._write(f"  {title}")
        self._write(self.SEP_THIN)

    def kv(self, key: str, value, indent: int = 2):
        self._write(f"{' ' * indent}{key:<30s}: {value}")

    def table_row(self, cells: list, widths: list):
        row = "  " + "  ".join(str(c).ljust(w) for c, w in zip(cells, widths))
        self._write(row)

    def timestamp(self, label: str = ""):
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self._write(f"  {label + '  ' if label else ''}{ts}")

    def close(self):
        self._write("")
        self._write(self.SEP_THICK)
        self.timestamp("Log cerrado:")
        self._write(self.SEP_THICK)
        self._file.close()
        print(f"\nLog guardado en: {self.path}")
