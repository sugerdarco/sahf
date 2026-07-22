"""
Everything a run produces goes under out/:

  out/
    logs/                app.log — process-level log (setup, errors, warnings)
    runs/
      run_YYYYmmdd_HHMMSS_<label>/
        meta.json         config + prompt for this run
        steps.jsonl        one JSON object per decoding step (gate scores, path
                            taken, outlier flags, escalation info, chosen token)
        result.json         final prompt/output text + summary stats

steps.jsonl (not a single json array) so a run can be tailed/inspected while it's
still in progress, and so a crash mid-generation doesn't corrupt the whole file.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional


def setup_app_logging(out_dir: str = "out") -> logging.Logger:
    log_dir = Path(out_dir) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("sahf_lite")
    logger.setLevel(logging.INFO)
    if not logger.handlers:  # avoid duplicate handlers on repeated setup() calls
        fh = logging.FileHandler(log_dir / "app.log")
        fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
        logger.addHandler(fh)
        sh = logging.StreamHandler()
        sh.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
        logger.addHandler(sh)
    return logger


class RunLogger:
    def __init__(self, out_dir: str = "out", run_label: Optional[str] = None):
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_name = f"run_{stamp}" + (f"_{run_label}" if run_label else "")
        self.run_dir = Path(out_dir) / "runs" / self.run_name
        self.run_dir.mkdir(parents=True, exist_ok=True)

        self.steps_path = self.run_dir / "steps.jsonl"
        self.meta_path = self.run_dir / "meta.json"
        self.result_path = self.run_dir / "result.json"
        self._steps_file = open(self.steps_path, "a")

    def log_meta(self, meta: dict) -> None:
        with open(self.meta_path, "w") as f:
            json.dump(meta, f, indent=2, default=str)

    def log_step(self, record: dict) -> None:
        self._steps_file.write(json.dumps(record, default=str) + "\n")
        self._steps_file.flush()

    def log_final(self, prompt: str, output_text: str, history: list) -> None:
        summary = {
            "prompt": prompt,
            "output": output_text,
            "num_steps": len(history),
            "num_fast_path": sum(1 for h in history if h["path"] == "A_fast_passthrough"),
            "num_fusion_path": sum(1 for h in history if h["path"] == "B_fusion_pipeline"),
            "num_escalations": sum(1 for h in history if h.get("escalated")),
        }
        with open(self.result_path, "w") as f:
            json.dump(summary, f, indent=2, default=str)

    def close(self) -> None:
        self._steps_file.close()
