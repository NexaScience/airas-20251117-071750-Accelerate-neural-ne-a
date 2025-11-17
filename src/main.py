import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import List

import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
CONFIG_DIR = SCRIPT_DIR.parent / "config"


class Tee:
    """Tee helper duplicating a stream into a log file while echoing to console."""

    def __init__(self, log_path: Path, stream):
        self.file = log_path.open("w")
        self.stream = stream

    def write(self, data):
        self.file.write(data)
        self.stream.write(data)
        self.file.flush()
        self.stream.flush()

    def flush(self):
        self.file.flush()
        self.stream.flush()


def run_subprocess(cmd: List[str], cwd: Path, stdout_path: Path, stderr_path: Path):
    with stdout_path.open("w") as out_f, stderr_path.open("w") as err_f:
        proc = subprocess.Popen(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        # Stream output live
        for line in proc.stdout:
            sys.stdout.write(line)
            out_f.write(line)
            out_f.flush()
        for line in proc.stderr:
            sys.stderr.write(line)
            err_f.write(line)
            err_f.flush()
        proc.wait()
        if proc.returncode != 0:
            raise RuntimeError(f"Subprocess '{' '.join(cmd)}' exited with {proc.returncode}")


def load_config(path: Path):
    return yaml.safe_load(path.read_text())


def main():
    parser = argparse.ArgumentParser(description="Experiment orchestrator – runs all variations & evaluation.")
    parser.add_argument("--smoke-test", action="store_true", help="Run variations defined in smoke_test.yaml")
    parser.add_argument("--full-experiment", action="store_true", help="Run variations in full_experiment.yaml")
    parser.add_argument("--results-dir", required=True, help="Directory to store all outputs")
    args = parser.parse_args()

    if args.smoke_test == args.full_experiment:
        parser.error("Exactly one of --smoke-test or --full-experiment must be supplied.")

    cfg_file = CONFIG_DIR / ("smoke_test.yaml" if args.smoke_test else "full_experiment.yaml")
    cfg = load_config(cfg_file)
    experiments = cfg.get("experiments", [])
    results_root = Path(args.results_dir)
    results_root.mkdir(parents=True, exist_ok=True)

    for exp in experiments:
        run_id = exp["run_id"]
        run_dir = results_root / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        # Write individual run config
        run_cfg_path = run_dir / "config.yaml"
        with run_cfg_path.open("w") as f:
            yaml.dump(exp, f)

        print("Launching run_id =", run_id)
        cmd = [sys.executable, "-m", "src.train", "--config", str(run_cfg_path), "--results-dir", str(run_dir)]
        stdout_log = run_dir / "stdout.log"
        stderr_log = run_dir / "stderr.log"
        run_subprocess(cmd, cwd=SCRIPT_DIR.parent, stdout_path=stdout_log, stderr_path=stderr_log)

    # After all runs – evaluation
    print("All runs finished.  Initiating evaluation…")
    eval_cmd = [sys.executable, "-m", "src.evaluate", "--results-dir", str(results_root)]
    run_subprocess(eval_cmd, cwd=SCRIPT_DIR.parent, stdout_path=results_root / "evaluate_stdout.log", stderr_path=results_root / "evaluate_stderr.log")


if __name__ == "__main__":
    main()
