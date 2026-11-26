#!/usr/bin/env python3
"""Round-1 QA suite for the Multiverse of Monsoon repository.

Run from the repository root:
    python tests/verify_all.py --url http://localhost:8000

This file is intentionally self-contained and only writes to stdout. It does not
modify serve.py, train/, dashboard/, docs/, model weights, or repository files.
"""
from __future__ import annotations

import argparse
import ast
import importlib.util
import json
import math
import os
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

try:
    import numpy as np
except ImportError:  # Give a clear error instead of a long traceback.
    print("[FAIL] NumPy is required. Install it with: python -m pip install numpy")
    raise SystemExit(1)

ROOT = Path(__file__).resolve().parents[1]
CHANNELS = ["t2m", "tp", "u10", "v10", "msl", "tcwv"]
EXPECTED_INPUT_SHAPE = (3, 6, 32, 36)
EXPECTED_OUTPUT_SHAPE = (10, 6, 32, 36)
EXPECTED_PERTURBATION_SHAPE = (6, 32, 36)
MAX_LATENCY_SECONDS = 5.0
MAX_RAM_BYTES = 6 * 1024**3


class VerificationSuite:
    def __init__(self, base_url: str, timeout: float = 20.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.passed = 0
        self.failed = 0
        self.skipped = 0
        self.initial_states: np.ndarray | None = None
        self.doy_sincos = [math.sin(2 * math.pi * 180 / 365.0),
                           math.cos(2 * math.pi * 180 / 365.0)]
        self.baseline_rollout: np.ndarray | None = None
        self.baseline_inference_ms: float | None = None

    def result(self, name: str, ok: bool, detail: str = "") -> None:
        status = "PASS" if ok else "FAIL"
        print(f"[{status}] {name}" + (f" — {detail}" if detail else ""))
        if ok:
            self.passed += 1
        else:
            self.failed += 1

    def skip(self, name: str, detail: str = "") -> None:
        print(f"[SKIP] {name}" + (f" — {detail}" if detail else ""))
        self.skipped += 1

    def section(self, title: str) -> None:
        print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")

    def run_command(self, command: list[str], timeout: float = 180.0) -> tuple[int, str]:
        try:
            completed = subprocess.run(
                command,
                cwd=ROOT,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
            output = (completed.stdout or "") + (completed.stderr or "")
            return completed.returncode, output.strip()
        except subprocess.TimeoutExpired as exc:
            output = (exc.stdout or "") + (exc.stderr or "")
            if isinstance(output, bytes):
                output = output.decode("utf-8", errors="replace")
            return 124, f"Command timed out after {timeout:.0f}s. {output}".strip()
        except OSError as exc:
            return 127, f"Could not start command: {exc}"

    # ------------------------------------------------------------------
    # Repository and static checks
    # ------------------------------------------------------------------
    def check_required_paths_and_stats(self) -> dict[str, Any] | None:
        self.section("1. REPOSITORY FILES AND NORMALISATION STATS")

        serve_path = ROOT / "serve.py"
        self.result("serve.py exists at repository root", serve_path.is_file(), str(serve_path))

        stats_path = ROOT / "assets" / "stats.json"
        if not stats_path.is_file():
            self.result("assets/stats.json exists", False, "Required stats file is missing")
            return None

        try:
            stats = json.loads(stats_path.read_text(encoding="utf-8"))
            if not isinstance(stats, dict):
                raise ValueError("stats.json must contain a JSON object")
            channels = stats.get("channels")
            means = np.asarray(stats.get("mean"), dtype=np.float64)
            stds = np.asarray(stats.get("std"), dtype=np.float64)
            valid = (
                channels == CHANNELS
                and means.shape == (6,)
                and stds.shape == (6,)
                and np.isfinite(means).all()
                and np.isfinite(stds).all()
                and (stds > 0).all()
            )
            self.result(
                "stats.json has six expected channels and valid mean/std arrays",
                bool(valid),
                f"channels={channels}; mean shape={means.shape}; std shape={stds.shape}",
            )
            if not valid:
                return None
            return stats
        except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
            self.result("stats.json can be parsed", False, str(exc))
            return None

    def check_weights(self) -> None:
        weights_path = ROOT / "weights" / "monsoon_unet.pt"
        if not weights_path.is_file():
            self.result("trained weights exist", False, "Missing weights/monsoon_unet.pt")
            return

        try:
            if weights_path.stat().st_size == 0:
                self.result("weights file is non-empty", False, "File size is zero")
                return
            head = weights_path.open("rb").read(4096)
            is_lfs_pointer = head.startswith(b"version https://git-lfs.github.com/spec/v1")
            # PyTorch .pt files are binary (commonly ZIP/PK or pickle). A readable
            # Git LFS pointer or other plain-text placeholder is not a real weight file.
            non_text = sum(byte not in (9, 10, 13) and not (32 <= byte <= 126) for byte in head)
            is_binary = non_text > 0
            self.result("weights file is not a Git LFS pointer", not is_lfs_pointer,
                        f"size={weights_path.stat().st_size} bytes")
            self.result("weights/monsoon_unet.pt is a binary file", is_binary and not is_lfs_pointer,
                        f"binary-content bytes in sample={non_text}")
        except OSError as exc:
            self.result("weights file can be read", False, str(exc))

    def tracked_data_archive_check(self) -> None:
        """Catch obvious accidental commits of the large/raw climate dataset."""
        code, output = self.run_command(["git", "ls-files", "-z"], timeout=20)
        if code != 0:
            self.skip("tracked climate-data archive scan", "Not a Git checkout or git is unavailable")
            return

        tracked = [item for item in output.split("\x00") if item]
        forbidden: list[str] = []
        for item in tracked:
            path = Path(item)
            lower_name = path.name.lower()
            suffix = path.suffix.lower()
            if lower_name == "india_climate_train.nc" or suffix in {".grib", ".grb", ".grib2"}:
                forbidden.append(item)
            elif ("2023" in lower_name or "2024" in lower_name) and suffix in {
                ".nc", ".grib", ".grb", ".grib2", ".csv", ".parquet", ".npz"
            }:
                forbidden.append(item)
        self.result(
            "no obvious raw/historical climate archives are tracked by Git",
            not forbidden,
            ", ".join(forbidden[:5]) if forbidden else f"checked {len(tracked)} tracked paths",
        )

    @staticmethod
    def _module_name_from_import(node: ast.AST) -> list[str]:
        names: list[str] = []
        if isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.append(node.module)
        return names

    def check_network_imports(self) -> None:
        """Scan application/training Python, not this QA script or the official tools."""
        excluded_parts = {
            ".git", ".venv", "venv", "env", "__pycache__", "tests", "monsoon_r1",
            "site-packages", "dashboard",
        }
        banned_roots = {
            "requests", "httpx", "aiohttp", "urllib3", "socket", "ftplib",
            "paramiko", "boto3", "botocore", "selenium", "playwright", "curl_cffi",
        }
        banned_exact = {"urllib.request"}
        offenders: list[str] = []

        for path in ROOT.rglob("*.py"):
            relative = path.relative_to(ROOT)
            if any(part in excluded_parts for part in relative.parts):
                continue
            try:
                source = path.read_text(encoding="utf-8")
                tree = ast.parse(source, filename=str(relative))
            except (OSError, UnicodeDecodeError, SyntaxError) as exc:
                offenders.append(f"{relative}: cannot inspect Python source ({exc})")
                continue

            for node in ast.walk(tree):
                for module in self._module_name_from_import(node):
                    root_name = module.split(".", 1)[0]
                    if root_name in banned_roots or module in banned_exact:
                        offenders.append(f"{relative}: import {module}")
                # Catch common dynamic imports too, e.g. __import__('requests').
                if isinstance(node, ast.Call) and node.args:
                    func_name = ""
                    if isinstance(node.func, ast.Name):
                        func_name = node.func.id
                    if func_name in {"__import__", "import_module"}:
                        first = node.args[0]
                        if isinstance(first, ast.Constant) and isinstance(first.value, str):
                            module = first.value
                            if module.split(".", 1)[0] in banned_roots or module in banned_exact:
                                offenders.append(f"{relative}: dynamic import {module}")

        self.result(
            "no prohibited network-client imports in application/training Python",
            not offenders,
            "; ".join(offenders[:8]) if offenders else "application Python sources inspected",
        )

    def check_optional_repo_scan(self) -> None:
        self.section("2. OFFICIAL REPOSITORY SCAN")
        try:
            module_available = importlib.util.find_spec("monsoon_r1.repo_scan") is not None
        except (ImportError, ModuleNotFoundError, ValueError):
            module_available = False

        if not module_available:
            self.skip("monsoon_r1.repo_scan", "Module is not installed/available in this environment")
            return

        code, output = self.run_command([sys.executable, "-m", "monsoon_r1.repo_scan", "."])
        self.result("python -m monsoon_r1.repo_scan . exits with code 0", code == 0,
                    f"exit={code}; {output[-1200:]}")

    def check_official_validator(self) -> None:
        self.section("2. OFFICIAL API CONTRACT VALIDATOR")
        command = [sys.executable, "-m", "monsoon_r1.schema_validator", self.base_url]
        code, output = self.run_command(command, timeout=180.0)
        self.result(
            "official schema_validator exits with code 0",
            code == 0,
            f"exit={code}; {output[-1600:] or 'no validator output'}",
        )

    def run_static_checks(self) -> None:
        self.section("3. STATIC ANTI-DISQUALIFICATION CHECKS")
        self.check_weights()
        self.tracked_data_archive_check()
        self.check_network_imports()

    # ------------------------------------------------------------------
    # API request helpers and data generation
    # ------------------------------------------------------------------
    def post_json(self, endpoint: str, payload: dict[str, Any] | None = None) -> tuple[int, dict[str, Any], float]:
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        body = b"" if payload is None else json.dumps(payload, allow_nan=False).encode("utf-8")
        request = urllib.request.Request(
            url,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        start = time.perf_counter()
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                raw = response.read()
                elapsed = time.perf_counter() - start
                if response.geturl().rstrip("/") != url.rstrip("/"):
                    raise RuntimeError(f"Unexpected redirect: {url} -> {response.geturl()}")
                if len(raw) > 80 * 1024 * 1024:
                    raise RuntimeError(f"Response is too large: {len(raw)} bytes (limit 80 MiB)")
                parsed = json.loads(raw.decode("utf-8"))
                if not isinstance(parsed, dict):
                    raise ValueError("Response JSON must be an object")
                return response.status, parsed, elapsed
        except urllib.error.HTTPError as exc:
            elapsed = time.perf_counter() - start
            detail = exc.read().decode("utf-8", errors="replace")[:500]
            raise RuntimeError(f"HTTP {exc.code} from {url}: {detail}") from exc
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise RuntimeError(f"Request to {url} failed: {exc}") from exc

    def make_initial_states(self, stats: dict[str, Any]) -> np.ndarray:
        means = np.asarray(stats["mean"], dtype=np.float32)
        stds = np.asarray(stats["std"], dtype=np.float32)
        rows = np.linspace(-1.0, 1.0, 32, dtype=np.float32)[:, None]
        cols = np.linspace(-1.0, 1.0, 36, dtype=np.float32)[None, :]
        spatial_pattern = 0.55 * np.sin(np.pi * rows) + 0.45 * np.cos(np.pi * cols)
        states = np.empty(EXPECTED_INPUT_SHAPE, dtype=np.float32)

        for day in range(3):
            for channel in range(6):
                temporal_shift = (day - 1) * 0.025 * stds[channel]
                field = means[channel] + stds[channel] * (0.06 * spatial_pattern) + temporal_shift
                if channel == 1:  # Precipitation must be physically non-negative.
                    field = np.maximum(field, 0.2)
                states[day, channel] = field
        return states

    def make_payload(self, states: np.ndarray, perturbation: np.ndarray | None) -> dict[str, Any]:
        return {
            "init_states": np.asarray(states, dtype=np.float32).tolist(),
            "doy_sincos": self.doy_sincos,
            "steps": 10,
            "perturbation": None if perturbation is None else np.asarray(perturbation, dtype=np.float32).tolist(),
        }

    def call_simulate(self, states: np.ndarray, perturbation: np.ndarray | None,
                      label: str) -> tuple[np.ndarray | None, float | None]:
        try:
            status, body, wall_seconds = self.post_json("/simulate", self.make_payload(states, perturbation))
            if status != 200:
                self.result(label, False, f"HTTP status {status}")
                return None, None
            if "rollout" not in body:
                self.result(label, False, "Response is missing 'rollout'")
                return None, None
            rollout = np.asarray(body["rollout"], dtype=np.float32)
            inference_ms_raw = body.get("inference_ms")
            try:
                inference_ms = float(inference_ms_raw)
                inference_valid = math.isfinite(inference_ms) and inference_ms >= 0.0
            except (TypeError, ValueError):
                inference_ms = None
                inference_valid = False
            self.result(f"{label}: inference_ms is a finite non-negative number",
                        inference_valid, f"inference_ms={inference_ms_raw!r}")
            return rollout, wall_seconds
        except (RuntimeError, ValueError, TypeError) as exc:
            self.result(label, False, str(exc))
            return None, None

    @staticmethod
    def output_invariant_errors(rollout: np.ndarray) -> list[str]:
        errors: list[str] = []
        if rollout.shape != EXPECTED_OUTPUT_SHAPE:
            errors.append(f"shape {rollout.shape}, expected {EXPECTED_OUTPUT_SHAPE}")
            return errors
        if not np.isfinite(rollout).all():
            errors.append("contains NaN or Inf")
            return errors
        min_rain = float(np.min(rollout[:, 1]))
        if min_rain < 0.0:
            errors.append(f"negative rainfall detected (minimum {min_rain:.6g} mm/day)")
        min_temp_c = float(np.min(rollout[:, 0]) - 273.15)
        max_temp_c = float(np.max(rollout[:, 0]) - 273.15)
        if min_temp_c < -40.0 or max_temp_c > 60.0:
            errors.append(f"temperature outside -40..60 C ({min_temp_c:.2f}..{max_temp_c:.2f} C)")
        return errors

    def check_output_invariants(self, name: str, rollout: np.ndarray | None) -> bool:
        if rollout is None:
            return False
        errors = self.output_invariant_errors(rollout)
        self.result(name, not errors, "; ".join(errors) if errors else "shape, finite values, rain and temperature valid")
        return not errors

    # ------------------------------------------------------------------
    # API/physics checks
    # ------------------------------------------------------------------
    def check_health(self) -> None:
        self.section("4. API HEALTH")
        try:
            status, body, _ = self.post_json("/health")
            valid = status == 200 and body.get("status") == "ready" and bool(body.get("model_name"))
            self.result("POST /health reports ready with a model_name", valid,
                        f"HTTP {status}; body={body}")
            model_name = str(body.get("model_name", ""))
            if "placeholder-persistence" in model_name:
                self.result("server is not still using placeholder-persistence", False,
                            "Member 1 must replace the starter persistence model")
        except RuntimeError as exc:
            self.result("POST /health is reachable", False, str(exc))

    def check_baseline_output(self) -> None:
        self.section("5. BASELINE SIMULATION AND PHYSICAL INVARIANTS")
        if self.initial_states is None:
            self.skip("baseline simulation", "Valid stats.json was not available")
            return
        rollout, wall = self.call_simulate(self.initial_states, None, "baseline /simulate request")
        if rollout is None:
            return
        if rollout.shape == EXPECTED_OUTPUT_SHAPE and np.isfinite(rollout).all():
            self.baseline_rollout = rollout
        self.check_output_invariants("baseline output invariants", rollout)
        if wall is not None:
            self.result("baseline response completes", True, f"wall time={wall:.3f}s")

        # Detect the exact starter implementation, which copies the last input day
        # into every forecast step. This is a diagnostic beyond shape validation.
        expected_persistence = np.repeat(self.initial_states[-1][None, ...], 10, axis=0)
        is_persistence = (
            rollout.shape == EXPECTED_OUTPUT_SHAPE
            and np.allclose(rollout, expected_persistence, rtol=0.0, atol=1e-6)
        )
        self.result("forecast is not the exact repeat-yesterday placeholder", not is_persistence,
                    "all 10 days exactly repeat init_states[-1]" if is_persistence else "not an exact persistence copy")

    def check_zero_shock(self) -> None:
        self.section("6. ZERO-SHOCK BIT-EQUALITY")
        if self.initial_states is None:
            self.skip("zero-shock test", "Valid stats.json was not available")
            return
        no_shock, _ = self.call_simulate(self.initial_states, None, "null perturbation request")
        zero_pert = np.zeros(EXPECTED_PERTURBATION_SHAPE, dtype=np.float32)
        zero_shock, _ = self.call_simulate(self.initial_states, zero_pert, "zero-array perturbation request")
        if no_shock is None or zero_shock is None:
            self.result("null and all-zero shocks are bit-identical", False, "One or both requests failed")
            return
        if no_shock.shape != EXPECTED_OUTPUT_SHAPE or zero_shock.shape != EXPECTED_OUTPUT_SHAPE:
            self.result("null and all-zero shocks are bit-identical", False,
                        f"unexpected shapes: {no_shock.shape}, {zero_shock.shape}")
            return
        max_diff = float(np.max(np.abs(no_shock - zero_shock)))
        self.result("null and all-zero shocks are bit-identical", max_diff == 0.0,
                    f"max absolute difference={max_diff:.10g}")

    def check_extreme_shocks_and_sensitivity(self) -> None:
        self.section("7. EXTREME SHOCKS AND CAUSAL SENSITIVITY")
        if (self.initial_states is None or self.baseline_rollout is None
                or self.baseline_rollout.shape != EXPECTED_OUTPUT_SHAPE
                or not np.isfinite(self.baseline_rollout).all()):
            self.skip("extreme-shock tests", "A valid baseline forecast is unavailable")
            return

        # Approximate Bay of Bengal region on the 32x36 grid (south-to-north,
        # west-to-east). A local patch exercises a geographically bounded shock.
        row_slice = slice(2, 14)
        col_slice = slice(17, 30)
        positive = np.zeros(EXPECTED_PERTURBATION_SHAPE, dtype=np.float32)
        positive[0, row_slice, col_slice] = 5.0       # t2m +5 K
        positive[2, row_slice, col_slice] = 10.0      # u10 +10 m/s
        positive[3, row_slice, col_slice] = 10.0      # v10 +10 m/s

        negative = np.zeros(EXPECTED_PERTURBATION_SHAPE, dtype=np.float32)
        negative[0, row_slice, col_slice] = -5.0     # t2m -5 K
        negative[2, row_slice, col_slice] = -10.0    # u10 -10 m/s
        negative[3, row_slice, col_slice] = -10.0    # v10 -10 m/s
        negative[1, row_slice, col_slice] = -20.0    # tp -20 mm/day

        shock_differences: list[float] = []
        for label, perturbation in (("positive extreme shock", positive),
                                    ("negative extreme shock", negative)):
            rollout, _ = self.call_simulate(self.initial_states, perturbation, f"{label} request")
            if rollout is None:
                continue
            valid = self.check_output_invariants(f"{label} physical invariants", rollout)
            if rollout.shape == EXPECTED_OUTPUT_SHAPE and np.isfinite(rollout).all():
                diff = float(np.max(np.abs(rollout - self.baseline_rollout)))
                shock_differences.append(diff)
                self.result(f"{label} changes the forecast", diff > 1e-4,
                            f"max trajectory difference={diff:.6g}")

        # Perturb one cell of the latest initial temperature field, independently
        # of the perturbation argument, to ensure the server uses the supplied state.
        changed_state = self.initial_states.copy()
        changed_state[-1, 0, 10:14, 18:22] += 0.5
        changed_rollout, _ = self.call_simulate(changed_state, None, "altered initial-state request")
        if changed_rollout is not None and changed_rollout.shape == EXPECTED_OUTPUT_SHAPE and np.isfinite(changed_rollout).all():
            state_diff = float(np.max(np.abs(changed_rollout - self.baseline_rollout)))
            self.result("changing initial_state changes the forecast", state_diff > 1e-4,
                        f"max trajectory difference={state_diff:.6g}")
        else:
            self.result("changing initial_state changes the forecast", False, "altered-state response invalid")

        if shock_differences:
            max_shock_diff = max(shock_differences)
            self.result("at least one shock has non-zero causal sensitivity",
                        max_shock_diff > 1e-4,
                        f"largest shock trajectory difference={max_shock_diff:.6g}")
        else:
            self.result("at least one shock has non-zero causal sensitivity", False,
                        "No valid extreme-shock response was available")

    # ------------------------------------------------------------------
    # Performance and optional memory checks
    # ------------------------------------------------------------------
    def check_latency(self) -> None:
        self.section("8. CPU LATENCY BENCHMARK")
        if self.initial_states is None:
            self.skip("five-run latency benchmark", "Valid stats.json was not available")
            return

        times: list[float] = []
        all_ok = True
        for run_number in range(1, 6):
            rollout, wall = self.call_simulate(
                self.initial_states, None, f"latency run {run_number}/5 request"
            )
            if rollout is None or wall is None:
                all_ok = False
                continue
            times.append(wall)
            invariant_errors = self.output_invariant_errors(rollout)
            if invariant_errors:
                all_ok = False
                print(f"       run {run_number} output issue: {'; '.join(invariant_errors)}")
            print(f"       run {run_number}: {wall:.3f}s")

        if len(times) != 5:
            self.result("five consecutive 10-day requests completed", False,
                        f"only {len(times)} of 5 requests completed")
            return

        mean_time = float(np.mean(times))
        max_time = float(np.max(times))
        self.result("five consecutive 10-day requests return valid outputs", all_ok,
                    f"completed={len(times)}/5")
        self.result("mean wall-clock latency is strictly below 5.0s",
                    mean_time < MAX_LATENCY_SECONDS, f"mean={mean_time:.3f}s")
        self.result("maximum wall-clock latency is strictly below 5.0s",
                    max_time < MAX_LATENCY_SECONDS, f"max={max_time:.3f}s")

    def check_server_memory_if_possible(self, port: int) -> None:
        self.section("9. OPTIONAL SERVER RAM CHECK")
        try:
            import psutil  # type: ignore
        except ImportError:
            self.skip("server process RAM below 6 GB", "Install psutil for this optional check: python -m pip install psutil")
            return

        try:
            listeners = []
            for connection in psutil.net_connections(kind="inet"):
                if connection.status == psutil.CONN_LISTEN and connection.laddr and connection.laddr.port == port and connection.pid:
                    listeners.append(connection.pid)
            if not listeners:
                self.skip("server process RAM below 6 GB", "Could not identify the process listening on the requested port")
                return
            measurements = []
            for pid in sorted(set(listeners)):
                process = psutil.Process(pid)
                rss = process.memory_info().rss
                measurements.append((pid, rss))
            maximum = max(rss for _, rss in measurements)
            detail = ", ".join(f"PID {pid}: {rss / 1024**3:.2f} GB" for pid, rss in measurements)
            self.result("server process RAM is below 6 GB", maximum < MAX_RAM_BYTES, detail)
        except (psutil.AccessDenied, psutil.NoSuchProcess, OSError) as exc:
            self.skip("server process RAM below 6 GB", f"OS did not allow process inspection: {exc}")

    # ------------------------------------------------------------------
    # Main execution
    # ------------------------------------------------------------------
    def run(self, port: int) -> int:
        print("MULTIVERSE OF MONSOON — ROUND 1 QA VERIFICATION")
        print(f"Repository: {ROOT}")
        print(f"Server URL: {self.base_url}")

        stats = self.check_required_paths_and_stats()
        self.check_optional_repo_scan()
        self.check_official_validator()
        self.run_static_checks()
        self.check_health()

        if stats is not None:
            self.initial_states = self.make_initial_states(stats)

        self.check_baseline_output()
        self.check_zero_shock()
        self.check_extreme_shocks_and_sensitivity()
        self.check_latency()
        self.check_server_memory_if_possible(port)

        self.section("FINAL RESULT")
        print(f"PASS: {self.passed} | FAIL: {self.failed} | SKIP: {self.skipped}")
        if self.failed:
            print("OVERALL: FAIL — fix the failed checks before submission.")
            print("Note: Wi-Fi-off/offline rehearsal must still be performed manually.")
            return 1
        print("OVERALL: PASS — all executed checks passed.")
        if self.skipped:
            print("Some optional checks were skipped; review the SKIP messages above.")
        print("Reminder: perform one final clean-clone test with Wi-Fi disabled.")
        return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the Multiverse of Monsoon QA suite.")
    parser.add_argument("--url", default="http://localhost:8000",
                        help="Base URL of the running serve.py API (default: http://localhost:8000)")
    parser.add_argument("--timeout", type=float, default=30.0,
                        help="HTTP timeout in seconds for each API request (default: 30)")
    args = parser.parse_args()

    if args.timeout <= 0:
        parser.error("--timeout must be greater than zero")

    suite = VerificationSuite(args.url, timeout=args.timeout)
    try:
        return suite.run(port=urllib.parse.urlparse(args.url).port or 80)
    except KeyboardInterrupt:
        print("\n[FAIL] Verification interrupted by user.")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
