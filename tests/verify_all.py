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
import re
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
MAX_REPO_BYTES = 2 * 1024**3
MAX_WEIGHT_BYTES = 5 * 1024**2
TRAIN_DAYS = 5478  # 2005-01-01 through 2019-12-31 inclusive (15 years).


class VerificationSuite:
    def __init__(self, base_url: str, timeout: float = 20.0,
                 reference_dataset: Path | None = None) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.passed = 0
        self.failed = 0
        self.skipped = 0
        self.required_skips = 0
        self.warnings = 0
        self.reference_dataset = reference_dataset
        self.reference_day8: np.ndarray | None = None
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

    def warn(self, name: str, detail: str = "") -> None:
        print(f"[WARN] {name}" + (f" — {detail}" if detail else ""))
        self.warnings += 1

    def skip(self, name: str, detail: str = "", *, required: bool = False) -> None:
        suffix = " (required check not completed)" if required else ""
        print(f"[SKIP] {name}{suffix}" + (f" — {detail}" if detail else ""))
        self.skipped += 1
        if required:
            self.required_skips += 1

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

        required_paths = [
            "train/model.py", "train/train_monsoon.py",
            "weights/monsoon_unet.pt", "assets/static_maps.npy",
            "dashboard/index.html", "dashboard/style.css", "dashboard/app.js",
            "README.md", "R1_CONTRACT.md", "docs/IMD_THRESHOLDS.md",
            "docs/PROJECT_BRIEF.pdf",
        ]
        for relative in required_paths:
            self.result(f"required file exists: {relative}", (ROOT / relative).is_file())

        static_path = ROOT / "assets" / "static_maps.npy"
        if not static_path.is_file():
            self.result("static_maps.npy has shape [2,32,36] and finite values", False,
                        "assets/static_maps.npy is missing")
        else:
            try:
                static_maps = np.load(static_path, allow_pickle=False)
                shape_ok = static_maps.shape == (2, 32, 36)
                finite_ok = bool(np.isfinite(static_maps).all())
                land_ok = bool(np.min(static_maps[0]) >= -1e-4 and np.max(static_maps[0]) <= 1.0 + 1e-4) if shape_ok else False
                elevation_ok = bool(np.min(static_maps[1]) >= -500.0 and np.max(static_maps[1]) <= 6000.0) if shape_ok else False
                ok = shape_ok and finite_ok and land_ok and elevation_ok
                self.result("static_maps.npy has shape [2,32,36] and plausible finite maps", ok,
                            f"shape={static_maps.shape}; finite={finite_ok}; land_mask_range_ok={land_ok}; elevation_range_ok={elevation_ok}")
            except Exception as exc:
                self.result("static_maps.npy can be loaded", False, str(exc))

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
            weight_size = weights_path.stat().st_size
            self.result("model weights are below the 5 MiB team target",
                        weight_size < MAX_WEIGHT_BYTES,
                        f"size={weight_size / (1024 * 1024):.2f} MiB; limit strictly below 5 MiB")
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

    def check_repository_size(self) -> None:
        """Check the deliverable size, excluding local data/env/cache and Git internals."""
        ignored_dirs = {".git", ".venv", "venv", "env", "__pycache__", ".pytest_cache", "data"}
        total = 0
        count = 0
        for path in ROOT.rglob("*"):
            if not path.is_file():
                continue
            relative = path.relative_to(ROOT)
            if any(part in ignored_dirs for part in relative.parts):
                continue
            try:
                total += path.stat().st_size
                count += 1
            except OSError:
                continue
        self.result("deliverable working tree is strictly below 2 GB",
                    total < MAX_REPO_BYTES,
                    f"{total / (1024**3):.3f} GiB across {count} files (local data, environments and .git excluded)")

    def check_dashboard_offline(self) -> None:
        """Reject literal external URLs in the dashboard; localhost API calls are allowed."""
        self.section("3. DASHBOARD OFFLINE-SAFETY SCAN")
        dashboard = ROOT / "dashboard"
        if not dashboard.is_dir():
            self.result("dashboard directory exists", False, "dashboard/ is missing")
            return
        offenders: list[str] = []
        url_pattern = re.compile(r"https?://[^\s\"'<>`)]+", re.IGNORECASE)
        allowed_hosts = {"localhost", "127.0.0.1", "0.0.0.0", "::1"}
        for path in dashboard.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in {".html", ".js", ".css"}:
                continue
            try:
                content = path.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError) as exc:
                offenders.append(f"{path.relative_to(ROOT)}: cannot read ({exc})")
                continue
            for match in url_pattern.finditer(content):
                url = match.group(0).rstrip(".,;]")
                # Three.js uses this XML namespace in DOM/SVG creation. It is
                # an identifier, not a request to an external network service.
                if url == "http://www.w3.org/1999/xhtml":
                    continue
                host = urllib.parse.urlparse(url).hostname
                if host and host.lower() not in allowed_hosts:
                    line = content.count("\n", 0, match.start()) + 1
                    offenders.append(f"{path.relative_to(ROOT)}:{line} external URL {url}")
        self.result("dashboard contains no literal external network URLs", not offenders,
                    "; ".join(offenders[:8]) if offenders else "only local assets and localhost API references found")

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
            "huggingface_hub",
        }
        banned_exact = {"urllib.request", "urllib.error"}
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
        try:
            module_available = importlib.util.find_spec("monsoon_r1.schema_validator") is not None
        except (ImportError, ModuleNotFoundError, ValueError):
            module_available = False
        if not module_available:
            self.skip("official schema_validator", "Install/restore the official monsoon_r1 kit and rerun", required=True)
            return
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
        self.check_repository_size()
        self.tracked_data_archive_check()
        self.check_network_imports()
        self.check_dashboard_offline()

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
            self.result(f"{label}: wall-clock response is below 5.0 seconds",
                        wall_seconds < MAX_LATENCY_SECONDS,
                        f"wall time={wall_seconds:.3f}s")
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
        if rollout.shape == EXPECTED_OUTPUT_SHAPE and np.isfinite(rollout).all():
            min_tcwv = float(np.min(rollout[:, 5]))
            if min_tcwv < 0.0:
                self.warn("total column water vapour is physically non-negative",
                          f"observed minimum tcwv={min_tcwv:.4f} kg/m²; consider clamping or constraining channel 5")
            else:
                self.result("total column water vapour is non-negative", True,
                            f"minimum tcwv={min_tcwv:.4f} kg/m²")
        if wall is not None:
            self.result("baseline response completes", True, f"wall time={wall:.3f}s")
        self.check_anti_blur()

        # Detect the exact starter implementation, which copies the last input day
        # into every forecast step. This is a diagnostic beyond shape validation.
        expected_persistence = np.repeat(self.initial_states[-1][None, ...], 10, axis=0)
        is_persistence = (
            rollout.shape == EXPECTED_OUTPUT_SHAPE
            and np.allclose(rollout, expected_persistence, rtol=0.0, atol=1e-6)
        )
        self.result("forecast is not the exact repeat-yesterday placeholder", not is_persistence,
                    "all 10 days exactly repeat init_states[-1]" if is_persistence else "not an exact persistence copy")

    def check_anti_blur(self) -> None:
        """Compare Day-8 spatial variance with a real held-out Day-8 target."""
        self.section("5A. DAY-8 ANTI-BLUR CHECK")
        if self.baseline_rollout is None or self.reference_day8 is None:
            self.skip("Day-8 spatial variance ratio (0.40–2.50 per channel)",
                      "No real held-out reference found; supply data/india_climate_train.nc (not committed)",
                      required=True)
            return
        ratios: list[float] = []
        all_ok = True
        names = CHANNELS
        for channel, name in enumerate(names):
            predicted = self.baseline_rollout[7, channel]
            truth = self.reference_day8[channel]
            true_var = float(np.var(truth))
            pred_var = float(np.var(predicted))
            if not math.isfinite(true_var) or true_var <= 1e-12:
                all_ok = False
                self.result(f"anti-blur variance ratio for {name}", False,
                            f"reference spatial variance is invalid or too small: {true_var}")
                continue
            ratio = pred_var / true_var
            ratios.append(ratio)
            ok = 0.40 <= ratio <= 2.50
            all_ok = all_ok and ok
            self.result(f"anti-blur variance ratio for {name}", ok,
                        f"predicted/reference spatial variance={ratio:.3f}; allowed 0.40–2.50")
        self.result("all six Day-8 channels satisfy anti-blur variance limits",
                    all_ok and len(ratios) == 6,
                    "variance ratio checked against real held-out Day-8 fields")

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

    def check_input_history_usage(self) -> None:
        """Diagnostic: report whether the two older input days affect forecasts."""
        self.section("8. THREE-DAY INPUT HISTORY DIAGNOSTIC")
        if self.initial_states is None or self.baseline_rollout is None:
            self.skip("three-day history usage", "A valid baseline forecast is unavailable")
            return
        stats = json.loads((ROOT / "assets" / "stats.json").read_text(encoding="utf-8"))
        temp_std = float(stats["std"][0])
        for day_index in (0, 1):
            changed = self.initial_states.copy()
            changed[day_index, 0, 10:14, 18:22] += 0.5 * temp_std
            output, _ = self.call_simulate(changed, None,
                                           f"history day {day_index - 2} sensitivity request")
            if output is None or output.shape != EXPECTED_OUTPUT_SHAPE or not np.isfinite(output).all():
                self.warn(f"history day {day_index - 2} usage could not be measured",
                          "the modified-history request returned an invalid forecast")
                continue
            difference = float(np.max(np.abs(output - self.baseline_rollout)))
            if difference <= 1e-4:
                self.warn(f"forecast does not respond to history day {day_index - 2}",
                          "the current backend appears to use only init_states[-1]; notify Member 1")
            else:
                self.result(f"forecast responds to history day {day_index - 2}", True,
                            f"max forecast difference={difference:.6g}")

    @staticmethod
    def _distance_from_region(region_mask: np.ndarray) -> np.ndarray:
        rows, cols = np.where(region_mask)
        if len(rows) == 0:
            raise ValueError("shock region mask is empty")
        row_min, row_max = int(rows.min()), int(rows.max())
        col_min, col_max = int(cols.min()), int(cols.max())
        grid_rows, grid_cols = np.indices((32, 36))
        delta_rows = np.maximum(np.maximum(row_min - grid_rows, 0), grid_rows - row_max)
        delta_cols = np.maximum(np.maximum(col_min - grid_cols, 0), grid_cols - col_max)
        return np.sqrt(delta_rows.astype(np.float64) ** 2 + delta_cols.astype(np.float64) ** 2)

    @staticmethod
    def _radius_for_energy(distance: np.ndarray, energy: np.ndarray, fraction: float = 0.95) -> float:
        flat_energy = energy.reshape(-1)
        total = float(flat_energy.sum())
        if total <= 1e-20:
            return 0.0
        order = np.argsort(distance.reshape(-1))
        cumulative = np.cumsum(flat_energy[order])
        position = min(int(np.searchsorted(cumulative, fraction * total, side="left")), len(order) - 1)
        return float(distance.reshape(-1)[order[position]])

    def check_contract_shock_causality(self) -> None:
        """Exercise the two actual R1 shock definitions and spatial dynamics gates."""
        self.section("9. CONTRACT SHOCK LOCALISATION AND PROPAGATION")
        if (self.initial_states is None or self.baseline_rollout is None
                or self.baseline_rollout.shape != EXPECTED_OUTPUT_SHAPE):
            self.skip("contract shock causality", "A valid baseline forecast is unavailable")
            return

        latitudes = 6.5 + np.arange(32, dtype=np.float32)
        longitudes = 65.5 + np.arange(36, dtype=np.float32)
        bay_mask = ((latitudes[:, None] >= 8.0) & (latitudes[:, None] <= 20.0)
                    & (longitudes[None, :] >= 82.0) & (longitudes[None, :] <= 94.0))
        arabian_mask = ((latitudes[:, None] >= 8.0) & (latitudes[:, None] <= 20.0)
                        & (longitudes[None, :] >= 65.0) & (longitudes[None, :] <= 75.0))

        warm_ocean = np.zeros(EXPECTED_PERTURBATION_SHAPE, dtype=np.float32)
        warm_ocean[0, bay_mask] = 3.0
        warm_ocean[5, bay_mask] = 0.15 * self.initial_states[-1, 5, bay_mask]

        monsoon_stall = np.zeros(EXPECTED_PERTURBATION_SHAPE, dtype=np.float32)
        monsoon_stall[2, arabian_mask] = -0.40 * self.initial_states[-1, 2, arabian_mask]
        monsoon_stall[3, arabian_mask] = -0.40 * self.initial_states[-1, 3, arabian_mask]

        stats = json.loads((ROOT / "assets" / "stats.json").read_text(encoding="utf-8"))
        channel_std = np.asarray(stats["std"], dtype=np.float64).reshape(1, 6, 1, 1)

        for shock_name, perturbation, region in (
            ("warm_ocean", warm_ocean, bay_mask),
            ("monsoon_stall", monsoon_stall, arabian_mask),
        ):
            shocked, _ = self.call_simulate(self.initial_states, perturbation,
                                            f"{shock_name} contract shock request")
            if shocked is None or shocked.shape != EXPECTED_OUTPUT_SHAPE or not np.isfinite(shocked).all():
                self.result(f"{shock_name}: causal spatial diagnostics", False,
                            "shock forecast was invalid")
                continue

            response = (shocked.astype(np.float64) - self.baseline_rollout.astype(np.float64)) / channel_std
            max_response_std = float(np.max(np.abs(response)))
            self.result(f"{shock_name}: response stays within 5 channel standard deviations",
                        max_response_std <= 5.0,
                        f"maximum standardized response={max_response_std:.3f} std")

            energy_by_day = np.square(response).sum(axis=1)
            distance = self._distance_from_region(region)
            day1_total = float(energy_by_day[0].sum())
            if day1_total <= 1e-12:
                self.result(f"{shock_name}: non-zero Day-1 response", False,
                            "the shock produced no measurable Day-1 response")
                continue

            local_fraction = float(energy_by_day[0][distance <= 3.0].sum() / day1_total)
            self.result(f"{shock_name}: at least 60% of Day-1 response energy within 3 cells",
                        local_fraction >= 0.60,
                        f"local response energy={local_fraction:.1%}")

            day4_total = float(energy_by_day[3].sum())
            day8_total = float(energy_by_day[7].sum())
            day4_outer = (float(energy_by_day[3][distance > 3.0].sum() / day4_total)
                          if day4_total > 1e-12 else 0.0)
            day8_outer = (float(energy_by_day[7][distance > 3.0].sum() / day8_total)
                          if day8_total > 1e-12 else 0.0)
            spread_ok = day4_outer > 0.01 and day8_outer > 0.01
            self.result(f"{shock_name}: measurable outward response by Days 4 and 8",
                        spread_ok,
                        f"energy outside 3-cell buffer: Day 4={day4_outer:.1%}, Day 8={day8_outer:.1%}")

            r1 = self._radius_for_energy(distance, energy_by_day[0])
            r4 = self._radius_for_energy(distance, energy_by_day[3])
            r8 = self._radius_for_energy(distance, energy_by_day[7])
            growth_1_to_4 = max(0.0, r4 - r1)
            growth_4_to_8 = max(0.0, r8 - r4)
            speed_ok = growth_1_to_4 <= 8.0 * 3.0 and growth_4_to_8 <= 8.0 * 4.0
            self.result(f"{shock_name}: 95%-energy radius growth does not exceed 8 cells/day",
                        speed_ok,
                        f"r95 Day 1={r1:.1f}, Day 4={r4:.1f}, Day 8={r8:.1f}; growth={growth_1_to_4:.1f}/{growth_4_to_8:.1f} cells")

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
        """Measure the listening server's RSS with macOS/Linux-friendly fallbacks.

        psutil.net_connections() can raise AccessDenied on macOS because it scans
        system-wide sockets. `lsof` identifies only the requested listening port,
        and `ps` can report RSS for that PID without requiring root for the user's
        own process.
        """
        self.section("9. SERVER RAM CHECK")
        pids: list[int] = []

        # Preferred on macOS and commonly available on Linux.
        lsof_code, lsof_output = self.run_command(
            ["lsof", "-nP", "-t", f"-iTCP:{port}", "-sTCP:LISTEN"], timeout=10.0
        )
        if lsof_code == 0:
            for item in lsof_output.splitlines():
                try:
                    pids.append(int(item.strip()))
                except ValueError:
                    continue

        # Fallback for environments without lsof.
        if not pids:
            try:
                import psutil  # type: ignore
                for connection in psutil.net_connections(kind="inet"):
                    if (
                        connection.status == psutil.CONN_LISTEN
                        and connection.laddr
                        and connection.laddr.port == port
                        and connection.pid
                    ):
                        pids.append(int(connection.pid))
            except (ImportError, OSError):
                pass
            except Exception as exc:
                # Keep a useful diagnostic but still try the command-line fallback.
                print(f"[INFO] psutil socket lookup unavailable: {exc}")

        pids = sorted(set(pids))
        if not pids:
            self.skip(
                "server process RAM below 6 GB",
                f"Could not identify a listener on port {port}; install lsof or run the QA script with suitable process permissions",
                required=True,
            )
            return

        measurements: list[tuple[int, int]] = []
        for pid in pids:
            rss_bytes = None
            # `ps -o rss=` reports resident memory in KiB on macOS and Linux.
            ps_code, ps_output = self.run_command(
                ["ps", "-o", "rss=", "-p", str(pid)], timeout=10.0
            )
            if ps_code == 0:
                try:
                    rss_kib = int(ps_output.strip().splitlines()[-1])
                    if rss_kib > 0:
                        rss_bytes = rss_kib * 1024
                except (ValueError, IndexError):
                    rss_bytes = None

            if rss_bytes is None:
                try:
                    import psutil  # type: ignore
                    rss_bytes = int(psutil.Process(pid).memory_info().rss)
                except Exception:
                    rss_bytes = None

            if rss_bytes is not None:
                measurements.append((pid, rss_bytes))

        if not measurements:
            self.skip(
                "server process RAM below 6 GB",
                f"Found PID(s) {pids}, but neither ps nor psutil could read their RSS",
                required=True,
            )
            return

        maximum = max(rss for _, rss in measurements)
        detail = ", ".join(
            f"PID {pid}: {rss / (1024 ** 3):.2f} GB" for pid, rss in measurements
        )
        self.result(
            "server process RAM is below 6 GB",
            maximum < MAX_RAM_BYTES,
            detail,
        )

    def load_reference_validation_case(self) -> bool:
        """Load one chronological validation window for real Day-8 anti-blur checks."""
        if self.reference_dataset is None or not self.reference_dataset.is_file():
            self.skip("real held-out validation sample", 
                      f"dataset not found at {self.reference_dataset}; it must stay out of Git", required=True)
            return False
        try:
            import xarray as xr
            with xr.open_dataset(self.reference_dataset) as dataset:
                if "data" not in dataset or "time" not in dataset:
                    raise ValueError("NetCDF must contain 'data' and 'time' variables")
                data_var = dataset["data"]
                if data_var.ndim != 4 or tuple(data_var.shape[1:]) != (6, 32, 36):
                    raise ValueError(f"data must have shape [N,6,32,36], got {data_var.shape}")
                # Pick a chronological holdout window from the last two years.
                last_input_index = TRAIN_DAYS + 20
                if int(data_var.shape[0]) <= last_input_index + 8:
                    raise ValueError("dataset is too short for the selected Day-8 validation target")
                self.initial_states = np.asarray(
                    data_var.isel(time=slice(last_input_index - 2, last_input_index + 1)).values,
                    dtype=np.float32,
                )
                self.reference_day8 = np.asarray(
                    data_var.isel(time=last_input_index + 8).values,
                    dtype=np.float32,
                )
                doy_value = float(dataset["time"].dt.dayofyear.isel(time=last_input_index).values)
                self.doy_sincos = [math.sin(2 * math.pi * doy_value / 365.0),
                                   math.cos(2 * math.pi * doy_value / 365.0)]
            valid = (self.initial_states.shape == EXPECTED_INPUT_SHAPE
                     and self.reference_day8.shape == (6, 32, 36)
                     and np.isfinite(self.initial_states).all()
                     and np.isfinite(self.reference_day8).all())
            if not valid:
                raise ValueError("selected reference tensors have invalid shapes or non-finite values")
            self.result("real held-out validation sample loaded", True,
                        f"input shape={self.initial_states.shape}; Day-8 target shape={self.reference_day8.shape}")
            return True
        except Exception as exc:
            self.skip("real held-out validation sample", f"could not load dataset: {exc}", required=True)
            return False

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

        loaded_real_case = self.load_reference_validation_case()
        if not loaded_real_case and stats is not None:
            self.initial_states = self.make_initial_states(stats)

        self.check_baseline_output()
        self.check_zero_shock()
        self.check_extreme_shocks_and_sensitivity()
        self.check_contract_shock_causality()
        self.check_input_history_usage()
        self.check_latency()
        self.check_server_memory_if_possible(port)

        self.section("FINAL RESULT")
        print(f"PASS: {self.passed} | FAIL: {self.failed} | WARN: {self.warnings} | SKIP: {self.skipped}")
        if self.failed:
            print("OVERALL: FAIL — fix the failed checks before submission.")
            print("Note: Wi-Fi-off/offline rehearsal must still be performed manually.")
            return 1
        if self.required_skips:
            print("OVERALL: INCOMPLETE — required checks were skipped; the repository is not fully verified.")
            print("Restore the official monsoon_r1 validator and the local ignored training dataset, then rerun.")
            print("Reminder: perform one final clean-clone test with Wi-Fi disabled.")
            return 1
        print("OVERALL: PASS — all required executed checks passed.")
        if self.skipped:
            print("Some optional checks were skipped; review the SKIP messages above.")
        if self.warnings:
            print("Review WARN messages; they are diagnostics, not automatic failures.")
        print("Reminder: perform one final clean-clone test with Wi-Fi disabled.")
        return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the Multiverse of Monsoon QA suite.")
    parser.add_argument("--url", default="http://localhost:8000",
                        help="Base URL of the running serve.py API (default: http://localhost:8000)")
    parser.add_argument("--timeout", type=float, default=30.0,
                        help="HTTP timeout in seconds for each API request (default: 30)")
    parser.add_argument("--reference-dataset", default="data/india_climate_train.nc",
                        help="Path to local ignored NetCDF dataset for a real Day-8 anti-blur check")
    args = parser.parse_args()

    if args.timeout <= 0:
        parser.error("--timeout must be greater than zero")

    reference_path = Path(args.reference_dataset)
    if not reference_path.is_absolute():
        reference_path = ROOT / reference_path
    suite = VerificationSuite(args.url, timeout=args.timeout, reference_dataset=reference_path)
    try:
        return suite.run(port=urllib.parse.urlparse(args.url).port or 80)
    except KeyboardInterrupt:
        print("\n[FAIL] Verification interrupted by user.")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
