"""One-command runner for Financial Fraud Intelligence Checker.

Does everything end to end (Windows, macOS, Linux — stdlib only):

  1. installs Python deps (unless --skip-install)
  2. generates + scores sample data if data/outputs is empty (unless
     --skip-pipeline); pass --full for the full run instead of --sample
  3. starts the Python backend (python -m src.serve)
  4. starts the React+TypeScript frontend with `npm run dev`
     (or `npm run build` + backend-served dist/ with --build)

Usage:
    python run.py                  # sample data + backend :8000 + frontend :5173
    python run.py --live            # also enable live scoring (/ingest, /rings/refresh)
    python run.py --full            # full pipeline instead of --sample
    python run.py --build           # production: build dist/, serve from backend only
    python run.py --backend-only    # skip the frontend
    python run.py --frontend-only   # skip the backend (it must already run)
    python run.py --skip-pipeline --skip-install   # fastest restart, fail if outputs missing

Needs Node 18+ on PATH for the frontend; without it the backend still
starts and serves frontend/dist/ if built, else the legacy ui/ page.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import threading
import time
import urllib.request
import webbrowser
from pathlib import Path
from typing import NoReturn

ROOT = Path(__file__).resolve().parent
FRONTEND = ROOT / "frontend"
DEFAULT_OUTPUTS = ROOT / "data" / "outputs"
NEED = "transactions_scored.csv"


def die(msg: str, code: int = 1) -> NoReturn:
    print(f"[run] ERROR: {msg}", file=sys.stderr)
    raise SystemExit(code)


def run_foreground(cmd, cwd, env=None) -> None:
    """Run a setup step with inherited stdio; die on failure."""
    print(f"[run] $ {' '.join(map(str, cmd))}")
    try:
        subprocess.run(list(map(str, cmd)), cwd=cwd, env=env, check=True)
    except FileNotFoundError:
        die(f"command not found: {cmd[0]}")
    except subprocess.CalledProcessError as e:
        die(f"command failed (exit {e.returncode}): {' '.join(map(str, cmd))}")


def npm() -> str:
    """npm executable name for this OS (npm.cmd on Windows)."""
    if os.name == "nt":
        for cand in ("npm.cmd", "npm"):
            if shutil.which(cand):
                return cand
        return "npm.cmd"
    return "npm"


def ensure_python_deps(skip_install: bool) -> None:
    try:
        import pandas  # noqa: F401
        import yaml  # noqa: F401
    except ImportError:
        if skip_install:
            die("Python deps missing (pandas/pyyaml). Run pip install -r requirements.txt")
        print("[run] installing Python deps ...")
        run_foreground([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], ROOT)


def ensure_outputs(outputs: Path, full: bool, skip_pipeline: bool) -> None:
    if (outputs / NEED).exists():
        print(f"[run] outputs present in {outputs}")
        return
    if skip_pipeline:
        die(f"{outputs / NEED} missing. Run without --skip-pipeline or: python run_all.py --sample")
    print("[run] no scored outputs — running pipeline on sample data ..." if not full
          else "[run] no scored outputs — running FULL pipeline ...")
    cmd = [sys.executable, "run_all.py"] if full else [sys.executable, "run_all.py", "--sample"]
    run_foreground(cmd, ROOT)


def ensure_frontend(skip_install: bool, build: bool) -> str:
    """Prepare the frontend. Returns 'dev', 'dist', or 'legacy'."""
    if not (FRONTEND / "package.json").exists():
        print("[run] WARNING: frontend/ missing — backend serves legacy ui/")
        return "legacy"
    if shutil.which("node") is None or shutil.which(npm()) is None:
        if (FRONTEND / "dist" / "index.html").exists():
            print("[run] WARNING: node/npm not found — serving prebuilt frontend/dist/")
            return "dist"
        print("[run] WARNING: node/npm not found — serving legacy ui/ "
              "(install Node 18+ for the React UI)")
        return "legacy"
    if not (FRONTEND / "node_modules").is_dir():
        if skip_install:
            die("frontend/node_modules missing. Run without --skip-install or: cd frontend && npm install")
        print("[run] installing frontend deps (first run takes a while) ...")
        run_foreground([npm(), "install"], FRONTEND)
    if build:
        print("[run] building frontend/dist/ ...")
        run_foreground([npm(), "run", "build"], FRONTEND)
        return "dist"
    return "dev"


def spawn(cmd, cwd, env, name: str) -> "subprocess.Popen[str]":
    print(f"[run] $ {' '.join(map(str, cmd))}  (cwd={Path(cwd).name})")
    try:
        proc = subprocess.Popen(
            list(map(str, cmd)), cwd=cwd, env=env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1,
        )
    except FileNotFoundError:
        die(f"command not found: {cmd[0]}")
    threading.Thread(target=_pump, args=(proc, name), daemon=True).start()
    return proc


def _pump(proc: "subprocess.Popen[str]", name: str) -> None:
    assert proc.stdout is not None
    for line in proc.stdout:
        print(f"[{name}] {line}", end="")
    proc.stdout.close()


def wait_healthy(url: str, timeout: float) -> None:
    print(f"[run] waiting for backend {url} ...")
    start = time.monotonic()
    while time.monotonic() - start < timeout:
        try:
            with urllib.request.urlopen(url, timeout=5) as r:
                if r.status == 200:
                    print("[run] backend is up")
                    return
        except Exception:
            pass
        time.sleep(1.0)
    die(f"backend not healthy at {url} after {timeout:.0f}s (see [backend] log above)")


def open_browser(url: str) -> None:
    try:
        webbrowser.open(url)
    except Exception as e:  # headless machines etc.
        print(f"[run] (could not open browser: {e})")


def main() -> None:
    sys.stdout.reconfigure(line_buffering=True)  # live logs when stdout is piped
    p = argparse.ArgumentParser(description="Run backend + React frontend with one command")
    p.add_argument("--port", type=int, default=8000, help="backend port (default 8000)")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--frontend-port", type=int, default=5173, help="vite dev port (dev mode)")
    p.add_argument("--outputs", default=str(DEFAULT_OUTPUTS), help="outputs dir for --serve")
    p.add_argument("--live", action="store_true", help="enable live scoring endpoints")
    p.add_argument("--full", action="store_true", help="full pipeline instead of --sample")
    p.add_argument("--skip-pipeline", action="store_true")
    p.add_argument("--skip-install", action="store_true")
    p.add_argument("--build", action="store_true",
                   help="prod mode: npm run build, backend serves dist/, no vite dev server")
    p.add_argument("--backend-only", action="store_true", help="do not start the frontend")
    p.add_argument("--frontend-only", action="store_true", help="do not start the backend")
    p.add_argument("--no-open", action="store_true", help="do not open a browser tab")
    args = p.parse_args()

    if args.backend_only and args.frontend_only:
        die("pick at most one of --backend-only / --frontend-only")
    outputs = Path(args.outputs)
    backend_url = f"http://{args.host}:{args.port}"
    frontend_url = f"http://127.0.0.1:{args.frontend_port}"

    procs: list[tuple[str, subprocess.Popen]] = []

    if not args.frontend_only:
        ensure_python_deps(args.skip_install)
        ensure_outputs(outputs, args.full, args.skip_pipeline)
        mode = "dist" if args.build else "legacy"
        if not args.backend_only:
            mode = ensure_frontend(args.skip_install, args.build)
        elif args.build and (FRONTEND / "package.json").exists() and shutil.which("node"):
            # --backend-only --build: still produce dist/ so the backend serves React.
            mode = ensure_frontend(args.skip_install, True)
        cmd = [sys.executable, "-u", "-m", "src.serve",
               "--outputs", str(outputs), "--host", args.host, "--port", str(args.port)]
        if args.live:
            cmd.append("--live")
        procs.append(("backend", spawn(cmd, ROOT, None, "backend")))
        wait_healthy(f"{backend_url}/v1/health", timeout=600.0 if args.live else 60.0)
    else:
        mode = ensure_frontend(args.skip_install, args.build)
        try:
            with urllib.request.urlopen(f"{backend_url}/v1/health", timeout=5) as r:
                ok = r.status == 200
        except Exception:
            ok = False
        if not ok:
            print(f"[run] WARNING: no backend at {backend_url} — the dev proxy needs it running")

    if not args.backend_only and not args.frontend_only and mode == "dev":
        env = dict(os.environ, BACKEND_URL=backend_url)
        vite = [npm(), "run", "dev", "--", "--port", str(args.frontend_port), "--strictPort"]
        procs.append(("frontend", spawn(vite, FRONTEND, env, "frontend")))
        time.sleep(2.0)  # let vite print its URL first
    elif not args.backend_only and not args.frontend_only and mode == "dist":
        print("[run] prod mode: React served by the backend, no vite dev server")

    main_url = backend_url + "/" if (args.backend_only or mode != "dev") else frontend_url + "/"
    print(f"\n[run] ready → {main_url}")
    if not args.backend_only and mode == "dev":
        print(f"[run] backend API → {backend_url}/v1/health")
    print(f"[run] legacy UI → {backend_url}/legacy")
    print("[run] Ctrl+C to stop\n")
    if not args.no_open:
        open_browser(main_url)

    try:
        while True:
            for name, proc in list(procs):
                if proc.poll() is not None:
                    die(f"{name} exited with code {proc.returncode} (see [{name}] log above)")
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\n[run] stopping ...")
        for name, proc in procs:
            if proc.poll() is None:
                proc.terminate()
        deadline = time.monotonic() + 10
        for _, proc in procs:
            try:
                proc.wait(timeout=max(0, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                proc.kill()
        print("[run] stopped")


if __name__ == "__main__":
    main()
