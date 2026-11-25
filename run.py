"""Turntable video -> gaussian splat, driven from this machine through the Colab CLI.

  python run.py turntable.mp4                 # everything: VM up, reconstruct, download, VM down
  python run.py turntable.mp4 --stage train   # rerun one stage on a kept VM (--keep)
"""
import argparse, os, subprocess, sys, tempfile, time, tarfile, textwrap
from pathlib import Path

HERE = Path(__file__).resolve().parent
ENV = {**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"}


def colab(*a, check=True, capture=False, timeout=None):
    try:
        r = subprocess.run(["colab", *a], env=ENV, text=True, encoding="utf-8", errors="replace",
                           stdout=subprocess.PIPE if capture else None,
                           stderr=subprocess.STDOUT if capture else None, timeout=timeout)
    except subprocess.TimeoutExpired:  # the CLI's own --timeout does not cover a stalled websocket
        if check:
            raise SystemExit(f"colab {a[0]} stalled for {timeout}s")
        return ""
    if check and r.returncode:
        raise SystemExit(f"colab {' '.join(a)} failed\n{r.stdout or ''}")
    return r.stdout or ""


def py(session, code, timeout=300, check=True):
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8") as f:
        f.write(textwrap.dedent(code))
    try:
        return colab("exec", "-s", session, "-f", f.name, "--timeout", str(timeout), capture=True, check=check, timeout=timeout + 60)
    finally:
        os.unlink(f.name)


def remote(session, name, cmd, poll=15):
    """Run a shell command detached on the VM and tail its log, so a dropped websocket can't kill it."""
    py(session, f"""
        import subprocess, os
        os.makedirs('/content/gs/logs', exist_ok=True)
        L='/content/gs/logs/{name}.log'
        for p in (L, L+'.rc'):
            if os.path.exists(p): os.remove(p)
        subprocess.Popen(['bash','-c', "cd /content/gs && ( " + {cmd!r} + " ) > " + L + " 2>&1; echo $? > " + L + ".rc"], start_new_session=True)
    """)
    seen = misses = 0
    while True:
        out = py(session, f"""
            import os
            L='/content/gs/logs/{name}.log'
            b=open(L,'rb').read() if os.path.exists(L) else b''
            print('@@'+b[{seen}:].decode('utf-8','replace').replace(chr(13), chr(10)))
            print('@@RC='+(open(L+'.rc').read().strip() if os.path.exists(L+'.rc') else '')+'|'+str(len(b)))
        """, timeout=60, check=False)
        if "@@RC=" in out and "@@" in out:
            body = out.split("@@", 1)[1].rsplit("@@RC=", 1)
            txt = body[0].strip()
            if txt:
                print(textwrap.indent(txt, f"  [{name}] "), flush=True)
            rc, n = body[1].strip().splitlines()[0].split("|")
            seen = int(n)
            if rc:
                if rc != "0":
                    raise SystemExit(f"remote stage '{name}' failed (exit {rc})")
                return
            misses = 0
        else:
            misses += 1
            if misses >= 8:
                raise SystemExit(f"lost contact with VM during '{name}': {out[-2000:]}")
        time.sleep(poll)


def acquire(session, gpus, tries=6):
    """Colab returns 503 when a GPU type is out of stock, so cycle through the acceptable types."""
    if f"[{session}]" in colab("sessions", capture=True, check=False):
        print(f"reusing session {session}")
        return
    for k in range(tries):
        for g in gpus:
            print(f"requesting {g} ...", flush=True)
            colab("new", "-s", session, "--gpu", g, capture=True, check=False)
            if f"[{session}]" in colab("sessions", capture=True, check=False):
                print(f"got {g}")
                return
        time.sleep(45)
    raise SystemExit("no GPU available; try again later or pass --gpus")


def push(session, video):
    tgz = Path(tempfile.gettempdir()) / f"{session}_up.tgz"
    with tarfile.open(tgz, "w:gz") as t:
        for d in ("pipelines", "env"):
            t.add(HERE / d, d, filter=lambda i: None if "__pycache__" in i.name else i)
        t.add(video, "input" + Path(video).suffix.lower())
    py(session, "import os; os.makedirs('/content/gs', exist_ok=True)")
    colab("upload", "-s", session, str(tgz), "content/gs/up.tgz", capture=True)
    # LF endings matter for setup.sh when edited on Windows
    print(py(session, "import subprocess; print(subprocess.getoutput(\"cd /content/gs && tar xzf up.tgz && sed -i 's/\\r$//' env/*.sh && ls\"))").strip().splitlines()[-1])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video")
    ap.add_argument("--out", help="default: out/<video name>_<backend>")
    ap.add_argument("--backend", choices=["pi4", "colmap"], default="pi4", help="pose source; pi4 is non-commercial (CC BY-NC), colmap is permissively licensed and never installs Pi4")
    ap.add_argument("--stage", choices=["all", "matte", "pose", "train", "export"], default="all", help="start from this stage (VM must still hold earlier results)")
    ap.add_argument("--session", default="gstt")
    ap.add_argument("--gpus", default="A100,L4", help="acceptable Colab GPUs, in order of preference")
    ap.add_argument("--keep", action="store_true", help="leave the VM running (it bills until `colab stop -s SESSION`)")
    ap.add_argument("--train-args", default="", help='passed to 03_train.py, e.g. "--iters 15000 --cap 200000"')
    a = ap.parse_args()
    video = Path(a.video).resolve()
    out = Path(a.out or HERE / "out" / f"{video.stem}_{a.backend}").resolve()
    out.mkdir(parents=True, exist_ok=True)
    order = ["matte", "pose", "train", "export"]
    todo = order if a.stage == "all" else order[order.index(a.stage):]
    if todo != ["export"]:
        ok = False
        try:
            acquire(a.session, a.gpus.split(","))
            push(a.session, video)
            remote(a.session, "setup", f"BACKEND={a.backend} bash env/setup.sh")
            inp = "input" + video.suffix.lower()
            if "matte" in todo:
                remote(a.session, "matte", f"rm -rf work && envs/main/bin/python pipelines/turntable/01_matte.py {inp} work 2>&1 | grep --line-buffered -vi warn; exit ${{PIPESTATUS[0]}}")
            if "pose" in todo:
                pose = "envs/pi4/bin/python pipelines/turntable/02_pose_pi4.py" if a.backend == "pi4" else "envs/main/bin/python pipelines/turntable/02_pose_colmap.py"
                remote(a.session, "pose", f"{pose} work 2>&1 | grep --line-buffered -vi warn; exit ${{PIPESTATUS[0]}}")
            remote(a.session, "train", f"MPLBACKEND=agg envs/main/bin/python pipelines/turntable/03_train.py work result --pose pose_{a.backend}.npz {a.train_args} 2>&1 | grep --line-buffered -vi warn; exit ${{PIPESTATUS[0]}}")
            for f in ("splat.ply", "compare.mp4", "orbit.mp4", "metrics.json", "cameras.npz"):
                colab("download", "-s", a.session, f"content/gs/result/{f}", str(out / f), capture=True)
                if not (out / f).exists():
                    raise SystemExit(f"download of {f} failed")
            ok = True
        finally:
            if not a.keep:
                colab("stop", "-s", a.session, capture=True, check=False)
                print("VM released")
            elif not ok:
                print(f"VM kept; release it with: colab stop -s {a.session}")
    npx = "npx.cmd" if os.name == "nt" else "npx"
    st = [npx, "-y", "@playcanvas/splat-transform@3.12.0", "-w", "-q", str(out / "splat.ply"), "-N"]
    for ext in ("sog", "html"):  # compressed splat, and a self-contained viewer page
        if subprocess.run([*st, str(out / f"splat.{ext}")]).returncode:
            print(f"warning: splat.{ext} not written (splat-transform failed); splat.ply is fine")
    print("done ->", out)


if __name__ == "__main__":
    main()
