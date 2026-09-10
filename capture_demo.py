#!/usr/bin/env python3
"""capture_demo.py — screenshots (and an mp4) of the cross-field reader.

Drives out/bridges.html in headless Chrome through a real journey:
  pick two distant fields -> the work that bridges them -> the passages, with
  page numbers -> inside one of the theses.

Every frame is the real tool over the real corpus; nothing is mocked.
"""
from __future__ import annotations
import base64, json, shutil, subprocess, sys, tempfile, time, urllib.request
from pathlib import Path
from check_viewer import CDP, find_chrome, free_port

HERE = Path(__file__).resolve().parent
OUT = HERE / "out" / "demo"
W, H = 1600, 900
FIELD_A, FIELD_B = "Mathematics", "Arts and media"


SEQ = []


def shot(c, name, hold=1):
    """Capture one frame, repeated `hold` times to hold it in the video."""
    OUT.mkdir(parents=True, exist_ok=True)
    r = c.send("Page.captureScreenshot", format="png", captureBeyondViewport=False)
    data = base64.b64decode(r["data"])
    p = OUT / f"{len(SEQ):03d}-{name}.png"
    p.write_bytes(data)
    SEQ.extend([p] * hold)
    return p


def glide(c, name, to_js, steps=14, hold=1):
    """Scroll smoothly to a target, capturing each step, so the video moves."""
    y0 = c.js("window.scrollY")
    y1 = c.js(f"(function(){{var e={to_js}; if(!e) return window.scrollY;"
              f"var r=e.getBoundingClientRect();"
              f"return Math.max(0, window.scrollY + r.top - 90);}})()")
    for i in range(steps):
        t = (i + 1) / steps
        t = t * t * (3 - 2 * t)                     # ease in/out
        c.js(f"window.scrollTo(0, {y0} + ({y1} - {y0}) * {t})")
        shot(c, f"{name}-scroll")
    for _ in range(hold - 1):
        shot(c, f"{name}-hold")


def main():
    target = HERE / "out" / "bridges.html"
    port, prof = free_port(), tempfile.mkdtemp()
    proc = subprocess.Popen([find_chrome(), "--headless=new",
        f"--remote-debugging-port={port}", f"--user-data-dir={prof}",
        "--no-first-run", "--remote-allow-origins=*", "--disable-gpu",
        "--hide-scrollbars", f"--window-size={W},{H}", "about:blank"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        ws = None
        for _ in range(150):
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{port}/json/list",
                                            timeout=1) as r:
                    t = [x for x in json.load(r) if x.get("type") == "page"]
                if t:
                    ws = t[0]["webSocketDebuggerUrl"]; break
            except Exception:
                time.sleep(0.15)
        c = CDP(ws)
        c.send("Runtime.enable"); c.send("Page.enable")
        c.send("Emulation.setDeviceMetricsOverride", width=W, height=H,
               deviceScaleFactor=2, mobile=False)
        c.send("Page.navigate", url=target.as_uri())
        c.drain(9.0)

        shot(c, "open", hold=14)

        c.js(f"""(function(){{var a=document.getElementById('fa'),
          b=document.getElementById('fb');
          a.value={json.dumps(FIELD_A)}; b.value={json.dumps(FIELD_B)};
          a.dispatchEvent(new Event('change'));}})()""")
        time.sleep(1.2)
        shot(c, "pair", hold=20)

        c.js("""(function(){var h=document.querySelector('.bridge .bh');
          if(h) h.click();})()""")
        time.sleep(1.0)
        glide(c, "bridge", "document.querySelector('.bridge.open')", hold=12)

        glide(c, "passages", "document.querySelector('.bridge.open .side')", hold=18)
        glide(c, "more", "document.querySelectorAll('.bridge.open .th')[1]", hold=14)

        c.js("""(function(){var d=document.querySelector('.bridge.open .drill');
          if(d){ d.click(); }})()""")
        time.sleep(1.0)
        glide(c, "thesis", "document.querySelector('.doc')", hold=22)
    finally:
        proc.terminate()
        try: proc.wait(timeout=8)
        except Exception: proc.kill()
        shutil.rmtree(prof, ignore_errors=True)

    # assemble an mp4 at 2 fps (each still holds ~0.5s per duplicate)
    lst = OUT / "frames.txt"
    lst.write_text("".join(f"file '{p.name}'\nduration 0.1\n" for p in SEQ)
                   + f"file '{SEQ[-1].name}'\n")
    mp4 = OUT / "thesisgraph-demo.mp4"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat",
                    "-safe", "0", "-i", str(lst), "-vf",
                    "scale=1600:-2:flags=lanczos,format=yuv420p",
                    "-r", "30", "-c:v", "libx264", "-preset", "slow",
                    "-crf", "20", "-movflags", "+faststart", str(mp4)],
                   check=False)
    if mp4.exists():
        print(f"\n  video: {mp4}  ({mp4.stat().st_size/1e6:.1f} MB, "
              f"{len(SEQ)*0.1:.0f}s, {len(SEQ)} frames)")
    print(f"  stills: {OUT}")


if __name__ == "__main__":
    sys.exit(main())
