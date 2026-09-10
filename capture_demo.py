#!/usr/bin/env python3
"""capture_demo.py — screenshots (and an mp4) of the thesis corpus graph.

Drives out/public/index.html in headless Chrome through a real journey:
  the whole map -> one discipline and what it shares -> split it into
  sub-fields -> split a distant one -> the works that bridge the two.

Interaction is driven with dispatched mouse events at real node coordinates,
not by calling the app's internals, so the video exercises the same code path
a visitor does. Every frame is the real tool over the real corpus.
"""
from __future__ import annotations
import base64, json, shutil, subprocess, sys, tempfile, time, urllib.request
from pathlib import Path
from check_viewer import CDP, find_chrome, free_port

HERE = Path(__file__).resolve().parent
OUT = HERE / "out" / "demo"
W, H = 1600, 1000
FIELD_A, FIELD_B = "Computing", "Language and literature"

SEQ = []


def shot(c, name, hold=1):
    """Capture one frame, repeated `hold` times to hold it in the video."""
    OUT.mkdir(parents=True, exist_ok=True)
    r = c.send("Page.captureScreenshot", format="png", captureBeyondViewport=False)
    p = OUT / f"{len(SEQ):03d}-{name}.png"
    p.write_bytes(base64.b64decode(r["data"]))
    SEQ.extend([p] * hold)
    return p


# The app's state lives inside an IIFE, so there is no N/tx/ty to read. Node
# positions are recovered the way a user finds them: sweep the pointer across
# the canvas and record where each label's tooltip appears. The 9px step is
# below the app's own 10px pick radius, so every drawn node is hit.
_SWEEP = """(function(){
  var c=document.getElementById('cv'), rc=c.getBoundingClientRect();
  var tip=document.getElementById('tip'), out={};
  for(var y=6;y<rc.height;y+=9){ for(var x=6;x<rc.width;x+=9){
    c.dispatchEvent(new MouseEvent('mousemove',
      {clientX:rc.left+x, clientY:rc.top+y, bubbles:true}));
    if(tip.style.display!=='none'){
      var b=tip.querySelector('b');
      if(b && !(b.textContent in out))
        out[b.textContent]={x:rc.left+x, y:rc.top+y}; } } }
  c.dispatchEvent(new MouseEvent('mousemove',
    {clientX:rc.left+2, clientY:rc.top+2, bubbles:true}));   // clear the tooltip
  return out; })()"""


def node_map(c):
    """{label: {x, y}} for every node currently drawn, found by pointer sweep."""
    return c.js(_SWEEP)


def node_xy(nodes, label):
    xy = nodes.get(label)
    return dict(xy) if xy else None


def hover(c, xy):
    c.send("Input.dispatchMouseEvent", type="mouseMoved", x=xy["x"], y=xy["y"],
           buttons=0)


def click(c, xy):
    for t in ("mousePressed", "mouseReleased"):
        c.send("Input.dispatchMouseEvent", type=t, x=xy["x"], y=xy["y"],
               button="left", clickCount=1, buttons=1 if t == "mousePressed" else 0)
        time.sleep(0.05)


def zoom(c, name, steps=6, into=True, hold=1):
    """Zoom with real wheel events — one frame per notch, so it eases visibly.

    The app maps one notch to x1.12 in, x0.893 out; nothing here reaches past
    the page's own handler.
    """
    for _ in range(steps):
        c.js("""(function(){var c=document.getElementById('cv');
          var rc=c.getBoundingClientRect();
          c.dispatchEvent(new WheelEvent('wheel',{deltaY:%d,cancelable:true,
            bubbles:true,clientX:rc.left+rc.width/2,clientY:rc.top+rc.height/2}));
          })()""" % (-1 if into else 1))
        shot(c, f"{name}-zoom")
    for _ in range(hold - 1):
        shot(c, f"{name}-hold")


def main():
    target = HERE / "out" / "public" / "index.html"
    if not target.exists():
        sys.exit(f"no {target} — run: make_graph_app.py --public")
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
        c.drain(10.0)

        # 1. the whole map: every discipline, linked by the literature it shares
        shot(c, "map", hold=24)

        # 2. one discipline — the tooltip says what clicking it will do
        nodes = node_map(c)
        a = node_xy(nodes, FIELD_A)
        print(f"  {FIELD_A}: {a and (round(a['x']), round(a['y']))}")
        hover(c, a); time.sleep(0.4); shot(c, "hover", hold=10)

        # 3. select it: the panel lists what it links to, ⇄ marking cross-field
        click(c, a); time.sleep(0.6); shot(c, "select", hold=22)

        # 4. click again to split it into its sub-fields
        click(c, a); time.sleep(0.9); shot(c, "split", hold=22)

        # 5. and split a genuinely distant discipline, so both are open at once
        b = node_xy(node_map(c), FIELD_B)
        print(f"  {FIELD_B}: {b and (round(b['x']), round(b['y']))}")
        click(c, b); time.sleep(0.5)
        click(c, b); time.sleep(0.9); shot(c, "split2", hold=24)

        # 6. lean in on the links that cross between them
        zoom(c, "in", steps=7, into=True, hold=16)
        zoom(c, "out", steps=7, into=False, hold=4)

        # 7. the works that actually bridge disciplines
        c.js("""(function(){var t=document.querySelectorAll('.tab');
          for(var i=0;i<t.length;i++) if(t[i].dataset.v==='bridges'){t[i].click();return;}})()""")
        time.sleep(1.2); shot(c, "bridges", hold=24)

        # 8. one bridging work: which fields, and which theses cite it
        # the most-linked work on screen: the widest label the sweep found
        bmap = node_map(c)
        top = None
        if bmap:
            lbl = sorted(bmap, key=lambda k: (-len(k), k))[0]
            top = dict(bmap[lbl]); top["label"] = lbl
        print(f"  bridge: {top and top['label']}")
        if top:
            hover(c, top); time.sleep(0.3)
            click(c, top); time.sleep(0.8); shot(c, "work", hold=28)
    finally:
        proc.terminate()
        try: proc.wait(timeout=8)
        except Exception: proc.kill()
        shutil.rmtree(prof, ignore_errors=True)

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
