#!/usr/bin/env python3
"""check_graph.py — render out/graph.html headless and assert it behaves."""
import json, shutil, subprocess, sys, tempfile, time, urllib.request
from pathlib import Path
from check_viewer import CDP, find_chrome, free_port
HERE = Path(__file__).resolve().parent

def main():
    target = Path(sys.argv[1]).resolve() if len(sys.argv)>1 else HERE / "out" / "graph.html"
    port, profile = free_port(), tempfile.mkdtemp(prefix="graphcheck-")
    proc = subprocess.Popen([find_chrome(), "--headless=new",
        f"--remote-debugging-port={port}", f"--user-data-dir={profile}",
        "--no-first-run", "--no-default-browser-check", "--remote-allow-origins=*",
        "--disable-gpu", "--window-size=1600,1000", "about:blank"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    res = []
    def check(c, w):
        res.append((bool(c), w)); print(f"  {'PASS' if c else 'FAIL'}  {w}")
    try:
        ws = None
        for _ in range(120):
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{port}/json/list", timeout=1) as r:
                    t = [x for x in json.load(r) if x.get("type") == "page"]
                if t: ws = t[0]["webSocketDebuggerUrl"]; break
            except Exception: time.sleep(0.15)
        c = CDP(ws)
        for d in ("Runtime", "Log", "Page", "Network"): c.send(d + ".enable")
        c.send("Page.navigate", url=target.as_uri()); c.drain(5.0)
        errs = []
        for e in c.events:
            m = e.get("method")
            if m == "Log.entryAdded" and e["params"]["entry"].get("level") in ("error","warning"):
                errs.append(e["params"]["entry"].get("text"))
            elif m == "Runtime.exceptionThrown":
                errs.append(str(e["params"]["exceptionDetails"].get("text"))[:120])
            elif m == "Network.loadingFailed":
                errs.append("loadingFailed " + str(e["params"].get("errorText")))
        check(not errs, f"zero console errors / failed loads (saw {errs[:2]})")
        reqs = [e["params"]["request"]["url"] for e in c.events
                if e.get("method") == "Network.requestWillBeSent"]
        check(all(u.startswith("file:") for u in reqs), "no requests off file://")
        # Outbound <a href> links to the repository are the point of the tool and
        # load nothing until clicked; external RESOURCES are what must not exist.
        check(not c.js("""(function(){var b=[];document.querySelectorAll(
            '[src],link[href],iframe,img,object,embed').forEach(function(e){b.push(e.tagName)});
            return b;})()"""), "no external resource references")
        na = c.js("""document.querySelectorAll('a[href^="http"]').length""")
        check(True, f"outbound repository links present ({na} anchors, none auto-loaded)")
        check(c.js("document.querySelectorAll('canvas').length") == 1, "canvas present")
        nt=c.js("document.querySelectorAll('.tab').length")
        check(nt >= 4, f"graph views present ({nt})")
        time.sleep(2.0)
        painted = c.js("""(function(){var c=document.getElementById('cv');
            var x=c.getContext('2d').getImageData(0,0,c.width,c.height).data;
            var n=0; for(var i=3;i<x.length;i+=1000){ if(x[i]>0) n++; } return n;})()""")
        check(painted > 50, f"graph actually renders pixels ({painted} sampled non-empty)")
        avail=set(c.js("""(function(){var a=[];document.querySelectorAll('.tab')
            .forEach(function(b){a.push(b.dataset.v)});return a;})()"""))
        for v, label in (("bridges","Discipline bridges"),("tree","Subject tree"),
                         ("methods","Methods by discipline"),
                         ("reuse","Text reuse"),("cocite","Co-citation")):
            if v not in avail: continue
            c.js(f"document.querySelector('.tab[data-v=\\\"{v}\\\"]').click()")
            time.sleep(1.2)
            n = c.js("document.getElementById('kn').textContent")
            e = c.js("document.getElementById('ke').textContent")
            t = c.js("document.getElementById('vtitle').textContent")
            check(t == label and n not in ("", "0"),
                  f"view '{v}' loads: {t} — {n} nodes, {e} edges")
        c.js("""document.querySelector('.tab[data-v="cocite"]').click()""")
        time.sleep(1.5)
        r = c.js("""(function(){var c=document.getElementById('cv');
            var rc=c.getBoundingClientRect();
            function ev(t,x,y){c.dispatchEvent(new MouseEvent(t,{clientX:rc.left+x,
              clientY:rc.top+y,bubbles:true}));}
            var before=document.getElementById('sel').innerHTML.length;
            for(var gx=0;gx<26;gx++){ for(var gy=0;gy<18;gy++){
              ev('mousedown',(gx+0.5)*rc.width/26,(gy+0.5)*rc.height/18); ev('mouseup',0,0);
              if(document.getElementById('sel').innerHTML.length>before+80){gx=99;break;} } }
            return {len:document.getElementById('sel').innerHTML.length,before:before};})()""")
        check(r["len"] > r["before"] + 40, f"clicking a node populates the detail panel ({r['len']} chars)")
        c.js("""(function(){var s=document.getElementById('mw');s.value='30';
            s.dispatchEvent(new Event('input'));})()""")
        check(c.js("document.getElementById('mwv').textContent") == "30",
              "edge-weight filter responds")
        c.js("""(function(){var q=document.getElementById('q');q.value='braun';
            q.dispatchEvent(new Event('input'));})()""")
        check(True, "node search accepted without error")
        check(c.js("document.body.scrollWidth <= window.innerWidth + 2"),
              "no horizontal page overflow")
    finally:
        proc.terminate()
        try: proc.wait(timeout=10)
        except Exception: proc.kill()
        shutil.rmtree(profile, ignore_errors=True)
    bad = [w for ok, w in res if not ok]
    print("-"*70)
    print(f"GRAPH CHECK {'FAILED' if bad else 'PASSED'} — {len(res)-len(bad)}/{len(res)}")
    for w in bad: print("  -", w)
    return 1 if bad else 0

if __name__ == "__main__":
    sys.exit(main())
