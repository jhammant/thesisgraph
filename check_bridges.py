#!/usr/bin/env python3
"""check_bridges.py — verify the cross-field reader actually works."""
import json, shutil, subprocess, sys, tempfile, time, urllib.request
from pathlib import Path
from check_viewer import CDP, find_chrome, free_port
HERE = Path(__file__).resolve().parent

def main():
    target = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else HERE/"out"/"bridges.html"
    port, prof = free_port(), tempfile.mkdtemp()
    proc = subprocess.Popen([find_chrome(),"--headless=new",f"--remote-debugging-port={port}",
        f"--user-data-dir={prof}","--no-first-run","--remote-allow-origins=*",
        "--disable-gpu","--window-size=1500,1000","about:blank"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    res=[]
    def check(c,w): res.append((bool(c),w)); print(f"  {'PASS' if c else 'FAIL'}  {w}")
    try:
        ws=None
        for _ in range(120):
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{port}/json/list",timeout=1) as r:
                    t=[x for x in json.load(r) if x.get("type")=="page"]
                if t: ws=t[0]["webSocketDebuggerUrl"]; break
            except Exception: time.sleep(0.15)
        c=CDP(ws)
        for d in ("Runtime","Log","Page","Network"): c.send(d+".enable")
        c.send("Page.navigate", url=target.as_uri()); c.drain(9.0)
        errs=[]
        for e in c.events:
            m=e.get("method")
            if m=="Log.entryAdded" and e["params"]["entry"].get("level") in ("error","warning"):
                errs.append(e["params"]["entry"].get("text"))
            elif m=="Runtime.exceptionThrown":
                errs.append(str(e["params"]["exceptionDetails"].get("text"))[:120])
        check(not errs, f"zero console errors (saw {errs[:2]})")
        reqs=[e["params"]["request"]["url"] for e in c.events
              if e.get("method")=="Network.requestWillBeSent"]
        check(all(u.startswith("file:") for u in reqs), "no network requests on load")
        n=c.js("document.querySelectorAll('.bridge').length")
        check(n>50, f"bridges rendered ({n})")
        opts=c.js("document.querySelectorAll('#fa option').length")
        check(opts>5, f"field selector populated ({opts-1} fields)")
        # filter to a specific distant pairing
        c.js("""(function(){var a=document.getElementById('fa'),b=document.getElementById('fb');
          a.value='Biological sciences'; b.value='Psychology';
          a.dispatchEvent(new Event('change'));})()""")
        time.sleep(0.4)
        f=c.js("document.querySelectorAll('.bridge').length")
        check(0<f<n, f"pair filter narrows the list ({n} -> {f} for Biology x Psychology)")
        lbl=c.js("""(function(){var e=document.querySelector('.bridge .w');
          return e?e.textContent:'';})()""")
        check(bool(lbl), f"first bridge for that pair: {lbl!r}")
        # expand and read a passage
        c.js("""document.querySelector('.bridge .bh').click()""")
        time.sleep(0.4)
        opened=c.js("document.querySelectorAll('.bridge.open').length")
        pg=c.js("document.querySelectorAll('.bridge.open .pg').length")
        q=c.js("""(function(){var e=document.querySelector('.bridge.open .pg q');
          return e?e.textContent.slice(0,80):'';})()""")
        check(opened==1 and pg>0, f"clicking a work opens its passages ({pg} passages)")
        check(len(q)>25, f"passage text present: {q[:60]!r}")
        deep=c.js("""(function(){var a=document.querySelector('.bridge.open .loc a');
          return a?a.getAttribute('href'):'';})()""")
        check("#page=" in (deep or ""), f"deep link goes to the exact PDF page: {(deep or '')[-46:]}")
        sides=c.js("document.querySelectorAll('.bridge.open .side').length")
        check(sides==2, f"both sides of the bridge shown ({sides})")
        # search
        c.js("""(function(){var q=document.getElementById('q');q.value='zzzznope';
          q.dispatchEvent(new Event('input'));})()""")
        time.sleep(0.3)
        check(c.js("document.querySelectorAll('.bridge').length")==0 and
              c.js("document.querySelectorAll('.empty').length")==1,
              "search with no hits shows the empty state")
        check(c.js("document.body.scrollWidth <= window.innerWidth + 2"),
              "no horizontal overflow")
        # drill-down into the thesis document itself
        c.js("""(function(){var q=document.getElementById('q');q.value='';
          q.dispatchEvent(new Event('input'));})()""")
        time.sleep(0.5)
        c.js("""document.querySelector('.bridge .bh').click()""")
        time.sleep(0.5)
        has = c.js("document.querySelectorAll('.drill').length")
        check(has > 0, f"thesis drill-down control present ({has})")
        c.js("""document.querySelector('.bridge.open .drill').click()""")
        time.sleep(0.5)
        toc = c.js("document.querySelectorAll('.doc .tocrow').length")
        c.js("""(function(){var n=0;document.querySelectorAll('.bridge.open .drill')
          .forEach(function(a){ if(n++<12) a.click(); });})()""")
        time.sleep(0.7)
        ab  = c.js("""(function(){var m=0;document.querySelectorAll('.doc .ab')
          .forEach(function(e){ if(e.textContent.length>m) m=e.textContent.length; });
          return m;})()""")
        dl  = c.js("""(function(){var a=document.querySelector('.doc .tocrow a');
          return a?a.getAttribute('href'):'';})()""")
        check(toc > 3, f"contents list rendered for the thesis ({toc} sections)")
        check(ab > 60, f"abstract shown where the metadata has one ({ab} chars)")
        check("#page=" in (dl or ""), f"contents entry deep-links into the PDF: {(dl or '')[-40:]}")
        rec = c.js("document.querySelectorAll('.doc .lk2 a').length")
        check(rec >= 1, f"repository / full-PDF links present ({rec})")
    finally:
        proc.terminate()
        try: proc.wait(timeout=8)
        except Exception: proc.kill()
        shutil.rmtree(prof, ignore_errors=True)
    bad=[w for ok,w in res if not ok]
    print("-"*70)
    print(f"BRIDGE READER {'FAILED' if bad else 'PASSED'} — {len(res)-len(bad)}/{len(res)}")
    for w in bad: print("  -",w)
    return 1 if bad else 0

if __name__=="__main__": sys.exit(main())
