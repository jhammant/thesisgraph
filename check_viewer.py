#!/usr/bin/env python3
"""check_viewer.py — render out/viewer.html from file:// in headless Chromium
and assert it behaves: zero console errors, zero external references, working
filters, and click-to-scroll cross-linking between the two panes.

Paths anchor to this file's location, not the shell's working directory.

    python check_viewer.py [path/to/viewer.html]
"""

from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

import websocket

HERE = Path(__file__).resolve().parent
CHROME_CANDIDATES = [
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/Applications/Chromium.app/Contents/MacOS/Chromium",
    shutil.which("chromium") or "",
    shutil.which("google-chrome") or "",
]


def find_chrome() -> str:
    for c in CHROME_CANDIDATES:
        if c and Path(c).exists():
            return c
    raise SystemExit("no Chrome/Chromium binary found")


def free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


class CDP:
    def __init__(self, ws_url: str):
        self.ws = websocket.create_connection(ws_url, timeout=30)
        self.n = 0
        self.events: list[dict] = []

    def send(self, method: str, **params):
        self.n += 1
        self.ws.send(json.dumps({"id": self.n, "method": method,
                                 "params": params}))
        while True:
            msg = json.loads(self.ws.recv())
            if msg.get("id") == self.n:
                if "error" in msg:
                    raise RuntimeError(f"{method}: {msg['error']}")
                return msg.get("result", {})
            self.events.append(msg)

    def drain(self, seconds: float):
        end = time.time() + seconds
        self.ws.settimeout(0.4)
        while time.time() < end:
            try:
                self.events.append(json.loads(self.ws.recv()))
            except Exception:
                pass
        self.ws.settimeout(30)

    def js(self, expr: str):
        r = self.send("Runtime.evaluate", expression=expr, returnByValue=True,
                      awaitPromise=True)
        if "exceptionDetails" in r:
            raise RuntimeError(f"JS threw: {r['exceptionDetails']}")
        return r["result"].get("value")


def main() -> int:
    target = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 \
        else HERE / "out" / "viewer.html"
    if not target.exists():
        raise SystemExit(f"missing {target}")

    port = free_port()
    profile = tempfile.mkdtemp(prefix="viewercheck-")
    proc = subprocess.Popen(
        [find_chrome(), "--headless=new", f"--remote-debugging-port={port}",
         f"--user-data-dir={profile}", "--no-first-run", "--no-default-browser-check",
         "--remote-allow-origins=*", "--disable-gpu", "--window-size=1600,1000",
         "about:blank"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    results: list[tuple[bool, str]] = []

    def check(cond: bool, what: str):
        results.append((bool(cond), what))
        print(f"  {'PASS' if cond else 'FAIL'}  {what}")

    try:
        ws_url = None
        for _ in range(100):
            try:
                with urllib.request.urlopen(
                        f"http://127.0.0.1:{port}/json/list", timeout=1) as r:
                    tabs = json.load(r)
                page = [t for t in tabs if t.get("type") == "page"]
                if page:
                    ws_url = page[0]["webSocketDebuggerUrl"]
                    break
            except Exception:
                time.sleep(0.15)
        if not ws_url:
            raise SystemExit("could not attach to headless Chrome")

        c = CDP(ws_url)
        c.send("Runtime.enable")
        c.send("Log.enable")
        c.send("Page.enable")
        c.send("Network.enable")
        c.send("Page.navigate", url=target.as_uri())
        c.drain(4.0)

        # ---- console errors / exceptions / failed loads
        errs = []
        for e in c.events:
            m = e.get("method")
            if m == "Log.entryAdded":
                entry = e["params"]["entry"]
                if entry.get("level") in ("error", "warning"):
                    errs.append(f"{entry['level']}: {entry.get('text')}")
            elif m == "Runtime.exceptionThrown":
                errs.append("exception: " + json.dumps(
                    e["params"]["exceptionDetails"].get("text", ""))[:200])
            elif m == "Runtime.consoleAPICalled":
                if e["params"].get("type") in ("error", "warning"):
                    errs.append("console." + e["params"]["type"])
            elif m == "Network.loadingFailed":
                errs.append("loadingFailed: " + str(e["params"].get("errorText")))
        check(not errs, f"zero console errors/warnings and no failed loads "
                        f"(saw {len(errs)}: {errs[:3]})")

        # ---- every network request must be the file itself
        reqs = [e["params"]["request"]["url"] for e in c.events
                if e.get("method") == "Network.requestWillBeSent"]
        external = [u for u in reqs if not u.startswith("file:")]
        check(not external, f"no network requests off file:// (saw {external[:3]})")

        # ---- no external references in the DOM at all
        ext = c.js("""(function(){
          var bad=[];
          document.querySelectorAll('[src]').forEach(function(e){bad.push('src:'+e.getAttribute('src'))});
          document.querySelectorAll('link[href]').forEach(function(e){bad.push('link:'+e.getAttribute('href'))});
          document.querySelectorAll('a[href^="http"]').forEach(function(e){bad.push('a:'+e.getAttribute('href'))});
          document.querySelectorAll('iframe,object,embed,img,video,audio').forEach(function(e){bad.push(e.tagName)});
          return bad;
        })()""")
        check(not ext, f"no external resource references in the DOM (saw {ext[:3]})")

        # ---- structure
        panes = c.js("document.querySelectorAll('.pane').length")
        check(panes == 2, f"two document panes are present (got {panes})")
        total = c.js("document.querySelectorAll('mark').length")
        check(total > 0, f"matched runs are highlighted (got {total} <mark> spans)")
        sideA = c.js("document.querySelectorAll('mark[data-side=\"A\"]').length")
        sideB = c.js("document.querySelectorAll('mark[data-side=\"B\"]').length")
        check(sideA > 0 and sideB > 0,
              f"highlights exist in both panes (A={sideA}, B={sideB})")

        vis = ("Array.prototype.filter.call(document.querySelectorAll('mark'),"
               "function(m){return !m.classList.contains('off')}).length")

        # ---- filter: minimum run length
        base = c.js(vis)
        c.js("""(function(){var s=document.getElementById('minlen');
                 s.value='40';s.dispatchEvent(new Event('change'));})()""")
        at40 = c.js(vis)
        c.js("""(function(){var s=document.getElementById('minlen');
                 s.value='8';s.dispatchEvent(new Event('change'));})()""")
        back = c.js(vis)
        check(at40 < base and back == base,
              f"run-length filter works and is reversible "
              f"(all={base}, >=40 words={at40}, back={back})")

        # ---- filter: citation class
        c.js("""(function(){var s=document.getElementById('cite');
                 s.value='attributed';s.dispatchEvent(new Event('change'));})()""")
        attr = c.js(vis)
        c.js("""(function(){var s=document.getElementById('cite');
                 s.value='all';s.dispatchEvent(new Event('change'));})()""")
        check(0 < attr < base,
              f"citation-class filter works (attributed={attr} of {base})")

        # ---- filter: direction-flagged only
        c.js("""(function(){var e=document.getElementById('dironly');
                 e.checked=true;e.dispatchEvent(new Event('change'));})()""")
        dironly = c.js(vis)
        c.js("""(function(){var e=document.getElementById('dironly');
                 e.checked=false;e.dispatchEvent(new Event('change'));})()""")
        check(0 < dironly < base,
              f"direction-flagged filter works ({dironly} of {base})")

        # ---- click-to-scroll cross-linking
        moved = c.js("""(function(){
          var paneB=document.getElementById('paneB');
          var before=paneB.scrollTop;
          var marksA=document.querySelectorAll('.pane#paneA mark[data-side="A"]');
          var el=null;
          for (var i=marksA.length-1;i>=0;i--){ if(!marksA[i].classList.contains('off')){el=marksA[i];break;} }
          if(!el) return {ok:false,why:'no visible mark in pane A'};
          el.click();
          var selB=document.querySelectorAll('#paneB mark.sel').length;
          var selA=document.querySelectorAll('#paneA mark.sel').length;
          return {ok:true, before:before, after:paneB.scrollTop,
                  selA:selA, selB:selB,
                  stat:(document.getElementById('stat')||{}).textContent||''};
        })()""")
        time.sleep(1.0)
        after = c.js("document.getElementById('paneB').scrollTop")
        check(moved.get("selA") == 1 and moved.get("selB") == 1,
              f"clicking a highlight selects its counterpart in the other pane "
              f"(A sel={moved.get('selA')}, B sel={moved.get('selB')})")
        check(after != moved.get("before"),
              f"clicking a highlight scrolls the other pane to its counterpart "
              f"(scrollTop {moved.get('before')} -> {after})")
        check("words" in (moved.get("stat") or ""),
              f"the status line reports the clicked match: "
              f"{(moved.get('stat') or '')[:90]!r}")

        # ---- search filter
        c.js("""(function(){var s=document.getElementById('srch');
                 s.value='zzzzznotpresentzzzzz';s.dispatchEvent(new Event('input'));})()""")
        none_left = c.js(vis)
        c.js("""(function(){var s=document.getElementById('srch');
                 s.value='';s.dispatchEvent(new Event('input'));})()""")
        check(none_left == 0,
              f"free-text search filters down to nothing for an absent string "
              f"(got {none_left})")
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except Exception:
            proc.kill()
        shutil.rmtree(profile, ignore_errors=True)

    failed = [w for ok, w in results if not ok]
    print("-" * 70)
    if failed:
        print(f"VIEWER CHECK FAILED — {len(failed)} of {len(results)}:")
        for w in failed:
            print("  -", w)
        return 1
    print(f"VIEWER CHECK PASSED — {len(results)}/{len(results)} checks green "
          f"({target}).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
