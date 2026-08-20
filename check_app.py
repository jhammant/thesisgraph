#!/usr/bin/env python3
"""check_app.py — render out/app.html from file:// in headless Chromium and
assert the interactive application behaves: zero console errors, zero external
references, working filters, match list, keyboard navigation, minimaps, tab
switching, findings charts, synced scrolling and deep links.

Paths anchor to this file's location, not the shell's working directory.

    python check_app.py [path/to/app.html]
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

from check_viewer import CDP, find_chrome, free_port  # reuse the CDP driver

import json
import shutil
import subprocess
import tempfile
import urllib.request

HERE = Path(__file__).resolve().parent


def main() -> int:
    target = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 \
        else HERE / "out" / "app.html"
    if not target.exists():
        raise SystemExit(f"missing {target}")

    port = free_port()
    profile = tempfile.mkdtemp(prefix="appcheck-")
    proc = subprocess.Popen(
        [find_chrome(), "--headless=new", f"--remote-debugging-port={port}",
         f"--user-data-dir={profile}", "--no-first-run",
         "--no-default-browser-check", "--remote-allow-origins=*",
         "--disable-gpu", "--window-size=1700,1050", "about:blank"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    results: list[tuple[bool, str]] = []

    def check(cond, what):
        results.append((bool(cond), what))
        print(f"  {'PASS' if cond else 'FAIL'}  {what}")

    try:
        ws_url = None
        for _ in range(120):
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
        c.drain(5.0)

        errs = []
        for e in c.events:
            m = e.get("method")
            if m == "Log.entryAdded":
                en = e["params"]["entry"]
                if en.get("level") in ("error", "warning"):
                    errs.append(f"{en['level']}: {en.get('text')}")
            elif m == "Runtime.exceptionThrown":
                errs.append("exception: " + str(
                    e["params"]["exceptionDetails"].get("text"))[:160])
            elif m == "Network.loadingFailed":
                errs.append("loadingFailed: " + str(e["params"].get("errorText")))
        check(not errs, f"zero console errors/warnings, no failed loads "
                        f"(saw {len(errs)}: {errs[:2]})")

        reqs = [e["params"]["request"]["url"] for e in c.events
                if e.get("method") == "Network.requestWillBeSent"]
        check(all(u.startswith("file:") for u in reqs),
              "every request stays on file:// (no server, no CDN)")
        ext = c.js("""(function(){var b=[];
          document.querySelectorAll('[src],link[href],a[href^="http"],iframe,img')
            .forEach(function(e){b.push(e.tagName)});return b;})()""")
        check(not ext, f"no external resource references in the DOM (saw {ext[:3]})")

        vis = ("Array.prototype.filter.call(document.querySelectorAll('mark'),"
               "function(m){return !m.classList.contains('off')}).length")
        rows = "document.querySelectorAll('#list .item').length"

        base_marks = c.js(vis)
        base_rows = c.js(rows)
        check(base_marks > 0 and base_rows > 0,
              f"loads with highlights and a populated match list "
              f"(marks={base_marks}, list rows={base_rows})")
        check(c.js("document.querySelectorAll('.pane').length") == 2,
              "both document panes render side by side")
        check(c.js("document.querySelectorAll('#miniA i').length") > 0 and
              c.js("document.querySelectorAll('#miniB i').length") > 0,
              "density minimaps render ticks for both documents")

        # ---- min-run-length slider
        c.js("""(function(){var s=document.getElementById('min');
                 s.value='40';s.dispatchEvent(new Event('input'));})()""")
        at40_marks, at40_rows = c.js(vis), c.js(rows)
        c.js("""(function(){var s=document.getElementById('min');
                 s.value='8';s.dispatchEvent(new Event('input'));})()""")
        check(at40_marks < base_marks and at40_rows < base_rows
              and c.js(vis) == base_marks,
              f"run-length slider filters marks and list, and is reversible "
              f"(marks {base_marks}->{at40_marks}, rows {base_rows}->{at40_rows})")

        # ---- citation class
        c.js("""(function(){var s=document.getElementById('cite');
                 s.value='attributed';s.dispatchEvent(new Event('change'));})()""")
        att = c.js(rows)
        c.js("""(function(){var s=document.getElementById('cite');
                 s.value='all';s.dispatchEvent(new Event('change'));})()""")
        check(0 < att < base_rows,
              f"citation-class filter works (attributed={att} of {base_rows})")

        # ---- chapter bucket
        c.js("""(function(){var s=document.getElementById('bucket');
                 s.value='methodology';s.dispatchEvent(new Event('change'));})()""")
        meth = c.js(rows)
        c.js("""(function(){var s=document.getElementById('bucket');
                 s.value='all';s.dispatchEvent(new Event('change'));})()""")
        check(0 < meth < base_rows,
              f"chapter filter works (methodology={meth} of {base_rows})")

        # ---- direction-only
        c.js("""(function(){var e=document.getElementById('dir');
                 e.checked=true;e.dispatchEvent(new Event('change'));})()""")
        dr = c.js(rows)
        c.js("""(function(){var e=document.getElementById('dir');
                 e.checked=false;e.dispatchEvent(new Event('change'));})()""")
        check(0 < dr < base_rows,
              f"direction-flagged filter works ({dr} of {base_rows})")

        # ---- extra passes can be switched on
        c.js("""(function(){var c=document.querySelector('.passchk[value="near_verbatim"]');
                 if(c){c.checked=true;c.dispatchEvent(new Event('change'));}})()""")
        withnear = c.js(rows)
        c.js("""(function(){var c=document.querySelector('.passchk[value="near_verbatim"]');
                 if(c){c.checked=false;c.dispatchEvent(new Event('change'));}})()""")
        check(withnear > base_rows,
              f"enabling the near-verbatim pass adds matches "
              f"({base_rows} -> {withnear})")

        # ---- sorting
        c.js("""(function(){var s=document.getElementById('sortby');
                 s.value='len';s.dispatchEvent(new Event('change'));})()""")
        first_len = c.js("""(function(){var e=document.querySelector('#list .item .w');
                 return e?parseInt(e.textContent,10):0;})()""")
        check(first_len >= 40,
              f"'longest first' really sorts by length (top row = {first_len}w)")

        # ---- clicking a list row selects and scrolls BOTH panes
        moved = c.js("""(function(){
          var a=document.getElementById('paneA'), b=document.getElementById('paneB');
          var rowsEl=document.querySelectorAll('#list .item');
          var r=rowsEl[Math.min(3,rowsEl.length-1)];
          var b0=b.scrollTop, a0=a.scrollTop;
          r.click();
          return {a0:a0,b0:b0,id:r.getAttribute('data-id'),
                  selA:document.querySelectorAll('#paneA mark.sel').length,
                  selB:document.querySelectorAll('#paneB mark.sel').length,
                  status:document.getElementById('status').textContent};
        })()""")
        time.sleep(1.2)
        a1 = c.js("document.getElementById('paneA').scrollTop")
        b1 = c.js("document.getElementById('paneB').scrollTop")
        check(moved["selA"] == 1 and moved["selB"] == 1,
              f"clicking a match selects it in BOTH panes "
              f"(A={moved['selA']}, B={moved['selB']})")
        check(a1 != moved["a0"] and b1 != moved["b0"],
              f"clicking a match scrolls both panes "
              f"(A {moved['a0']}->{a1}, B {moved['b0']}->{b1})")
        check("words" in (moved["status"] or ""),
              f"status line describes the selected match: "
              f"{(moved['status'] or '')[:80]!r}")

        # ---- deep link written to the URL
        check("#m=" in c.js("location.hash"),
              f"selection is deep-linked in the URL hash ({c.js('location.hash')[:40]})")

        # ---- keyboard navigation
        before = c.js("location.hash")
        c.js("""document.dispatchEvent(new KeyboardEvent('keydown',
             {key:'j',bubbles:true}))""")
        time.sleep(0.4)
        after_j = c.js("location.hash")
        c.js("""document.dispatchEvent(new KeyboardEvent('keydown',
             {key:'k',bubbles:true}))""")
        time.sleep(0.4)
        after_k = c.js("location.hash")
        tail = lambda h: h.rsplit("%3A", 1)[-1].rsplit(":", 1)[-1] or h
        check(after_j != before and after_k == before,
              f"j/k step forward and back through matches "
              f"({tail(before)} -> {tail(after_j)} -> {tail(after_k)})")

        # ---- clicking a highlight in the text
        hl = c.js("""(function(){
          var b=document.getElementById('paneB');
          var ms=document.querySelectorAll('#paneB mark');
          var el=null;
          for(var i=0;i<ms.length;i++){ if(!ms[i].classList.contains('off')){el=ms[i];break;} }
          if(!el) return {ok:false};
          var a0=document.getElementById('paneA').scrollTop;
          el.click();
          return {ok:true,a0:a0,selA:document.querySelectorAll('#paneA mark.sel').length};
        })()""")
        time.sleep(1.0)
        check(hl.get("ok") and hl.get("selA") == 1,
              "clicking a highlight in the later document selects its "
              "counterpart in the earlier one")

        # ---- minimap click jumps
        mm = c.js("""(function(){
          var a=document.getElementById('paneA');
          var t=document.querySelector('#miniA i');
          if(!t) return {ok:false};
          var before=a.scrollTop; t.click();
          return {ok:true,before:before};
        })()""")
        time.sleep(0.8)
        check(mm.get("ok") and
              c.js("document.getElementById('paneA').scrollTop") != mm["before"],
              "clicking a minimap tick jumps the pane to that match")

        # ---- tabs and charts
        c.js("""document.querySelector('.tab[data-v="findings"]').click()""")
        time.sleep(0.6)
        svgs = c.js("document.querySelectorAll('#findings svg').length")
        barcount = c.js("document.querySelectorAll('#findings svg rect').length")
        shown = c.js("""getComputedStyle(document.getElementById('findings')).display""")
        check(shown != "none" and svgs >= 6 and barcount > 20,
              f"Findings tab renders inline SVG charts "
              f"({svgs} charts, {barcount} bars, display={shown})")
        check(c.js("""document.querySelectorAll('#findings tr.focal').length""") == 1,
              "the control-comparison table highlights the focal pair")
        c.js("""document.querySelector('.tab[data-v="compare"]').click()""")
        time.sleep(0.4)
        check(c.js("""getComputedStyle(document.getElementById('findings')).display""")
              == "none" and c.js(vis) > 0,
              "switching back to Compare restores the side-by-side view")

        # ---- synced scrolling
        sync = c.js("""(function(){
          var e=document.getElementById('sync');
          e.checked=true; e.dispatchEvent(new Event('change'));
          var a=document.getElementById('paneA'), b=document.getElementById('paneB');
          var a0=a.scrollTop;
          b.scrollTop = Math.floor(b.scrollHeight*0.55);
          b.dispatchEvent(new Event('scroll'));
          return {a0:a0};
        })()""")
        time.sleep(1.0)
        a_after = c.js("document.getElementById('paneA').scrollTop")
        check(a_after != sync["a0"],
              f"synced scrolling moves the other pane "
              f"({sync['a0']} -> {a_after})")

        # ---- search
        c.js("""(function(){var s=document.getElementById('q');
                 s.value='zzzznotpresentzzzz';s.dispatchEvent(new Event('input'));})()""")
        check(c.js(vis) == 0 and c.js(rows) == 0,
              "free-text search filters down to nothing for an absent string")
        c.js("""(function(){var s=document.getElementById('q');
                 s.value='';s.dispatchEvent(new Event('input'));})()""")

        # ---- reset
        c.js("""document.getElementById('reset').click()""")
        time.sleep(0.3)
        check(c.js(vis) == base_marks,
              f"reset restores the default filter state "
              f"({c.js(vis)} == {base_marks})")

        # ---- no horizontal overflow
        check(c.js("document.body.scrollWidth <= window.innerWidth + 2"),
              "the page does not scroll horizontally")
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
        print(f"APP CHECK FAILED — {len(failed)} of {len(results)}:")
        for w in failed:
            print("  -", w)
        return 1
    print(f"APP CHECK PASSED — {len(results)}/{len(results)} checks green ({target}).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
