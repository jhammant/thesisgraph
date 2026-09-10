#!/usr/bin/env python3
"""make_bridge_app.py — the cross-field reader.

A graph tells you Education and Biology have a Jaccard of 0.017. That is not
actionable. This answers the next three questions instead:

    which works bridge these two fields?
    which theses on each side cite them?
    where exactly — page, chapter, and the sentence — so you can go and read it?

Every passage carries a deep link to that page of the source PDF.
"""
from __future__ import annotations
import argparse, base64, gzip, json
from pathlib import Path

HERE = Path(__file__).resolve().parent
B = json.loads((HERE / "corpus" / "bridges.json").read_text(encoding="utf-8"))
D = json.loads((HERE / "corpus" / "docs.json").read_text(encoding="utf-8"))
OUT = HERE / "out" / "bridges.html"
PUB = HERE / "out" / "public" / "bridges.html"

CSS = """
:root{--bg:#0f1115;--panel:#171a21;--card:#1c202a;--fg:#e8eaf0;--mut:#8b93a7;
--line:#262b36;--acc:#5b9cff;--acc2:#ffb302;--ok:#4ecf8f}
*{box-sizing:border-box}html,body{margin:0;height:100%}
body{background:var(--bg);color:var(--fg);font:14px/1.6 -apple-system,BlinkMacSystemFont,
"Segoe UI",Roboto,Helvetica,Arial,sans-serif;display:flex;flex-direction:column;height:100vh}
header{background:var(--panel);border-bottom:1px solid var(--line);padding:11px 16px;flex:0 0 auto}
.top{display:flex;gap:14px;align-items:center;flex-wrap:wrap}
h1{font-size:15px;margin:0;font-weight:650}
h1 small{display:block;font-weight:400;color:var(--mut);font-size:11.5px;margin-top:2px}
.sel{display:flex;gap:8px;align-items:center;margin-left:auto;flex-wrap:wrap}
select,input[type=search]{background:var(--bg);border:1px solid var(--line);color:var(--fg);
border-radius:7px;padding:6px 9px;font:inherit}
select{max-width:230px}
a.nav{color:var(--acc);text-decoration:none;font-size:12px}
main{flex:1 1 auto;overflow-y:auto;padding:16px}
.wrap{max-width:1180px;margin:0 auto}
.count{color:var(--mut);font-size:12px;margin-bottom:12px}
.bridge{background:var(--panel);border:1px solid var(--line);border-radius:11px;
margin-bottom:12px;overflow:hidden}
.bh{padding:12px 15px;cursor:pointer;display:flex;gap:12px;align-items:baseline;flex-wrap:wrap}
.bh:hover{background:var(--card)}
.bh .w{font-weight:650;font-size:14.5px}
.bh .ti{color:var(--mut);font-size:12.5px;flex:1 1 260px}
.pair{font-size:11.5px;background:var(--card);border:1px solid var(--line);
border-radius:999px;padding:2px 10px;white-space:nowrap}
.pair b{color:var(--acc)}
.nn{color:var(--mut);font-size:11.5px;white-space:nowrap}
.body{display:none;border-top:1px solid var(--line);padding:4px 15px 14px}
.bridge.open .body{display:block}
.sides{display:grid;grid-template-columns:repeat(auto-fit,minmax(330px,1fr));
gap:16px;margin-top:10px}
.side h4{margin:8px 0 8px;font-size:12px;color:var(--acc);text-transform:uppercase;
letter-spacing:.05em}
.th{background:var(--card);border:1px solid var(--line);border-radius:9px;padding:10px 12px;
margin-bottom:9px}
.th .t{font-size:12.8px;font-weight:600;line-height:1.4}
.th .m{color:var(--mut);font-size:11px;margin:3px 0 7px}
.pg{border-left:2px solid var(--line);padding:4px 0 4px 10px;margin:7px 0;font-size:12.3px}
.pg .loc{color:var(--mut);font-size:11px;display:flex;gap:8px;align-items:center;flex-wrap:wrap}
.pg .loc a{color:var(--acc2);text-decoration:none;font-weight:600}
.pg .loc a:hover{text-decoration:underline}
.pg q{display:block;margin-top:3px;color:var(--fg);quotes:none}
.pg q:before{content:'"'}.pg q:after{content:'"'}
.chip{background:#0f1115;border:1px solid var(--line);border-radius:5px;padding:0 5px;font-size:10px}
.empty{color:var(--mut);padding:40px;text-align:center;font-style:italic}
.mapwrap{background:var(--panel);border:1px solid var(--line);border-radius:12px;
padding:16px 18px;margin-bottom:16px}
.maphead b{color:var(--acc)}
.mapsub{color:var(--mut);font-size:12px;margin-top:5px;max-width:760px;line-height:1.55}
.mscroll{overflow-x:auto;margin-top:14px}
table.mx{border-collapse:separate;border-spacing:2px}
table.mx th{font-weight:600;font-size:10.5px;color:var(--mut)}
table.mx th.rl{text-align:right;padding-right:8px;white-space:nowrap;max-width:130px}
table.mx th.vt{height:104px;vertical-align:bottom;padding:0}
table.mx th.vt span{display:block;writing-mode:vertical-rl;transform:rotate(180deg);
white-space:nowrap;margin:0 auto 4px}
table.mx td{width:26px;height:26px;border-radius:4px;text-align:center;
font-size:10px;color:#08101f;font-weight:700}
table.mx td.c{cursor:pointer;outline:1px solid transparent}
table.mx td.c:hover{outline:2px solid var(--acc2)}
table.mx td.z{background:#14171f}
table.mx td.dg{background:#0c0e13}
.feath{color:var(--mut);font-size:11px;text-transform:uppercase;letter-spacing:.05em;
margin:16px 0 8px}
.feat{background:var(--panel);border:1px solid var(--line);border-radius:9px;
padding:9px 13px;margin-bottom:7px;cursor:pointer;display:flex;gap:10px;
align-items:baseline;flex-wrap:wrap}
.feat:hover{border-color:var(--acc)}
.feat .fw{font-weight:650}
.feat .ft{color:var(--mut);font-size:12.5px;flex:1 1 240px}
.backrow{margin-bottom:12px}
.backlink{color:var(--acc);cursor:pointer;font-size:12.5px;text-decoration:none}
.backlink:hover{text-decoration:underline}
.roster{background:var(--panel);border:1px solid var(--acc);border-radius:11px;
padding:12px 15px;margin-bottom:16px}
.rh{font-size:12.5px;font-weight:650;margin-bottom:4px}
.mut2{color:var(--mut);font-weight:400;font-size:11.5px}
.rrow{padding:6px 0;border-bottom:1px solid var(--line)}
.rrow a.drill{font-size:12.6px;font-weight:600}
.rrow .rm{color:var(--mut);font-size:11px;margin-top:2px}
a.drill{color:var(--acc);cursor:pointer;text-decoration:none}
a.drill:hover{text-decoration:underline}
.doc{background:#12151c;border:1px solid var(--line);border-radius:8px;padding:10px 12px;
margin:6px 0 9px}
.doc .dm{color:var(--mut);font-size:11px;margin-bottom:6px}
.doc .ab{font-size:12px;color:var(--fg);opacity:.9;margin-bottom:9px;line-height:1.55}
.toch{color:var(--acc);font-size:10.5px;text-transform:uppercase;letter-spacing:.05em;
margin:8px 0 5px}
.tocrow{display:flex;gap:8px;align-items:baseline;font-size:11.8px;padding:2px 0;
border-bottom:1px solid #1d2230}
.tocrow a{color:var(--acc2);text-decoration:none;font-weight:600;min-width:44px}
.tocrow .tb{color:var(--mut);font-size:10px;min-width:86px}
.tocrow .tt{flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.lk2{margin-top:8px;font-size:11.5px}
.lk2 a{color:var(--acc);text-decoration:none;margin-right:10px}
.note{color:var(--mut);font-size:11.5px;border-top:1px solid var(--line);margin-top:22px;
padding-top:12px;line-height:1.6}
"""

JS = r"""
(function(){
var W=window.__B__.works, A=document.getElementById('fa'), Bx=document.getElementById('fb'),
    Q=document.getElementById('q'), L=document.getElementById('list'), C=document.getElementById('count');
var fields={}; W.forEach(function(w){ fields[w.a]=1; fields[w.b]=1; });
var fs=Object.keys(fields).sort();
[A,Bx].forEach(function(s,i){
  s.innerHTML='<option value="">'+(i?'…and any field':'any field…')+'</option>'+
    fs.map(function(f){return '<option>'+esc(f)+'</option>';}).join(''); });
function esc(s){return String(s).replace(/[&<>"]/g,function(c){
  return c==='&'?'&amp;':c==='<'?'&lt;':c==='>'?'&gt;':'&quot;';});}
function match(w){
  var a=A.value,b=Bx.value;
  if(a&&b){ if(!((w.a===a&&w.b===b)||(w.a===b&&w.b===a))) return false; }
  else if(a){ if(w.a!==a&&w.b!==a) return false; }
  else if(b){ if(w.a!==b&&w.b!==b) return false; }
  var q=(Q.value||'').toLowerCase();
  if(q){
    var hay=(w.w+' '+w.title+' '+w.a+' '+w.b).toLowerCase();
    if(hay.indexOf(q)<0){
      var deep=false;
      for(var k in w.sides){ w.sides[k].forEach(function(e){
        if((e.t+' '+k).toLowerCase().indexOf(q)>=0) deep=true; }); }
      if(!deep) return false; }
  }
  return true;
}
function docPanel(id){
  var d=(window.__DOCS__||{})[id];
  if(!d) return '<div class="mut" style="font-size:11.5px">No document record.</div>';
  var toc=(d.toc||[]).map(function(t){
    var link=d.pdf?(d.pdf+'#page='+t.pg):d.u;
    return '<div class="tocrow">'+
      (link?'<a href="'+esc(link)+'" target="_blank" rel="noopener">p.'+t.pg+'</a>':'p.'+t.pg)+
      (t.pr?'<span class="chip">pr.'+esc(t.pr)+'</span>':'')+
      '<span class="tb">'+esc(t.b||'')+'</span>'+
      '<span class="tt">'+esc(t.h)+'</span></div>'; }).join('');
  return '<div class="doc">'+
    '<div class="dm">'+esc(d.a||'unknown author')+' · '+esc(d.i||'')+' · '+(d.y||'')+
    ' · '+(d.pg||'?')+' pages · '+((d.w||0)/1000).toFixed(0)+'k words</div>'+
    (d.ab?'<div class="ab">'+esc(d.ab)+'</div>':'')+
    (toc?'<div class="tocb"><div class="toch">contents — jump into the PDF</div>'+toc+'</div>':'')+
    '<div class="lk2">'+
      (d.u?'<a href="'+esc(d.u)+'" target="_blank" rel="noopener">repository record ↗</a>':'')+
      (d.pdf?' <a href="'+esc(d.pdf)+'" target="_blank" rel="noopener">full PDF ↗</a>':'')+
    '</div></div>';
}
function passage(p,e){
  var pg=p.pg, link=e.pdf?(e.pdf+'#page='+pg):e.u;
  return '<div class="pg"><div class="loc">'+
    (link?'<a href="'+esc(link)+'" target="_blank" rel="noopener">read p.'+pg+' ↗</a>':'p.'+pg)+
    (p.pr?'<span class="chip">printed p.'+esc(p.pr)+'</span>':'')+
    (p.b?'<span class="chip">'+esc(p.b)+'</span>':'')+
    (p.h?'<span class="chip">'+esc(p.h.slice(0,44))+'</span>':'')+
    '</div><q>'+esc(p.s)+'</q></div>';
}
function roster(vis){
  /* Every thesis linking the two selected fields, across ALL their shared
     works — deduplicated, with how many of those works each one cites. */
  if(!(A.value&&Bx.value)) return '';
  var by={};
  vis.forEach(function(w){
    Object.keys(w.sides).forEach(function(k){
      w.sides[k].forEach(function(e){
        var d=(window.__DOCS__||{})[e.id]||{};
        var side=(d.d===A.value)?A.value:(d.d===Bx.value)?Bx.value:null;
        if(!side) return;
        var b=by[side]=by[side]||{};
        var r=b[e.id]=b[e.id]||{t:e.t,y:e.y,sf:e.sf,u:e.u,id:e.id,works:[]};
        if(r.works.indexOf(w.w)<0) r.works.push(w.w);
      }); }); });
  var sides=[A.value,Bx.value].filter(function(k){return by[k];});
  if(!sides.length) return '';
  return '<div class="roster"><div class="rh">every thesis linking these two '+
    'fields <span class="mut2">— click a title to open the thesis, or a work to '+
    'jump to it below</span></div><div class="sides">'+
    sides.map(function(k){
      var rows=Object.keys(by[k]).map(function(id){return by[k][id];})
        .sort(function(x,y){return y.works.length-x.works.length ||
              (x.t<y.t?-1:1);});
      return '<div class="side"><h4>'+esc(k)+' — '+rows.length+' theses</h4>'+
        rows.map(function(r){
          return '<div class="rrow"><a class="drill" data-d="'+esc(r.id)+'">'+
            esc(r.t)+'</a><div class="rm">'+esc(r.sf||'')+(r.y?' · '+r.y:'')+
            ' · cites '+r.works.length+' shared work'+(r.works.length===1?'':'s')+
            ': '+r.works.map(esc).join(', ')+'</div>'+
            '<div class="docwrap" data-for="'+esc(r.id)+'"></div></div>'; }).join('')+
        '</div>'; }).join('')+'</div></div>';
}
/* ---------- the map: every discipline pair, as a clickable matrix --------- */
var MATRIX=null;
function matrixData(){
  if(MATRIX) return MATRIX;
  var pairs={}, tot={};
  W.forEach(function(w){
    var k=[w.a,w.b].sort().join('\u0000');
    pairs[k]=(pairs[k]||0)+1;
    tot[w.a]=(tot[w.a]||0)+1; tot[w.b]=(tot[w.b]||0)+1;
  });
  var fields=Object.keys(tot).sort(function(x,y){return tot[y]-tot[x]||(x<y?-1:1);});
  var max=0; for(var k in pairs) if(pairs[k]>max) max=pairs[k];
  MATRIX={fields:fields,pairs:pairs,max:max,tot:tot};
  return MATRIX;
}
function shortName(f){
  return f.replace('History, philosophy and religion','History & philosophy')
          .replace('Language and literature','Language & lit')
          .replace('Business and economics','Business & econ')
          .replace('Medicine and health','Medicine')
          .replace('Earth and environment','Earth & env')
          .replace('Biological sciences','Biology')
          .replace('Physical sciences','Physics & chem')
          .replace('Social sciences','Social science')
          .replace('Arts and media','Arts & media')
          .replace('Sport and agriculture','Sport & agri');
}
function renderMatrix(){
  var m=matrixData(), f=m.fields, n=f.length, out=[];
  out.push('<div class="mapwrap"><div class="maphead">'+
    '<div><b>'+W.length.toLocaleString()+'</b> works bridge two fields'+
    ' &middot; <b>'+Object.keys(m.pairs).length+'</b> of '+(n*(n-1)/2)+
    ' possible pairings have one</div>'+
    '<div class="mapsub">Every cell is a pair of disciplines. Brighter means more '+
    'shared literature; <b>empty means they read nothing in common</b>. '+
    'Click a cell to read the works that bridge them.</div></div>');
  out.push('<div class="mscroll"><table class="mx"><tr><th></th>');
  f.forEach(function(c){ out.push('<th class="vt"><span>'+esc(shortName(c))+'</span></th>'); });
  out.push('</tr>');
  f.forEach(function(r,ri){
    out.push('<tr><th class="rl">'+esc(shortName(r))+'</th>');
    f.forEach(function(c,ci){
      if(ci===ri){ out.push('<td class="dg"></td>'); return; }
      var v=m.pairs[[r,c].sort().join('\u0000')]||0;
      if(!v){ out.push('<td class="z" title="'+esc(r)+' \u21c4 '+esc(c)+
        ' — no shared literature"></td>'); return; }
      var t=Math.pow(v/m.max,0.45);
      out.push('<td class="c" style="background:rgba(91,156,255,'+(0.10+0.85*t).toFixed(3)+')"'+
        ' data-a="'+esc(r)+'" data-b="'+esc(c)+'" title="'+esc(r)+' \u21c4 '+esc(c)+
        ' — '+v+' bridging work'+(v===1?'':'s')+'">'+(v>=10?v:'')+'</td>');
    });
    out.push('</tr>');
  });
  out.push('</table></div></div>');
  return out.join('');
}
function featured(){
  /* a few strong, legible crossovers so the page opens on something real */
  var picks=W.filter(function(w){
    var n=0; for(var k in w.sides) n+=w.sides[k].length;
    return n>=6 && w.a!==w.b;
  }).slice(0,10);
  if(!picks.length) return '';
  return '<div class="feath">Or start with one of these</div>'+
    picks.map(function(w){
      var i=W.indexOf(w);
      return '<div class="feat" data-a="'+esc(w.a)+'" data-b="'+esc(w.b)+'">'+
        '<span class="fw">'+esc(w.w)+'</span> <span class="ft">'+esc(w.title)+'</span>'+
        '<span class="pair"><b>'+esc(w.a)+'</b> \u21c4 <b>'+esc(w.b)+'</b></span></div>';
    }).join('');
}

function render(){
  var landing = !A.value && !Bx.value && !(Q.value||'').trim();
  if(landing){
    C.textContent='';
    L.innerHTML=renderMatrix()+featured();
    L.querySelectorAll('td.c, .feat').forEach(function(el){
      el.onclick=function(){ A.value=el.dataset.a; Bx.value=el.dataset.b;
        render(); window.scrollTo(0,0); };
    });
    return;
  }
  var vis=W.filter(match);
  C.textContent=vis.length+' bridging work'+(vis.length===1?'':'s')+
    (A.value||Bx.value?' for this pairing':'')+
    ' · '+vis.reduce(function(a,w){var n=0;for(var k in w.sides)n+=w.sides[k].length;return a+n;},0)+
    ' theses · click a work to read the passages';
  var back='<div class="backrow"><a class="backlink" id="back">\u2190 back to the map</a></div>';
  if(!vis.length){ L.innerHTML=back+'<div class="empty">No bridging works found for that '+
    'pairing. Not every pair of fields shares specific literature — that absence '+
    'is itself the finding.</div>'; return; }
  L.innerHTML=back+roster(vis)+vis.map(function(w,i){
    var sides=Object.keys(w.sides).sort(function(x,y){
      return w.sides[y].length-w.sides[x].length; });
    return '<div class="bridge" data-i="'+i+'">'+
      '<div class="bh"><span class="w">'+esc(w.w)+'</span>'+
      '<span class="ti">'+esc(w.title)+'</span>'+
      '<span class="pair"><b>'+esc(w.a)+'</b> ⇄ <b>'+esc(w.b)+'</b></span>'+
      '<span class="nn">'+w.n+' theses cite it'+
        (function(){var shown=0;for(var k in w.sides)shown+=w.sides[k].length;
          return shown<w.n?' · '+shown+' with a located passage':'';})()+
      '</span></div>'+
      '<div class="body"><div class="sides">'+
      sides.map(function(k){
        return '<div class="side"><h4>'+esc(k)+' ('+w.sides[k].length+')</h4>'+
          w.sides[k].map(function(e){
            return '<div class="th"><div class="t">'+
              (e.u?'<a href="'+esc(e.u)+'" target="_blank" rel="noopener" '+
                   'style="color:inherit;text-decoration:none">'+esc(e.t)+' ↗</a>':esc(e.t))+
              '</div><div class="m">'+esc(e.sf||'')+(e.y?' · '+e.y:'')+
              (e.id?' · <a class="drill" data-d="'+esc(e.id)+'">open thesis ▾</a>':'')+'</div>'+
              '<div class="docwrap" data-for="'+esc(e.id||'')+'"></div>'+
              e.p.map(function(p){return passage(p,e);}).join('')+'</div>'; }).join('')+
          '</div>'; }).join('')+
      '</div></div></div>'; }).join('');
  var bk=document.getElementById('back');
  if(bk) bk.onclick=function(){ A.value=''; Bx.value=''; Q.value=''; render();
                                window.scrollTo(0,0); };
  L.querySelectorAll('.bh').forEach(function(h){
    h.onclick=function(){ h.parentNode.classList.toggle('open'); }; });
  L.querySelectorAll('.drill').forEach(function(a){
    a.onclick=function(ev){ ev.stopPropagation();
      var w=a.closest('.th,.rrow').querySelector('.docwrap');
      if(w.innerHTML){ w.innerHTML=''; a.textContent=a.dataset.lbl||'open thesis ▾'; }
      else { a.dataset.lbl=a.dataset.lbl||a.textContent;
        w.innerHTML=docPanel(a.dataset.d); } }; });
}
[A,Bx].forEach(function(s){ s.onchange=render; });
Q.oninput=render;
document.getElementById('swap').onclick=function(){
  var t=A.value; A.value=Bx.value; Bx.value=t; render(); };
render();
})();
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--public", action="store_true")
    args = ap.parse_args()
    out = PUB if args.public else OUT
    # The host serves static files uncompressed, and the payload is ~18 MB raw.
    # Rather than gut the data, gzip it into the page and inflate in the browser
    # with DecompressionStream: ~4x smaller over the wire, still one self-
    # contained file that opens from file:// with no server and no network.
    raw = json.dumps({"works": B["works"], "docs": D},
                     separators=(",", ":")).encode("utf-8")
    payload = base64.b64encode(gzip.compress(raw, 9)).decode("ascii")
    print(f"  payload {len(raw)/1e6:.1f} MB -> {len(payload)/1e6:.1f} MB embedded "
          f"({100*len(payload)/len(raw):.0f}%)")
    graph_link = "index.html" if args.public else "graph.html"
    appjs = json.dumps(JS)
    html = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Where fields cross over</title><style>{CSS}</style></head><body>
<header><div class="top">
  <h1>Where fields cross over
    <small>the specific works two fields share, and the page to read it on</small></h1>
  <div class="sel">
    <select id="fa"></select>
    <button class="nav" id="swap" style="background:none;border:0;cursor:pointer">⇄</button>
    <select id="fb"></select>
    <input type="search" id="q" placeholder="search work, title or thesis">
    <a class="nav" href="{graph_link}">graph view →</a>
    <a class="nav" href="https://github.com/jhammant/thesisgraph" target="_blank" rel="noopener">source →</a>
  </div>
</div></header>
<main><div class="wrap">
  <div class="count" id="count"></div>
  <div id="list"></div>
  <p class="note">A bridging work is one cited by theses in two different
  disciplines, and <b>not</b> by many others — universal methods literature such
  as Braun &amp; Clarke is excluded, because "both fields use thematic analysis"
  is not a crossover. Passages are the sentence in which the citing thesis
  mentions the work, quoted briefly with its page for reference; follow the link
  to read it in context in the original deposit. Sources: openly deposited
  theses in White Rose eTheses Online.
  <a href="https://github.com/jhammant/thesisgraph" target="_blank" rel="noopener" style="color:var(--acc)">Source
  and method on GitHub</a>.</p>
</div></main>
<script type="text/plain" id="p">{payload}</script>
<script>window.__APPJS__={appjs};</script>
<script>
(async function(){{
  var b64=document.getElementById('p').textContent.trim();
  var bin=Uint8Array.from(atob(b64),function(c){{return c.charCodeAt(0);}});
  var txt;
  if (typeof DecompressionStream!=='undefined') {{
    var ds=new DecompressionStream('gzip');
    var stream=new Blob([bin]).stream().pipeThrough(ds);
    txt=await new Response(stream).text();
  }} else {{
    document.getElementById('list').innerHTML=
      '<div class="empty">This browser lacks DecompressionStream; '+
      'please use a current Chrome, Safari or Firefox.</div>';
    return;
  }}
  var D=JSON.parse(txt);
  window.__B__={{works:D.works}}; window.__DOCS__=D.docs;
  var s=document.createElement('script'); s.textContent=window.__APPJS__; document.body.appendChild(s);
}})();
</script>
</body></html>"""
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    print(f"written {out} ({out.stat().st_size/1024:.0f} KB)")


if __name__ == "__main__":
    main()
