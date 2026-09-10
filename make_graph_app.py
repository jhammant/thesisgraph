#!/usr/bin/env python3
"""make_graph_app.py — self-contained graph explorer for the thesis corpus.

Layout is a STATIC RADIAL SECTOR map, not a force simulation: nodes are grouped
into angular wedges by parent discipline, so cross-discipline edges cut across
the centre and are immediately visible. Being static, it cannot oscillate — the
force version bounced, and for "which fields share literature" the sector
arrangement carries more information than an emergent one anyway.

The Subject tree view is expandable: click a node to split it into its
sub-fields, click again to collapse. Granularity is therefore a thing you steer,
not a number baked in at build time.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path

HERE = Path(__file__).resolve().parent
G = json.loads((HERE / "corpus" / "graph.json").read_text(encoding="utf-8"))
T = json.loads((HERE / "corpus" / "hierarchy.json").read_text(encoding="utf-8"))
OUT = HERE / "out" / "graph.html"
PUB = HERE / "out" / "public" / "index.html"

VIEWS = [
    ("tree", "Subject tree", "Every subject, expandable. Click a node to split "
     "it into sub-fields; click again to collapse. Wedges are disciplines, rings "
     "are depth, and edges are shared cited literature — so links crossing the "
     "centre are cross-discipline."),
    ("cocite", "Co-citation", "The canonical works of UK doctoral research, "
     "grouped by the discipline that cites them most. Two works are linked when "
     "the same thesis cites both."),
    ("bridges", "Discipline bridges", "The 16 disciplines, linked by how much "
     "cited literature they share. Thin or missing links between large fields "
     "are the structural holes."),
    ("methods", "Methods by discipline", "Which research methods each discipline "
     "actually uses; edge weight is the percentage of its theses using it."),
    ("reuse", "Text reuse", "Theses linked by surviving verbatim overlap after "
     "artefact and template filtering. Screening signals, NOT findings."),
]

CSS = """
:root{--bg:#0f1115;--panel:#171a21;--fg:#e8eaf0;--mut:#8b93a7;--line:#262b36;
--acc:#5b9cff;--acc2:#ffb302}
*{box-sizing:border-box}html,body{margin:0;height:100%;overflow:hidden}
body{background:var(--bg);color:var(--fg);font:13px/1.5 -apple-system,BlinkMacSystemFont,
"Segoe UI",Roboto,Helvetica,Arial,sans-serif;display:flex;flex-direction:column}
header{background:var(--panel);border-bottom:1px solid var(--line);padding:9px 14px;
display:flex;gap:13px;align-items:center;flex-wrap:wrap;flex:0 0 auto}
.brand{font-weight:650}.brand small{display:block;font-weight:400;color:var(--mut);font-size:11px}
.tabs{display:flex;gap:4px}
.tab{background:transparent;border:1px solid var(--line);color:var(--mut);
border-radius:7px;padding:5px 11px;cursor:pointer;font:inherit}
.tab[aria-selected=true]{background:var(--acc);color:#08101f;border-color:var(--acc);font-weight:650}
.ctl{display:flex;gap:12px;align-items:center;margin-left:auto;flex-wrap:wrap}
.ctl label{color:var(--mut);font-size:11.5px;display:inline-flex;gap:6px;align-items:center}
input[type=search]{background:var(--bg);border:1px solid var(--line);color:var(--fg);
border-radius:6px;padding:4px 8px;font:inherit;width:160px}
input[type=range]{accent-color:var(--acc);width:104px}
button.mini{background:var(--bg);border:1px solid var(--line);color:var(--fg);
border-radius:6px;padding:4px 9px;cursor:pointer;font:inherit}
button.mini:hover{border-color:var(--acc)}
main{flex:1 1 auto;display:flex;min-height:0}
#wrap{flex:1 1 auto;position:relative;min-width:0}
canvas{display:block;width:100%;height:100%;cursor:grab}canvas.drag{cursor:grabbing}
aside{width:320px;flex:0 0 320px;background:var(--panel);border-left:1px solid var(--line);
padding:13px;overflow-y:auto}
aside h3{margin:0 0 5px;font-size:13px}
aside p.d{color:var(--mut);font-size:11.5px;margin:0 0 12px;line-height:1.5}
.kv{display:flex;justify-content:space-between;border-bottom:1px solid var(--line);
padding:5px 0;font-size:12px}.kv span:first-child{color:var(--mut)}
#sel .t{font-weight:650;margin:10px 0 3px;font-size:13px}
.lk{margin:7px 0 4px}.lk a{color:var(--acc);text-decoration:none;font-size:12px}
.lk a:hover{text-decoration:underline}
.lk .hint{display:block;color:var(--mut);font-size:10.5px;margin-top:2px}
.exh{margin-top:11px;color:var(--mut);font-size:10.5px;text-transform:uppercase;letter-spacing:.05em}
.ex a{display:block;color:var(--fg);font-size:11.5px;text-decoration:none;padding:3px 0;
border-bottom:1px solid var(--line)}.ex a:hover{color:var(--acc)}
#nbr{margin-top:9px;font-size:11.5px}
#nbr div{padding:3px 0;border-bottom:1px solid var(--line);color:var(--mut);cursor:pointer}
#nbr div:hover{color:var(--fg)}
.legend{margin-top:13px;font-size:11px;color:var(--mut)}
.legend i{display:inline-block;width:10px;height:10px;border-radius:3px;margin-right:5px;
vertical-align:-1px}
#tip{position:absolute;pointer-events:none;background:#000d;border:1px solid var(--line);
border-radius:6px;padding:6px 9px;font-size:11.5px;max-width:290px;display:none;z-index:5}
.note{margin-top:13px;color:var(--mut);font-size:11px;line-height:1.5;
border-top:1px solid var(--line);padding-top:10px}
.badge{background:var(--acc);color:#08101f;border-radius:4px;padding:0 5px;font-size:10px;
font-weight:650;margin-left:6px}
"""

JS = r"""
(function(){
var GD=window.__G__, TD=window.__T__, view='tree';
var cv=document.getElementById('cv'), ctx=cv.getContext('2d'), tip=document.getElementById('tip');
var cam={x:0,y:0,k:1}, W=0,Hh=0, N=[], E=[], sel=null, drag=null, pan=null, maxw=1;
var open={}, PAL=['#5b9cff','#ffb302','#4ecf8f','#ff7a7a','#c58cff','#4fd0e0','#ffa05c',
 '#9fd356','#f472b6','#7dd3fc','#fbbf24','#a3e635','#fb7185','#38bdf8','#facc15','#94a3b8'];
var groups={}, ng=0, minw=0;
function col(g){ if(!(g in groups)) groups[g]=PAL[(ng++)%PAL.length]; return groups[g]; }
function esc(s){return String(s).replace(/[&<>"]/g,function(c){
  return c==='&'?'&amp;':c==='<'?'&lt;':c==='>'?'&gt;':'&quot;';});}
function resize(){ var r=cv.getBoundingClientRect(), d=window.devicePixelRatio||1;
  cv.width=r.width*d; cv.height=r.height*d; ctx.setTransform(d,0,0,d,0,0); W=r.width; Hh=r.height; }
window.addEventListener('resize',function(){resize();layout();draw();});

/* ---------------- data assembly ---------------- */
function treeVisible(){
  var vis=[], byId={};
  TD.nodes.forEach(function(n){ byId[n.id]=n; });
  function walk(n){
    if(open[n.id] && (TD.kids[n.id]||[]).length){ TD.kids[n.id].forEach(function(c){ walk(byId[c]); }); }
    else vis.push(n);
  }
  TD.nodes.filter(function(n){ return n.parent===null; }).forEach(walk);
  var idset={}; vis.forEach(function(n){ idset[n.id]=1; });
  var es=TD.edges.filter(function(e){ return idset[e.s]&&idset[e.t]; });
  return {nodes:vis, edges:es};
}
function build(){
  var raw = view==='tree' ? treeVisible() : GD[view];
  var map={};
  N = raw.nodes.map(function(nd,i){ map[nd.id]=i; return {
      ref:nd.id, label:nd.label, title:nd.title||'', n:nd.n, grp:nd.grp,
      depth:nd.depth||0, ex:nd.ex||[], u:nd.u||'', pdf:nd.pdf||'', q:nd.q||'',
      kids:(view==='tree'?(TD.kids[nd.id]||[]).length:0),
      x:0,y:0,r:6,deg:0,fx:null,fy:null}; });
  maxw=1;
  E = raw.edges.map(function(e){ if(e.w>maxw) maxw=e.w;
      return {s:map[e.s],t:map[e.t],w:e.w,j:e.j}; })
      .filter(function(e){ return e.s!==undefined&&e.t!==undefined; });
  E.forEach(function(e){ N[e.s].deg++; N[e.t].deg++; });
  var mn=1e9,mx=0; N.forEach(function(n){ if(n.n<mn)mn=n.n; if(n.n>mx)mx=n.n; });
  N.forEach(function(n){ n.r=5+11*Math.sqrt((n.n-mn)/Math.max(mx-mn,1)); });
}

/* -------- static radial sector layout: deterministic, cannot oscillate ------ */
function layout(){
  var gs={}; N.forEach(function(n){ (gs[n.grp]=gs[n.grp]||[]).push(n); });
  var keys=Object.keys(gs).sort(function(a,b){
    return gs[b].length-gs[a].length || (a<b?-1:1); });
  var total=N.length||1, ang=-Math.PI/2, GAP=0.06;
  var R=Math.min(W,Hh)*0.42;
  keys.forEach(function(k){
    var arr=gs[k], span=Math.max((arr.length/total)*Math.PI*2-GAP, 0.10);
    arr.sort(function(a,b){ return (a.depth-b.depth)||(b.n-a.n)||(a.label<b.label?-1:1); });
    var depths={}; arr.forEach(function(n){ depths[n.depth]=1; });
    var dk=Object.keys(depths).sort();
    if(dk.length>1){                       // rings by depth (tree view)
      dk.forEach(function(d,di){
        var row=arr.filter(function(n){ return String(n.depth)===d; });
        var rad=R*(0.42+0.29*di);
        row.forEach(function(n,i){
          var a=ang+span*((i+0.5)/row.length);
          n.x=Math.cos(a)*rad; n.y=Math.sin(a)*rad; });
      });
    } else {                               // rings by index
      var rings=Math.max(1,Math.round(Math.sqrt(arr.length/2.2)));
      arr.forEach(function(n,i){
        var ring=i%rings, slot=Math.floor(i/rings), per=Math.ceil(arr.length/rings);
        /* A single ring sits near the rim rather than at 0.46R, so the top-level
           view fills the canvas instead of huddling in the middle. */
        var rad=R*(rings===1 ? 0.88 : 0.50+0.45*ring/(rings-1));
        var a=ang+span*((slot+0.5)/per);
        n.x=Math.cos(a)*rad; n.y=Math.sin(a)*rad; });
    }
    ang+=span+GAP;
  });
  N.forEach(function(n){ if(n.fx!==null){ n.x=n.fx; n.y=n.fy; } });
}

/* ---------------- drawing ---------------- */
function trim(s){                        // shorten on a word boundary, not mid-word
  if(s.length<=40) return s;
  var c=s.slice(0,40), i=c.lastIndexOf(' ');
  return (i>22?c.slice(0,i):c)+'\u2026'; }
function tx(n){ return n.x*cam.k+W/2+cam.x; }
function ty(n){ return n.y*cam.k+Hh/2+cam.y; }
function draw(){
  ctx.clearRect(0,0,W,Hh);
  var nb=null;
  if(sel!==null&&N[sel]){ nb={}; E.forEach(function(e){ if(e.w<minw)return;
    if(e.s===sel)nb[e.t]=1; if(e.t===sel)nb[e.s]=1; }); nb[sel]=1; }
  for(var i=0;i<E.length;i++){ var e=E[i]; if(e.w<minw) continue;
    var a=N[e.s], b=N[e.t], cross=a.grp!==b.grp;
    var on=!nb||(nb[e.s]&&nb[e.t]);
    ctx.globalAlpha= on?(cross?0.30:0.11)+0.42*e.w/maxw : 0.025;
    ctx.strokeStyle= (nb&&on)?'#ffb302':(cross?'#7fa8d8':'#5d6675');
    ctx.lineWidth=Math.max(0.4,(0.35+2.3*e.w/maxw)*cam.k);
    ctx.beginPath(); ctx.moveTo(tx(a),ty(a)); ctx.lineTo(tx(b),ty(b)); ctx.stroke(); }
  ctx.globalAlpha=1;
  var q=(document.getElementById('q').value||'').toLowerCase();
  var lab=[];
  N.forEach(function(n,i){
    var on=!nb||nb[i], hit=q&&(n.label+' '+n.title).toLowerCase().indexOf(q)>=0;
    ctx.globalAlpha=on?1:0.14;
    ctx.beginPath(); ctx.arc(tx(n),ty(n),n.r*cam.k,0,6.2832);
    ctx.fillStyle=col(n.grp); ctx.fill();
    if(n.kids&&!open[n.ref]){ ctx.lineWidth=1.6*cam.k; ctx.strokeStyle='#fff8'; ctx.stroke(); }
    if(hit||i===sel){ ctx.lineWidth=2.6; ctx.strokeStyle=hit?'#fff':'#ffb302'; ctx.stroke(); }
    if(cam.k>0.62)
      lab.push({n:n,on:on,p:(i===sel?300:hit?200:(nb&&nb[i])?100:0)+n.r}); });
  /* Labels are placed after every node is drawn, so a label is never painted
     under a later circle. Each sits on the outward side of its own node, and
     a collision nudges it clear; a label that cannot be placed within DRIFT
     of its node is dropped rather than printed over its neighbour. */
  var DRIFT=26, boxes=[];
  lab.sort(function(a,b){ return b.p-a.p; });
  ctx.textBaseline='middle';
  lab.forEach(function(L){
    var n=L.n, fs=10.5*Math.min(cam.k,1.6), hh=fs*0.62;
    ctx.font=fs+'px -apple-system,sans-serif';
    var t=trim(n.label), w=ctx.measureText(t).width, gap=n.r*cam.k+5;
    var x=n.x<0 ? tx(n)-gap-w : tx(n)+gap, y0=ty(n), y=y0, clash=true;
    for(var k=0;k<8&&clash;k++){
      clash=false;
      for(var j=0;j<boxes.length;j++){ var o=boxes[j];
        if(x-3<o.r&&x+w+3>o.l&&y-hh<o.b&&y+hh>o.t){
          y = (y<=(o.t+o.b)/2) ? o.t-hh-1.5 : o.b+hh+1.5; clash=true; break; } } }
    if(clash||Math.abs(y-y0)>DRIFT) return;
    boxes.push({l:x-3,r:x+w+3,t:y-hh,b:y+hh});
    ctx.globalAlpha=L.on?0.94:0.1; ctx.fillStyle='#e8eaf0';
    ctx.fillText(t,x,y); });
  ctx.textBaseline='alphabetic';
  ctx.globalAlpha=1;
}
function pick(mx,my){ var best=null,bd=1e9;
  N.forEach(function(n,i){ var dx=tx(n)-mx, dy=ty(n)-my, d=dx*dx+dy*dy, rr=n.r*cam.k+7;
    if(d<Math.max(100,rr*rr)&&d<bd){bd=d;best=i;} });
  return best; }

/* ---------------- panel ---------------- */
function info(){
  var el=document.getElementById('sel');
  if(sel===null||!N[sel]){ el.innerHTML='<span style="color:var(--mut)">Click a node '+
    'for detail and links.'+(view==='tree'?' Click again to split it into sub-fields.':'')+
    '</span>'; return; }
  var n=N[sel], links='';
  if(n.u) links='<div class="lk"><a href="'+esc(n.u)+'" target="_blank" rel="noopener">open this thesis ↗</a>'+
    (n.pdf&&n.pdf!==n.u?' &middot; <a href="'+esc(n.pdf)+'" target="_blank" rel="noopener">PDF</a>':'')+'</div>';
  else if(n.q) links='<div class="lk"><a href="https://scholar.google.com/scholar?q='+
    encodeURIComponent(n.q)+'" target="_blank" rel="noopener">find this work ↗</a>'+
    '<span class="hint">search — cited works are not themselves in the corpus</span></div>';
  var ex=''; if(n.ex&&n.ex.length) ex='<div class="exh">example theses</div><div class="ex">'+
    n.ex.map(function(e){ return '<a href="'+esc(e.u)+'" target="_blank" rel="noopener">'+
      esc(e.t)+'</a>'; }).join('')+'</div>';
  var ns=[]; E.forEach(function(e){ if(e.w<minw)return;
    if(e.s===sel)ns.push([N[e.t],e.w]); else if(e.t===sel)ns.push([N[e.s],e.w]); });
  ns.sort(function(a,b){return b[1]-a[1];});
  el.innerHTML='<div class="t">'+esc(n.label)+
      (n.kids?'<span class="badge">'+(open[n.ref]?'open':'+'+n.kids)+'</span>':'')+'</div>'+
    (n.title?'<div style="color:var(--mut);font-size:11.5px">'+esc(n.title)+'</div>':'')+
    links+
    '<div class="kv"><span>group</span><span>'+esc(n.grp)+'</span></div>'+
    '<div class="kv"><span>'+(view==='cocite'?'citing theses':view==='tree'?'theses':'weight')+
      '</span><span>'+n.n.toLocaleString()+'</span></div>'+
    '<div class="kv"><span>links</span><span>'+ns.length+'</span></div>'+ex+
    '<div id="nbr">'+ns.slice(0,20).map(function(p,i){
      var cross=p[0].grp!==n.grp;
      return '<div data-i="'+N.indexOf(p[0])+'">'+(cross?'⇄ ':'')+esc(p[0].label.slice(0,30))+
        ' <b style="float:right;color:'+(cross?'var(--acc2)':'var(--acc)')+'">'+p[1]+'</b></div>'; }).join('')+
    '</div>';
  document.querySelectorAll('#nbr div').forEach(function(d){
    d.onclick=function(){ sel=+d.dataset.i; info(); draw(); }; });
}
function legend(){
  var c={}; N.forEach(function(n){ c[n.grp]=(c[n.grp]||0)+1; });
  var ks=Object.keys(c).sort(function(a,b){return c[b]-c[a];}).slice(0,16);
  document.getElementById('leg').innerHTML=ks.map(function(k){
    return '<div><i style="background:'+col(k)+'"></i>'+esc(k)+' ('+c[k]+')</div>'; }).join('');
}
function refresh(keepSel){
  var prev = keepSel&&sel!==null&&N[sel] ? N[sel].ref : null;
  build(); layout();
  sel = prev===null?null:(function(){ for(var i=0;i<N.length;i++) if(N[i].ref===prev) return i;
    return null; })();
  document.getElementById('kn').textContent=N.length.toLocaleString();
  document.getElementById('ke').textContent=E.length.toLocaleString();
  legend(); info(); draw();
}
function load(v){
  view=v; sel=null; groups={}; ng=0; cam={x:0,y:0,k:1};
  document.querySelectorAll('.tab').forEach(function(b){
    b.setAttribute('aria-selected',String(b.dataset.v===v)); });
  var meta=VIEWS.filter(function(x){return x[0]===v;})[0];
  document.getElementById('vtitle').textContent=meta[1];
  document.getElementById('vdesc').textContent=meta[2];
  document.getElementById('treectl').style.display = v==='tree'?'inline-flex':'none';
  refresh(false);
}
/* ---------------- interaction ---------------- */
cv.addEventListener('mousemove',function(ev){
  var r=cv.getBoundingClientRect(), mx=ev.clientX-r.left, my=ev.clientY-r.top;
  if(drag!==null){ N[drag].fx=N[drag].x=(mx-W/2-cam.x)/cam.k;
    N[drag].fy=N[drag].y=(my-Hh/2-cam.y)/cam.k; draw(); return; }
  if(pan){ cam.x+=mx-pan.x; cam.y+=my-pan.y; pan={x:mx,y:my}; draw(); return; }
  var i=pick(mx,my);
  if(i!==null){ var n=N[i]; tip.style.display='block';
    tip.style.left=Math.min(mx+14,W-300)+'px'; tip.style.top=(my+10)+'px';
    tip.innerHTML='<b>'+esc(n.label)+'</b>'+(n.title?'<br>'+esc(n.title):'')+
      '<br>'+esc(n.grp)+' · '+n.n.toLocaleString()+' · '+n.deg+' links'+
      (n.kids?'<br><i>click to '+(open[n.ref]?'collapse':'split into '+n.kids)+'</i>':''); }
  else tip.style.display='none'; });
cv.addEventListener('mousedown',function(ev){
  var r=cv.getBoundingClientRect(), mx=ev.clientX-r.left, my=ev.clientY-r.top;
  var i=pick(mx,my); cv.classList.add('drag');
  if(i!==null){ if(sel===i&&view==='tree'&&N[i].kids){
      open[N[i].ref]=!open[N[i].ref]; refresh(true); return; }
    sel=i; drag=i; info(); draw(); }
  else pan={x:mx,y:my}; });
window.addEventListener('mouseup',function(){ drag=null; pan=null; cv.classList.remove('drag'); });
cv.addEventListener('wheel',function(ev){ ev.preventDefault();
  cam.k=Math.max(0.2,Math.min(7,cam.k*(ev.deltaY<0?1.12:0.893))); draw(); },{passive:false});
cv.addEventListener('dblclick',function(ev){
  var r=cv.getBoundingClientRect(), i=pick(ev.clientX-r.left,ev.clientY-r.top);
  if(i!==null){ N[i].fx=N[i].fy=null; layout(); draw(); } else { sel=null; info(); draw(); } });
document.querySelectorAll('.tab').forEach(function(b){ b.onclick=function(){ load(b.dataset.v); }; });
document.getElementById('q').addEventListener('input',draw);
document.getElementById('mw').addEventListener('input',function(){
  minw=+this.value; document.getElementById('mwv').textContent=minw; info(); draw(); });
document.getElementById('depth').addEventListener('input',function(){
  var d=+this.value; document.getElementById('depthv').textContent=d;
  open={}; if(d>0){ TD.nodes.forEach(function(n){ if(n.depth<d) open[n.id]=1; }); }
  refresh(false); });
document.getElementById('fit').onclick=function(){ cam={x:0,y:0,k:1};
  N.forEach(function(n){ n.fx=n.fy=null; }); layout(); draw(); };
resize(); load('tree');
})();
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--public", action="store_true",
                    help="build the publishable bundle: drops the text-reuse view, "
                         "which names individual theses as linked by overlap. Those "
                         "are unreviewed screening signals about identifiable "
                         "people and do not belong on a public URL.")
    args = ap.parse_args()
    global VIEWS, G, OUT
    if args.public:
        VIEWS = [v for v in VIEWS if v[0] != "reuse"]
        G = {k: v for k, v in G.items() if k != "reuse"}
        OUT = PUB
    # The reader is only published beside the public bundle, so the cross-link
    # is emitted there and nowhere else — a local build must not offer a 404.
    xlink = ('<a class="mini" href="bridges.html" style="text-decoration:none;'
             'line-height:1.9">&#8644; where fields cross over</a>'
             if args.public else "")
    payload = json.dumps(G, separators=(",", ":"), sort_keys=True).replace("</", "<\\/")
    tpay = json.dumps(T, separators=(",", ":"), sort_keys=True).replace("</", "<\\/")
    tabs = "".join(f'<button class="tab" data-v="{k}">{t}</button>' for k, t, _ in VIEWS)
    html = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Thesis corpus graph</title><style>{CSS}</style></head><body>
<header>
  <div class="brand">Thesis corpus graph
    <small>24,656 UK doctoral theses &middot; White Rose eTheses</small></div>
  <div class="tabs">{tabs}</div>
  <div class="ctl">
    <label id="treectl">granularity
      <input type="range" id="depth" min="0" max="2" step="1" value="0">
      <b id="depthv">0</b></label>
    <label>min edge <input type="range" id="mw" min="0" max="80" value="0">
      <b id="mwv">0</b></label>
    <label><input type="search" id="q" placeholder="search"></label>
    {xlink}<button class="mini" id="fit">reset view</button>
  </div>
</header>
<main>
  <div id="wrap"><canvas id="cv"></canvas><div id="tip"></div></div>
  <aside>
    <h3 id="vtitle"></h3><p class="d" id="vdesc"></p>
    <div class="kv"><span>nodes</span><span id="kn"></span></div>
    <div class="kv"><span>edges</span><span id="ke"></span></div>
    <div id="sel"></div>
    <div class="legend" id="leg"></div>
    <p class="note">Wedges are disciplines; links crossing the centre are
    cross-discipline and drawn brighter. Drag to pan, scroll to zoom, drag a node
    to move it, double-click it to put it back. Layout is static and
    deterministic — the same data always gives the same picture.<br><br>
    Built from openly deposited theses in White Rose eTheses Online. Titles,
    metadata and links are public bibliographic facts; no thesis text is
    reproduced here. Method and source:
    <a href="https://github.com/jhammant/thesisgraph" target="_blank" rel="noopener" style="color:var(--acc)">source &amp; method on GitHub</a>.</p>
  </aside>
</main>
<script type="application/json" id="p">{payload}</script>
<script type="application/json" id="t">{tpay}</script>
<script>window.__G__=JSON.parse(document.getElementById('p').textContent);
window.__T__=JSON.parse(document.getElementById('t').textContent);
var VIEWS={json.dumps([[k, t, d] for k, t, d in VIEWS])};</script>
<script>{JS}</script>
</body></html>"""
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(html, encoding="utf-8")
    print(f"written {OUT} ({OUT.stat().st_size/1024:.0f} KB)")


if __name__ == "__main__":
    main()
