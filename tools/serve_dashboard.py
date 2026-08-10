"""Live training dashboard -- stdlib-only HTTP server (no dependencies).

Auto-discovers every run folder that has a metrics.jsonl (e.g. ckpt = phase-1,
ckpt2 = phase-2) and shows one TAB per run, plus a separate "data prep" tab that
tracks every data_prep memmap (data/*.manifest.json). The page polls every 3s so
scores/tokens appear as soon as they are flushed.

    python tools/serve_dashboard.py --port 8000        # then open http://localhost:8000
"""
import argparse
import glob
import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

DEFAULT_TAB = ""   # set from --metrics dir; the tab selected on first load


# ---------- data access ----------
RUNS_DIR = "checkpoints"


def discover_runs():
    """Every checkpoint folder holding a metrics.jsonl becomes a tab."""
    return sorted(os.path.basename(os.path.dirname(p))
                  for p in glob.glob(os.path.join(RUNS_DIR, "*", "metrics.jsonl")))


def read_metrics(run):
    meta, train, ev = {}, {}, {}
    try:
        with open(os.path.join(RUNS_DIR, run, "metrics.jsonl")) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    o = json.loads(line)
                except json.JSONDecodeError:
                    continue
                t = o.get("t")
                if t == "meta":
                    meta.update(o)
                elif t == "train":
                    train[o["step"]] = o
                elif t == "eval":
                    ev[o["step"]] = o
    except FileNotFoundError:
        pass
    return {"meta": meta,
            "train": [train[s] for s in sorted(train)],
            "eval": [ev[s] for s in sorted(ev)]}


def _python_procs():
    """cmdline arg-lists of running `python ...` processes (excludes shells)."""
    procs = []
    for cmd in glob.glob("/proc/*/cmdline"):
        try:
            with open(cmd, "rb") as f:
                parts = [p for p in f.read().split(b"\x00") if p]
        except OSError:
            continue
        if parts and parts[0].rsplit(b"/", 1)[-1].startswith(b"python"):
            procs.append([p.decode(errors="ignore") for p in parts])
    return procs


def active_run():
    """Which run folder is being written right now: train.py --ckpt-dir, or
    finetune_sst2.py --run-dir (which defaults to ckpt_sst2 when not passed)."""
    for args in _python_procs():
        if any(a.endswith("train.py") or a == "mini_enc_transformer.training.pretrain" for a in args) \
                and "--ckpt-dir" in args:
            v = args[args.index("--ckpt-dir") + 1]
            return os.path.basename(v.rstrip("/"))
        if any(a.endswith("finetune_sst2.py") or a == "mini_enc_transformer.training.finetune_sst2"
               for a in args):
            v = args[args.index("--run-dir") + 1] if "--run-dir" in args else "checkpoints/ckpt_sst2"
            return os.path.basename(v.rstrip("/"))
    return None


def prep_alive():
    return any(a.endswith("prep.py") or a == "mini_enc_transformer.data.prep"
               for args in _python_procs() for a in args)


def read_preps():
    preps = []
    for mp in sorted(glob.glob("data/*.manifest.json")):
        try:
            with open(mp) as f:
                m = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        preps.append({"name": os.path.basename(mp)[:-len(".manifest.json")],
                      "tokens": m.get("tokens_written", 0), "docs": m.get("docs_seen", 0),
                      "target": m.get("target_tokens", 0), "dataset": m.get("dataset", "")})
    return {"preps": preps, "alive": prep_alive()}


# ---------- page ----------
PAGE = r"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>training dashboard</title>
<style>
  :root{--bg:#0b0e14;--panel:#151a23;--fg:#e6e9ef;--muted:#8b93a7;--line:#2a3140;
        --accent:#5aa9ff;--accent2:#ffb454;--good:#4ec9a5;--warn:#c08cff}
  @media (prefers-color-scheme: light){:root{--bg:#f5f7fa;--panel:#fff;--fg:#1a1f2b;
        --muted:#5b6472;--line:#e2e6ee;--accent:#2f7ee0;--accent2:#e08a1e;--good:#1f9d78;--warn:#7b3fd4}}
  *{box-sizing:border-box} body{margin:0;background:var(--bg);color:var(--fg);
    font:14px/1.5 -apple-system,Segoe UI,Roboto,sans-serif}
  .wrap{max-width:1100px;margin:0 auto;padding:22px}
  h1{font-size:18px;margin:0 0 14px}
  .tabs{display:flex;gap:6px;flex-wrap:wrap;margin-bottom:16px;border-bottom:1px solid var(--line)}
  .tab{padding:8px 15px;cursor:pointer;border:1px solid transparent;border-bottom:none;border-radius:9px 9px 0 0;
       color:var(--muted);font-weight:600;font-size:13px;user-select:none}
  .tab:hover{color:var(--fg)}
  .tab.sel{background:var(--panel);border-color:var(--line);color:var(--fg)}
  .tab .dot{display:inline-block;width:7px;height:7px;border-radius:50%;margin-left:6px;vertical-align:middle}
  .live{background:var(--good)} .idle{background:var(--muted)}
  .pill{display:inline-block;padding:2px 10px;border-radius:999px;font-size:12px;font-weight:600;margin-left:8px}
  .pl-live{background:rgba(78,201,165,.15);color:var(--good)} .pl-paused{background:rgba(255,180,84,.15);color:var(--accent2)}
  .pl-done{background:rgba(90,169,255,.15);color:var(--accent)}
  .tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:12px}
  .tile{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:12px 14px}
  .tile .k{color:var(--muted);font-size:12px} .tile .v{font-size:21px;font-weight:650;margin-top:3px;font-variant-numeric:tabular-nums}
  .bar{height:7px;background:var(--line);border-radius:6px;overflow:hidden;margin-top:9px}
  .bar>i{display:block;height:100%;background:var(--accent);width:0}
  .charts{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-top:14px}
  @media(max-width:760px){.charts{grid-template-columns:1fr}}
  .card{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:14px}
  .card h2{font-size:13px;margin:0 0 8px;color:var(--muted);font-weight:600}
  svg{width:100%;height:230px;display:block} .lgd{font-size:12px;color:var(--muted);margin-top:6px}
  .dot2{display:inline-block;width:9px;height:9px;border-radius:2px;margin:0 5px 0 12px;vertical-align:middle}
  .sub{color:var(--muted);font-size:13px}
</style></head><body><div class="wrap">
<h1>training dashboard <span id="foot" class="sub"></span></h1>
<div class="tabs" id="tabs"></div>
<div id="view"></div>
</div>
<script>
const W=500,H=230,P={l:44,r:12,t:12,b:26};
let TAB=null, DEFAULT_TAB="__DEFAULT__";
const prevPrep={};
function sc(v,lo,hi,a,b){return hi===lo?(a+b)/2:a+(v-lo)/(hi-lo)*(b-a);}
function fmt(n,d=3){return n==null?"–":Number(n).toFixed(d);}
function draw(svg, series, xmax, yfix){
  const pts=[].concat(...series.map(s=>s.data));
  if(!pts.length){svg.innerHTML='<text x="12" y="24" fill="var(--muted)" font-size="12">no data yet</text>';return;}
  let ylo=yfix?yfix[0]:Math.min(...pts.map(p=>p.y)), yhi=yfix?yfix[1]:Math.max(...pts.map(p=>p.y));
  const pad=(yhi-ylo)*0.08||0.1; if(!yfix){ylo-=pad;yhi+=pad;}
  const X=x=>sc(x,0,xmax,P.l,W-P.r), Y=y=>sc(y,ylo,yhi,H-P.b,P.t);
  let g="";
  for(let i=0;i<=4;i++){const yy=ylo+(yhi-ylo)*i/4,py=Y(yy);
    g+=`<line x1="${P.l}" y1="${py}" x2="${W-P.r}" y2="${py}" stroke="var(--line)"/>`;
    g+=`<text x="${P.l-6}" y="${py+3}" text-anchor="end" fill="var(--muted)" font-size="10">${fmt(yy,yhi<=1?2:1)}</text>`;}
  for(let i=0;i<=4;i++){const px=X(xmax*i/4);
    g+=`<text x="${px}" y="${H-8}" text-anchor="middle" fill="var(--muted)" font-size="10">${Math.round(xmax*i/4)}</text>`;}
  for(const s of series){ if(!s.data.length)continue;
    g+=`<path d="${s.data.map((p,i)=>(i?'L':'M')+X(p.x)+' '+Y(p.y)).join(' ')}" fill="none" stroke="${s.color}" stroke-width="2"/>`;
    const L=s.data[s.data.length-1]; g+=`<circle cx="${X(L.x)}" cy="${Y(L.y)}" r="3" fill="${s.color}"/>`;}
  svg.innerHTML=g;
}
function tile(k,v){return `<div class="tile"><div class="k">${k}</div><div class="v">${v}</div></div>`;}

async function loadTabs(){
  const r=await (await fetch('/api/runs')).json();
  if(TAB===null) TAB = (DEFAULT_TAB && r.runs.includes(DEFAULT_TAB)) ? DEFAULT_TAB : (r.active || r.runs[r.runs.length-1] || '__prep__');
  const bar=document.getElementById('tabs'); let h="";
  for(const run of r.runs){
    const live = run===r.active;
    h+=`<div class="tab ${run===TAB?'sel':''}" onclick="TAB='${run}';render()">${run}<span class="dot ${live?'live':'idle'}"></span></div>`;
  }
  h+=`<div class="tab ${TAB==='__prep__'?'sel':''}" onclick="TAB='__prep__';render()">data prep</div>`;
  bar.innerHTML=h;
  return r;
}
async function renderRun(run, active){
  const m=await (await fetch('/api/metrics?run='+encodeURIComponent(run))).json();
  const meta=m.meta||{}, tr=m.train||[], ev=m.eval||[];
  const maxs=meta.max_steps|| (tr.length?tr[tr.length-1].step:1);
  const cur=tr.length?tr[tr.length-1]:{step:0,loss:null,lr:null,tokens:0};
  const lastEv=ev.length?ev[ev.length-1]:{};
  const done=cur.step>=maxs && maxs>1;
  const pill = done?['DONE','pl-done']:(run===active?['LIVE','pl-live']:['PAUSED','pl-paused']);
  let rate=null,eta="–"; const timed=tr.filter(p=>p.time);
  if(timed.length>=2){const a=timed[timed.length-2],b=timed[timed.length-1];const d=(b.step-a.step)/(b.time-a.time);
    if(d>0){rate=d*60;const rem=(maxs-b.step)/d;eta=rem>3600?(rem/3600).toFixed(1)+'h':Math.max(0,Math.round(rem/60))+'m';}}
  const pct=100*cur.step/maxs;
  // JEPA runs log collapse telemetry; MLM runs don't. Show the extra panels only there.
  const jepa = ev.length>0 && ev[ev.length-1].eff_rank!=null;
  // Runs carrying a grafted layer log per-layer health so the recovery is watchable
  // while it happens. ENT_MAX = log(seq_len) is the uniform-attention ceiling: a head
  // sitting there is averaging, not selecting.
  const graft = ev.length>0 && ev[ev.length-1].l4_np!=null;
  const PL = (meta.probe_layers&&meta.probe_layers.length)?meta.probe_layers:[4,5];
  const ENT_MAX = Math.log(meta.seq_len||128);
  const GC = ['var(--accent)','var(--accent2)','var(--warn)','var(--good)'];
  document.getElementById('view').innerHTML=
    `<div class="sub" style="margin-bottom:10px">${run} <span class="pill ${pill[1]}">${pill[0]}</span> · `+
    `${meta.dataset||'?'} · ${meta.params_M||'?'}M params · eff batch ${meta.eff_batch||'?'}</div>`+
    `<div class="tiles">
       <div class="tile"><div class="k">step</div><div class="v">${cur.step} / ${maxs}</div><div class="bar"><i style="width:${pct}%"></i></div></div>`+
       tile('train loss',fmt(cur.loss))+tile('val loss',fmt(lastEv.val_loss))+
       tile(jepa?'cos(pred,tgt)':'masked acc',lastEv.masked_acc!=null?(100*lastEv.masked_acc).toFixed(1)+'%':'–')+
       tile('lr',cur.lr!=null?cur.lr.toExponential(2):'–')+
       (jepa?tile('eff rank',fmt(lastEv.eff_rank,1))+tile('target std',fmt(lastEv.target_std))+
             tile('cos to mean',fmt(lastEv.cos_to_mean)):'')+
       (graft?PL.map(L=>tile('L'+L+' never+',lastEv['l'+L+'_np']!=null?
              (100*lastEv['l'+L+'_np']).toFixed(0)+'%':'–')+
              tile('L'+L+' attn ent',fmt(lastEv['l'+L+'_ent'],2))).join(''):'')+
       tile('tokens seen',cur.tokens?(cur.tokens/1e6).toFixed(0)+'M':'–')+
       tile('steps/min',rate?rate.toFixed(0):'–')+tile('eta',eta)+
    `</div>
     <div class="charts">
       <div class="card"><h2>loss vs step</h2><svg id="c1" viewBox="0 0 ${W} ${H}"></svg>
         <div class="lgd"><span class="dot2" style="background:var(--accent)"></span>train<span class="dot2" style="background:var(--accent2)"></span>val</div></div>
       <div class="card"><h2>masked-token accuracy vs step</h2><svg id="c2" viewBox="0 0 ${W} ${H}"></svg>
         <div class="lgd"><span class="dot2" style="background:var(--good)"></span>masked_acc</div></div>
       <div class="card"><h2>learning rate vs step</h2><svg id="c3" viewBox="0 0 ${W} ${H}"></svg>
         <div class="lgd"><span class="dot2" style="background:var(--warn)"></span>lr`+
         (tr.some(p=>p.lr_boost!=null)?`<span class="dot2" style="background:var(--accent2)"></span>lr (grafted layers)`:'')+
        `</div></div>`+
       (graft?`<div class="card"><h2>grafted layers — FFN never-positive</h2><svg id="c6" viewBox="0 0 ${W} ${H}"></svg>
         <div class="lgd">`+PL.map((L,i)=>`<span class="dot2" style="background:${GC[i%4]}"></span>L${L}`).join('')+
        ` &nbsp;1.0 = the whole FFN is switched off</div></div>
       <div class="card"><h2>grafted layers — attention entropy</h2><svg id="c7" viewBox="0 0 ${W} ${H}"></svg>
         <div class="lgd">`+PL.map((L,i)=>`<span class="dot2" style="background:${GC[i%4]}"></span>L${L}`).join('')+
        ` &nbsp;${ENT_MAX.toFixed(2)} = uniform, i.e. averaging not selecting</div></div>`:'')+
       (jepa?`<div class="card"><h2>collapse watch — effective rank</h2><svg id="c4" viewBox="0 0 ${W} ${H}"></svg>
         <div class="lgd"><span class="dot2" style="background:var(--accent)"></span>eff_rank (falling = collapsing)</div></div>
       <div class="card"><h2>collapse watch — spread</h2><svg id="c5" viewBox="0 0 ${W} ${H}"></svg>
         <div class="lgd"><span class="dot2" style="background:var(--good)"></span>target_std<span class="dot2" style="background:var(--accent2)"></span>cos_to_mean (→1 = collapsing)</div></div>`:'')+
     `</div>`;
  draw(document.getElementById('c1'),
    [{color:'var(--accent)',data:tr.map(p=>({x:p.step,y:p.loss}))},
     {color:'var(--accent2)',data:ev.map(p=>({x:p.step,y:p.val_loss}))}], maxs, null);
  draw(document.getElementById('c2'),
    [{color:'var(--good)',data:ev.map(p=>({x:p.step,y:p.masked_acc}))}], maxs, [0,1]);
  // LR spans warmup->0, so pin the floor at 0 to keep the cosine shape readable.
  if(jepa){
    draw(document.getElementById('c4'),
      [{color:'var(--accent)',data:ev.filter(p=>p.eff_rank!=null).map(p=>({x:p.step,y:p.eff_rank}))}], maxs, null);
    draw(document.getElementById('c5'),
      [{color:'var(--good)',data:ev.filter(p=>p.target_std!=null).map(p=>({x:p.step,y:p.target_std}))},
       {color:'var(--accent2)',data:ev.filter(p=>p.cos_to_mean!=null).map(p=>({x:p.step,y:p.cos_to_mean}))}], maxs, [0,1]);
  }
  if(graft){
    draw(document.getElementById('c6'),
      PL.map((L,i)=>({color:GC[i%4],
        data:ev.filter(p=>p['l'+L+'_np']!=null).map(p=>({x:p.step,y:p['l'+L+'_np']}))})),
      maxs, [0,1]);
    draw(document.getElementById('c7'),
      PL.map((L,i)=>({color:GC[i%4],
        data:ev.filter(p=>p['l'+L+'_ent']!=null).map(p=>({x:p.step,y:p['l'+L+'_ent']}))})),
      maxs, [0,ENT_MAX]);
  }
  draw(document.getElementById('c3'),
    [{color:'var(--warn)',data:tr.filter(p=>p.lr!=null).map(p=>({x:p.step,y:p.lr}))},
     {color:'var(--accent2)',data:tr.filter(p=>p.lr_boost!=null).map(p=>({x:p.step,y:p.lr_boost}))}],
    maxs, [0, Math.max(...tr.map(p=>p.lr||0), 1e-12)]);
}
async function renderPrep(){
  const r=await (await fetch('/api/preps')).json();
  const now=Date.now()/1000;
  let h='<div class="sub" style="margin-bottom:10px">data prep — all memmaps '+
        (r.alive?'<span class="pill pl-live">A PREP RUNNING</span>':'<span class="pill pl-paused">IDLE</span>')+'</div>';
  if(!r.preps.length) h+='<div class="card">no data_prep manifests found</div>';
  for(const p of r.preps){
    const tgt=p.target||1, pct=100*Math.min(1,p.tokens/tgt);
    let rate=null; if(prevPrep[p.name] && p.tokens>prevPrep[p.name].tok) rate=(p.tokens-prevPrep[p.name].tok)/(now-prevPrep[p.name].t);
    prevPrep[p.name]={tok:p.tokens,t:now};
    h+=`<div class="card" style="margin-bottom:12px"><h2>${p.name} <span class="sub">${p.dataset||''}</span></h2>
        <div class="bar" style="height:9px"><i style="width:${pct}%;background:var(--accent2)"></i></div>
        <div class="lgd" style="margin-top:8px"><b>${(p.tokens/1e6).toFixed(1)}M</b> / ${(tgt/1e6).toFixed(0)}M tokens · ${p.docs.toLocaleString()} docs · ${rate?(rate/1e3).toFixed(0)+'K tok/s':'…'}</div></div>`;
  }
  document.getElementById('view').innerHTML=h;
}
async function render(){
  const r=await loadTabs();
  if(TAB==='__prep__') await renderPrep(); else await renderRun(TAB, r.active);
  document.getElementById('foot').textContent='· updated '+new Date().toLocaleTimeString()+' · auto-refresh 3s';
}
render(); setInterval(render, 3000);
</script></body></html>"""


class Handler(BaseHTTPRequestHandler):
    def _send(self, body, ctype="application/json"):
        b = body.encode() if isinstance(body, str) else body
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(b)))
        # Without this the browser caches the page and keeps rendering an old build
        # of the JS after the dashboard is updated -- looks like the change did nothing.
        self.send_header("Cache-Control", "no-store, must-revalidate")
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        u = urlparse(self.path)
        if u.path == "/api/runs":
            self._send(json.dumps({"runs": discover_runs(), "active": active_run()}))
        elif u.path == "/api/metrics":
            run = parse_qs(u.query).get("run", [""])[0]
            self._send(json.dumps(read_metrics(run)) if run in discover_runs() else json.dumps({}))
        elif u.path == "/api/preps":
            self._send(json.dumps(read_preps()))
        else:
            self._send(PAGE.replace("__DEFAULT__", DEFAULT_TAB), "text/html; charset=utf-8")

    def log_message(self, *a):
        pass


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--metrics", default="checkpoints/ckpt2/metrics.jsonl", help="default tab = this run's folder")
    ap.add_argument("--prep-manifest", default=None, help="(accepted for compatibility; preps auto-discovered)")
    a = ap.parse_args()
    DEFAULT_TAB = os.path.basename(os.path.dirname(a.metrics)) if a.metrics else ""
    print(f"dashboard: http://localhost:{a.port}  (default tab: {DEFAULT_TAB or 'auto'})", flush=True)
    ThreadingHTTPServer(("127.0.0.1", a.port), Handler).serve_forever()
