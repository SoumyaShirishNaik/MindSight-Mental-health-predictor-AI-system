"use strict";

/* ── Constants ───────────────────────────────── */
const LABEL_COLORS = {
  "Normal":"#4ade80","Depression":"#60a5fa","Anxiety":"#fb923c",
  "Bipolar Disorder":"#c084fc","PTSD":"#f87171","OCD":"#22d3ee","Stress":"#fbbf24"
};
const RANKS = ["🥇","🥈","🥉"];
const MOOD_LABELS = ["","Very Bad","Bad","Okay","Good","Great!"];
const MOOD_EMOJIS = ["","😞","😟","😐","🙂","😄"];
const RECO_META = {
  "Depression":      {tags:["Professional","Social","Self-care","Monitoring"],urgency:["urg-high","urg-mid","urg-low","urg-low"],crisis:true},
  "Anxiety":         {tags:["Technique","Therapy","Lifestyle","Professional"],urgency:["urg-high","urg-high","urg-low","urg-mid"],crisis:false},
  "Bipolar Disorder":{tags:["Professional","Lifestyle","Monitoring","Avoid"],urgency:["urg-high","urg-mid","urg-low","urg-mid"],crisis:true},
  "PTSD":            {tags:["Therapy","Crisis","Technique","Self-compassion"],urgency:["urg-high","urg-high","urg-mid","urg-low"],crisis:true},
  "OCD":             {tags:["Therapy","Professional","Community","Technique"],urgency:["urg-high","urg-high","urg-mid","urg-mid"],crisis:false},
  "Stress":          {tags:["Identify","Technique","Boundaries","Recovery"],urgency:["urg-mid","urg-low","urg-mid","urg-low"],crisis:false},
  "Normal":          {tags:["Habits","Social","Physical","Routine"],urgency:["urg-low","urg-low","urg-low","urg-low"],crisis:false},
};
const URGENCY_LABEL={"urg-high":"Priority","urg-mid":"Helpful","urg-low":"Self-care"};
const NUM_COLORS=["rn-purple","rn-teal","rn-amber","rn-red"];

let currentPredId = null;
let fbRating = 3;
let moodScore = 1;

/* ── Particles ───────────────────────────────── */
(function(){
  const c=document.getElementById("particle-canvas");
  if(!c)return;
  const ctx=c.getContext("2d");
  let W,H,pts=[];
  function resize(){W=c.width=window.innerWidth;H=c.height=window.innerHeight}
  resize();
  window.addEventListener("resize",resize);
  for(let i=0;i<60;i++) pts.push({
    x:Math.random()*1920,y:Math.random()*1080,
    r:Math.random()*1.5+.3,vx:(Math.random()-.5)*.25,vy:(Math.random()-.5)*.25,
    a:Math.random()*.5+.1,c:["#7c6df0","#3dd6c8","#f06d9e"][Math.floor(Math.random()*3)]
  });
  function draw(){
    ctx.clearRect(0,0,W,H);
    pts.forEach(p=>{
      p.x+=p.vx;p.y+=p.vy;
      if(p.x<0)p.x=W;if(p.x>W)p.x=0;
      if(p.y<0)p.y=H;if(p.y>H)p.y=0;
      ctx.beginPath();ctx.arc(p.x,p.y,p.r,0,Math.PI*2);
      ctx.fillStyle=p.c+Math.round(p.a*255).toString(16).padStart(2,"0");
      ctx.fill();
    });
    requestAnimationFrame(draw);
  }
  draw();
})();

/* ── Nav ─────────────────────────────────────── */
document.querySelectorAll(".nav-link").forEach(l=>{
  l.addEventListener("click",()=>{
    document.querySelectorAll(".nav-link").forEach(x=>x.classList.remove("active"));
    l.classList.add("active");
  });
});

/* ── Char counter ────────────────────────────── */
const textInput=document.getElementById("text-input");
const charCount=document.getElementById("char-count");
textInput.addEventListener("input",()=>{
  const l=textInput.value.length;
  charCount.textContent=`${l} / 512`;
  charCount.classList.toggle("warn",l>450);
});
document.querySelectorAll(".qf-btn").forEach(b=>{
  b.addEventListener("click",()=>{textInput.value=b.dataset.text;textInput.dispatchEvent(new Event("input"))});
});

/* ── Health check ────────────────────────────── */
async function checkHealth(){
  try{
    const d=await fetch("/api/health").then(r=>r.json());
    const badge=document.getElementById("model-badge");
    if(d.model_ready){badge.innerHTML='<span class="badge-dot"></span>Model Ready ✓';badge.className="model-badge ready";}
    else{badge.innerHTML='<span class="badge-dot"></span>Demo Mode';badge.className="model-badge notready";}
  }catch{}
}

async function loadModelInfo(){
  try{
    const d=await fetch("/api/model_info").then(r=>r.json());
    if(d.test_accuracy) document.getElementById("model-acc").textContent=(d.test_accuracy*100).toFixed(1)+"%";
  }catch{}
}

/* ── Main analyze ────────────────────────────── */
const analyzeBtn=document.getElementById("analyze-btn");
analyzeBtn.addEventListener("click",runAnalysis);
textInput.addEventListener("keydown",e=>{if(e.ctrlKey&&e.key==="Enter")runAnalysis()});

async function runAnalysis(){
  const text=textInput.value.trim();
  if(text.length<10){textInput.style.borderColor="#f87171";textInput.focus();setTimeout(()=>textInput.style.borderColor="",1600);return;}
  analyzeBtn.disabled=true;
  analyzeBtn.querySelector(".btn-text").textContent="Analyzing…";
  try{
    const note=document.getElementById("user-note").value;
    const res=await fetch("/api/predict",{method:"POST",headers:{"Content-Type":"application/json"},
      body:JSON.stringify({text,note,session_id:"web-"+Date.now()})});
    const data=await res.json();
    if(data.error&&!data.prediction){alert("Error: "+data.error);return;}
    renderResults(data);
    if(data.pred_id) currentPredId=data.pred_id;
    loadStats();loadWeekly();
  }catch(e){alert("Network error – is the server running?")}
  finally{analyzeBtn.disabled=false;analyzeBtn.querySelector(".btn-text").textContent="Analyze";}
}

/* ── Render results ──────────────────────────── */
function renderResults(d){
  const sec=document.getElementById("results-section");
  sec.classList.remove("hidden");
  sec.scrollIntoView({behavior:"smooth",block:"start"});

  document.getElementById("rb-prediction").textContent=d.prediction;
  document.getElementById("rb-conf").textContent=d.confidence.toFixed(1)+"%";
  document.getElementById("rb-sev").textContent=d.severity+" Severity";
  if(d.inference_time_ms) document.getElementById("rb-time").textContent=d.inference_time_ms+"ms";
  document.getElementById("result-banner").style.borderColor=(d.color||"#7c6df0")+"55";

  document.getElementById("pred-label").textContent=d.prediction;
  document.getElementById("pred-desc").textContent=d.description;
  if(d.inference_time_ms) document.getElementById("inf-time").textContent="⚡ Inference: "+d.inference_time_ms+"ms";
  document.querySelector(".main-card").style.borderColor=(d.color||"#7c6df0")+"44";

  const circ=238.76;
  const offset=circ-(d.confidence/100)*circ;
  const ring=document.getElementById("ring-fill");
  ring.style.stroke=d.color;
  setTimeout(()=>ring.style.strokeDashoffset=offset,120);
  document.getElementById("conf-val").textContent=d.confidence.toFixed(1)+"%";

  const sevW={Low:28,Moderate:60,High:90};
  const sevC={Low:"#4ade80",Moderate:"#fb923c",High:"#f87171"};
  const sc=sevC[d.severity]||"#4ade80";
  document.getElementById("sev-text").textContent=d.severity;
  document.getElementById("sev-tag").textContent=d.severity;
  document.getElementById("sev-tag").style.cssText=`background:${sc}22;color:${sc};border:1px solid ${sc}44`;
  const sf=document.getElementById("sev-fill");
  sf.style.background=sc;
  setTimeout(()=>sf.style.width=(sevW[d.severity]||28)+"%",150);

  buildRecoHTML(d.recommendations,d.prediction,d.color);

  const sorted=Object.entries(d.distribution||{}).sort((a,b)=>b[1]-a[1]);
  document.getElementById("dist-bars").innerHTML=sorted.map(([label,prob])=>{
    const pct=(prob*100).toFixed(1);const col=LABEL_COLORS[label]||"#7c6df0";
    return `<div class="dist-row">
      <span class="dist-name">${label}</span>
      <div class="dist-track"><div class="dist-bar" style="width:0%;background:${col}" data-target="${pct}"></div></div>
      <span class="dist-pct" style="color:${col}">${pct}%</span></div>`;
  }).join("");
  setTimeout(()=>document.querySelectorAll(".dist-bar").forEach(b=>b.style.width=b.dataset.target+"%"),120);

  document.getElementById("top3-list").innerHTML=(d.top3||[]).map((item,i)=>`
    <div class="top3-item">
      <span class="top3-rank">${RANKS[i]}</span>
      <div class="top3-info">
        <div class="top3-name">${item.label}</div>
        <div class="top3-conf">${(item.confidence*100).toFixed(1)}% confidence</div>
        <div class="top3-bar" style="width:${(item.confidence*100).toFixed(0)}%;background:${item.color}"></div>
      </div>
      <div class="top3-dot" style="background:${item.color};box-shadow:0 0 8px ${item.color}88"></div>
    </div>`).join("");

  const tv=document.getElementById("token-viz");
  if(d.tokens&&d.tokens.length>0){
    tv.innerHTML=d.tokens.map((tok,i)=>{
      const score=d.token_scores[i]||0;
      const alpha=.12+score*.88;
      const bg=hexToRgba(d.color,alpha);
      const textCol=score>.5?"#fff":d.color;
      return `<span class="token-chip" style="background:${bg};border-color:${d.color}33;color:${textCol}" title="importance: ${(score*100).toFixed(0)}%">${tok}</span>`;
    }).join("");
  } else {
    tv.innerHTML=`<span style="color:var(--muted);font-size:.85rem;font-style:italic">Token importance available after model training.</span>`;
  }
}

/* ── Recommendations ─────────────────────────── */
function buildRecoHTML(recos,prediction,accentColor){
  const meta=RECO_META[prediction]||RECO_META["Normal"];
  const items=(recos||[]).map((text,i)=>{
    const urg=meta.urgency[i]||"urg-low";
    const tag=meta.tags[i]||"Wellness";
    const numCls=NUM_COLORS[i]||"rn-purple";
    return `<div class="reco-item">
      <div class="reco-num ${numCls}">${i+1}</div>
      <div class="reco-body">
        <p class="reco-text">${text}</p>
        <div class="reco-badges">
          <span class="reco-urgency ${urg}">${URGENCY_LABEL[urg]}</span>
          <span class="reco-tag-pill">${tag}</span>
        </div>
      </div>
      <span class="reco-arrow">›</span>
    </div>`;
  }).join("");
  const crisis=meta.crisis?`<div class="crisis-banner">
    <span class="crisis-icon">⚑</span>
    <span class="crisis-text">In crisis? Call <strong>988</strong> — free, 24/7, confidential.</span>
  </div>`:"";
  document.getElementById("reco-list").innerHTML=items+crisis;
}

/* ── Feedback modal ──────────────────────────── */
function openFeedback(){document.getElementById("feedback-modal").classList.remove("hidden")}
function closeFeedback(){document.getElementById("feedback-modal").classList.add("hidden")}

document.querySelectorAll(".star").forEach(s=>{
  s.addEventListener("click",()=>{
    fbRating=parseInt(s.dataset.v);
    document.querySelectorAll(".star").forEach((x,i)=>x.classList.toggle("active",i<fbRating));
  });
});

async function submitFeedback(){
  try{
    await fetch("/api/feedback",{method:"POST",headers:{"Content-Type":"application/json"},
      body:JSON.stringify({pred_id:currentPredId,rating:fbRating,
        actual_label:document.getElementById("fb-actual").value,
        comment:document.getElementById("fb-comment").value})});
    closeFeedback();
    document.getElementById("fb-comment").value="";
  }catch{}
}

/* ── Compare ─────────────────────────────────── */
async function runCompare(){
  const texts=["cmp1","cmp2","cmp3"].map(id=>document.getElementById(id).value.trim()).filter(t=>t.length>4);
  if(texts.length===0){alert("Enter at least one text to compare.");return;}
  try{
    const res=await fetch("/api/compare",{method:"POST",headers:{"Content-Type":"application/json"},
      body:JSON.stringify({texts})});
    const data=await res.json();
    if(data.error){alert(data.error);return;}
    renderCompare(data.results,texts);
  }catch(e){alert("Error: "+e.message)}
}

function renderCompare(results,texts){
  const container=document.getElementById("compare-results");
  container.classList.remove("hidden");
  container.innerHTML=results.map((r,i)=>{
    const sorted=Object.entries(r.distribution||{}).sort((a,b)=>b[1]-a[1]).slice(0,5);
    const bars=sorted.map(([label,prob])=>{
      const pct=(prob*100).toFixed(1);const col=LABEL_COLORS[label]||"#7c6df0";
      return `<div class="cmp-bar-row">
        <span class="cmp-bar-name">${label}</span>
        <div class="cmp-bar-track"><div class="cmp-bar-fill" style="width:${pct}%;background:${col}"></div></div>
        <span class="cmp-bar-pct" style="color:${col}">${pct}%</span>
      </div>`;
    }).join("");
    return `<div class="cmp-result-card" style="border-color:${r.color||"#7c6df0"}44">
      <p style="font-size:.7rem;color:var(--muted);margin-bottom:.3rem">TEXT ${i+1}</p>
      <div class="cmp-result-label">${r.prediction}</div>
      <div class="cmp-conf">${(r.confidence||0).toFixed(1)}% confidence · ${r.severity} severity</div>
      ${bars}
    </div>`;
  }).join("");
}

/* ── Mood journal ────────────────────────────── */
document.querySelectorAll(".mood-emoji").forEach(el=>{
  el.addEventListener("click",()=>{
    moodScore=parseInt(el.dataset.v);
    document.querySelectorAll(".mood-emoji").forEach(x=>x.classList.remove("active"));
    el.classList.add("active");
    document.getElementById("mood-label-text").textContent=MOOD_LABELS[moodScore];
  });
});

async function submitJournal(){
  const note=document.getElementById("journal-note").value.trim();
  try{
    await fetch("/api/journal",{method:"POST",headers:{"Content-Type":"application/json"},
      body:JSON.stringify({mood_score:moodScore,mood_label:MOOD_LABELS[moodScore],note})});
    document.getElementById("journal-note").value="";
    loadJournal();
  }catch{}
}

async function loadJournal(){
  try{
    const data=await fetch("/api/journal").then(r=>r.json());
    const container=document.getElementById("journal-entries");
    if(!data.entries||data.entries.length===0){
      container.innerHTML='<p class="empty-msg">No entries yet. Log your first mood above.</p>';return;
    }
    container.innerHTML=data.entries.map(e=>`
      <div class="journal-entry">
        <div class="je-header">
          <span class="je-emoji">${MOOD_EMOJIS[e.mood_score]||"😐"}</span>
          <div class="je-meta">
            <div class="je-label">${e.mood_label}</div>
            <div class="je-time">${fmtDateTime(e.at)}</div>
          </div>
        </div>
        ${e.note?`<div class="je-note">${e.note}</div>`:""}
      </div>`).join("");
  }catch{}
}

/* ── Export CSV ──────────────────────────────── */
function exportCSV(){window.location.href="/api/export"}

/* ── Dashboard ───────────────────────────────── */
async function loadStats(){
  try{
    const d=await fetch("/api/stats").then(r=>r.json());
    document.getElementById("total-pred").textContent=d.total_predictions||0;
    document.getElementById("avg-conf").textContent=(d.avg_confidence||0).toFixed(1)+"%";
    if(d.label_distribution){
      const sorted=Object.entries(d.label_distribution).sort((a,b)=>b[1]-a[1]);
      if(sorted.length) document.getElementById("top-category").textContent=sorted[0][0];
      renderGlobalDist(d.label_distribution);
    }
    const tbody=document.getElementById("recent-tbody");
    if(d.recent&&d.recent.length>0){
      tbody.innerHTML=d.recent.map((r,i)=>`<tr>
        <td style="color:var(--muted)">${i+1}</td>
        <td><span class="badge" style="background:${LABEL_COLORS[r.prediction]||"#7c6df0"}">${r.prediction}</span></td>
        <td style="color:${LABEL_COLORS[r.prediction]||"#7c6df0"}">${(r.confidence||0).toFixed(1)}%</td>
        <td><span style="color:${sevColor(r.severity)};font-size:.75rem;font-weight:600">${r.severity}</span></td>
        <td style="color:var(--muted)">${fmtTime(r.at)}</td>
      </tr>`).join("");
    } else {
      tbody.innerHTML=`<tr><td colspan="5" style="text-align:center;color:var(--muted);padding:2rem">No predictions yet.</td></tr>`;
    }
  }catch{}
}

function renderGlobalDist(dist){
  const container=document.getElementById("global-dist");
  const labels=Object.keys(LABEL_COLORS);
  const max=Math.max(...labels.map(l=>dist[l]||0),1);
  container.innerHTML=labels.map(label=>{
    const count=dist[label]||0;const hPct=(count/max)*95;const col=LABEL_COLORS[label];
    return `<div class="dv-col">
      <div class="dv-bar-wrap">
        <div class="dv-bar" style="height:0;background:${col};box-shadow:0 0 10px ${col}55" data-h="${hPct}"></div>
      </div>
      <div class="dv-count" style="color:${col}">${count}</div>
      <div class="dv-label">${label.replace(" ","\n")}</div>
    </div>`;
  }).join("");
  setTimeout(()=>document.querySelectorAll(".dv-bar").forEach(b=>b.style.height=b.dataset.h+"%"),120);
}

/* ── Weekly graph (real API data) ────────────── */
async function loadWeekly(){
  try{
    const d=await fetch("/api/weekly").then(r=>r.json());
    if(!d.days)return;
    renderWeeklySVG(d.days);
    renderHeatmap(d.days);
    updateWeeklySummary(d.days);
  }catch{}
}

function renderWeeklySVG(days){
  const svg=document.getElementById("weekly-svg");
  if(!svg)return;
  const W=700,H=220,PAD={top:20,right:20,bottom:40,left:40};
  const cW=W-PAD.left-PAD.right,cH=H-PAD.top-PAD.bottom;
  const counts=days.map(d=>d.total);
  const max=Math.max(...counts,1);
  const xStep=cW/(days.length-1);
  const pts=days.map((d,i)=>({x:PAD.left+i*xStep,y:PAD.top+cH-(d.total/max)*cH}));
  let html="";
  [0,.25,.5,.75,1].forEach(t=>{
    const y=PAD.top+cH*(1-t);
    html+=`<line class="wk-grid-line" x1="${PAD.left}" y1="${y}" x2="${W-PAD.right}" y2="${y}"/>`;
    if(t>0) html+=`<text class="wk-label" x="${PAD.left-6}" y="${y+4}" text-anchor="end">${Math.round(max*t)}</text>`;
  });
  const areaPath=`M${pts[0].x},${PAD.top+cH} `+pts.map(p=>`L${p.x},${p.y}`).join(" ")+` L${pts[pts.length-1].x},${PAD.top+cH} Z`;
  html+=`<defs><linearGradient id="wkGrad" x1="0" y1="0" x2="0" y2="1">
    <stop offset="0%" stop-color="#7c6df0" stop-opacity="0.4"/>
    <stop offset="100%" stop-color="#7c6df0" stop-opacity="0"/>
  </linearGradient></defs>`;
  html+=`<path d="${areaPath}" fill="url(#wkGrad)"/>`;
  html+=`<path d="M${pts.map(p=>`${p.x},${p.y}`).join(" L")}" class="wk-line" stroke="#7c6df0"/>`;
  pts.forEach((p,i)=>{
    const count=days[i].total;
    html+=`<circle class="wk-dot" cx="${p.x}" cy="${p.y}" fill="#7c6df0" stroke="#07090f" stroke-width="2"/>`;
    if(count>0) html+=`<text class="wk-val-label" x="${p.x}" y="${p.y-10}" text-anchor="middle" fill="#7c6df0">${count}</text>`;
    html+=`<text class="wk-label" x="${p.x}" y="${H-10}" text-anchor="middle">${days[i].day}</text>`;
  });
  svg.innerHTML=html;
}

function renderHeatmap(days){
  const container=document.getElementById("heatmap");
  if(!container)return;
  const labels=Object.keys(LABEL_COLORS);
  let html='<div class="hm-grid">';
  html+='<div class="hm-row"><div class="hm-label"></div>';
  days.forEach(d=>{html+=`<div class="hm-day-label">${d.day}</div>`;});
  html+='</div>';
  labels.forEach(label=>{
    const col=LABEL_COLORS[label];
    const vals=days.map(d=>d.by_category[label]||0);
    const max=Math.max(...vals,1);
    html+=`<div class="hm-row"><div class="hm-label" style="color:${col}">${label}</div>`;
    days.forEach((d,i)=>{
      const val=vals[i];
      const alpha=val===0?.04:.1+(val/max)*.75;
      const bg=hexToRgba(col,alpha);
      html+=`<div class="hm-cell" style="background:${bg};color:${val>0?col:"var(--muted2)"}">${val||""}</div>`;
    });
    html+='</div>';
  });
  html+='</div>';
  container.innerHTML=html;
}

function updateWeeklySummary(days){
  const total=days.reduce((a,d)=>a+d.total,0);
  const peakIdx=days.reduce((mi,d,i,arr)=>d.total>arr[mi].total?i:mi,0);
  const allCats={};
  days.forEach(d=>Object.entries(d.by_category).forEach(([k,v])=>allCats[k]=(allCats[k]||0)+v));
  const topCat=Object.entries(allCats).sort((a,b)=>b[1]-a[1])[0];
  const normCount=allCats["Normal"]||0;
  const normRate=total>0?Math.round((normCount/total)*100):0;
  document.getElementById("ws-total").textContent=total;
  document.getElementById("ws-peak").textContent=total>0?days[peakIdx].day:"—";
  document.getElementById("ws-top").textContent=topCat?topCat[0]:"—";
  document.getElementById("ws-normal").textContent=normRate+"%";
}

/* ── Helpers ─────────────────────────────────── */
function hexToRgba(hex,alpha){
  const r=parseInt(hex.slice(1,3),16),g=parseInt(hex.slice(3,5),16),b=parseInt(hex.slice(5,7),16);
  return `rgba(${r},${g},${b},${alpha})`;
}
function sevColor(s){return {Low:"#4ade80",Moderate:"#fb923c",High:"#f87171"}[s]||"#4ade80"}
function fmtTime(iso){try{return new Date(iso).toLocaleTimeString([],{hour:"2-digit",minute:"2-digit"})}catch{return"—"}}
function fmtDateTime(iso){try{return new Date(iso).toLocaleString([],{month:"short",day:"numeric",hour:"2-digit",minute:"2-digit"})}catch{return"—"}}

/* ── Init ────────────────────────────────────── */
document.getElementById("refresh-stats").addEventListener("click",()=>{loadStats();loadWeekly()});

checkHealth();
loadModelInfo();
loadStats();
loadWeekly();
loadJournal();
setInterval(()=>{loadStats();loadWeekly()},30000);
