// 결과 페이지: 원본 영상 + 캔버스 뼈대, 회차 타임라인, 회차별 측정값 vs 기준, 곡선 vs GLB 정석
const job = location.pathname.split("/").pop();
const EDGES = [[5,7],[7,9],[6,8],[8,10],[5,6],[11,12],[5,11],[6,12],[11,13],[13,15],[12,14],[14,16],[0,5],[0,6]];
const COLOR = {OK: "#3cc864", NG: "#e6484f", CHECK: "#9aa3ad"};
const ACTION_KO = {A001: "버피", A004: "런지", A008: "스쿼트 점프"};
const VIEW_KO = {front: "정면", diag: "사선", side: "측면"};
const METRIC_KO = {
  depth: "깊이 (서 있을 때 대비)", knee_min: "무릎 최소 각도", hip_min: "고관절 최소 각도",
  knee_spread_bottom: "무릎 벌어짐", body_line_min: "허리 처짐", body_line_max: "엉덩이 들림",
  curve_dist: "정석 곡선과 차이", elbow_min: "팔꿈치 최소 각도",
};
let R = null, active = null;

async function poll() {
  const r = await fetch(`/api/jobs/${job}`);
  if (!r.ok) { document.getElementById("wmsg").textContent = "작업을 찾을 수 없습니다"; return; }
  const s = await r.json();
  document.getElementById("wbar").style.width = `${Math.round((s.progress || 0) * 100)}%`;
  if (s.state === "error") {
    document.getElementById("wmsg").innerHTML = `<span class="err">오류: ${s.error}</span>`; return;
  }
  if (s.state !== "done") {
    document.getElementById("wmsg").textContent =
      s.state === "queued" ? "차례를 기다리는 중…" : `분석 중… ${Math.round((s.progress || 0) * 100)}%`;
    return setTimeout(poll, 800);
  }
  R = await (await fetch(`/api/jobs/${job}/result`)).json();
  show();
}

function fmt(x, m) {
  if (x === null || x === undefined) return "-";
  return m === "depth" || m.startsWith("body_line") || m === "knee_spread_bottom" ? x.toFixed(2) : `${x.toFixed(0)}°`;
}
function rangeText(c) {
  const f = v => fmt(v, c.metric);
  if (c.lo !== null && c.lo !== undefined && c.hi !== null && c.hi !== undefined) return `${f(c.lo)} ~ ${f(c.hi)}`;
  if (c.hi !== null && c.hi !== undefined) return `≤ ${f(c.hi)}`;
  if (c.lo !== null && c.lo !== undefined) return `≥ ${f(c.lo)}`;
  return "-";
}

function show() {
  const m = R.meta, s = R.summary;
  document.getElementById("wait").style.display = "none";
  document.getElementById("res").style.display = "grid";
  document.getElementById("sub").textContent =
    `${ACTION_KO[m.action] || m.action} · ${VIEW_KO[m.view] || m.view} · ${m.n_frames} Frame · 분석 ${m.seconds}초`;
  document.getElementById("stats").innerHTML =
    `<span class="pill">회차 ${s.reps}</span><span class="pill OK">OK ${s.OK}</span>` +
    `<span class="pill NG">NG ${s.NG}</span><span class="pill CHECK">CHECK ${s.CHECK}</span>` +
    (s.attention ? `<span class="pill warn">주의 ${s.attention}</span>` : "");
  const v = document.getElementById("v");
  v.src = `/api/jobs/${job}/video`;
  const list = document.getElementById("reps"), tl = document.getElementById("tl");
  const dur = m.n_frames / m.fps;
  R.events.forEach((e, i) => {
    const color = COLOR[e.verdict] || "#888";
    const seg = document.createElement("div");
    seg.className = "rep";
    seg.style.left = `${(e.start_s / dur) * 100}%`;
    seg.style.width = `${Math.max(0.6, ((e.end_s - e.start_s) / dur) * 100)}%`;
    seg.style.background = color;
    seg.title = `${e.rep}회 ${e.verdict}`;
    seg.onclick = ev => { ev.stopPropagation(); select(i); };
    tl.appendChild(seg);
    const rows = (e.checks || []).map(c =>
      `<tr><td>${METRIC_KO[c.metric] || c.metric}</td><td class="${c.ok ? "" : "NG"}">${fmt(c.value, c.metric)}</td>` +
      `<td>기준 ${rangeText(c)}${c.provisional ? " (참고)" : ""}</td></tr>`).join("");
    list.insertAdjacentHTML("beforeend",
      `<div class="repcard" id="rc${i}"><div class="head"><span>${e.rep}회 · ${e.start_s.toFixed(1)}~${e.end_s.toFixed(1)}초</span>` +
      `<span class="${e.verdict}">${e.verdict}${e.model_flag ? ' <span class="warn">· 주의</span>' : ""}</span></div>` +
      `<div class="msg ${e.verdict}">${e.message || ""}</div>${rows ? `<table>${rows}</table>` : ""}</div>`);
    document.getElementById(`rc${i}`).onclick = () => select(i);
  });
  tl.onclick = ev => {
    const r = tl.getBoundingClientRect();
    v.currentTime = ((ev.clientX - r.left) / r.width) * dur;
  };
  const tick = () => { draw(); requestAnimationFrame(tick); };
  requestAnimationFrame(tick);
  if (R.events.length) select(0, false);
}

function select(i, seek = true) {
  active = i;
  document.querySelectorAll(".repcard").forEach(el => el.classList.remove("active"));
  const el = document.getElementById(`rc${i}`);
  el.classList.add("active");
  el.scrollIntoView({block: "nearest"});
  const e = R.events[i];
  if (seek) { const v = document.getElementById("v"); v.currentTime = e.start_s; v.play(); }
  chart(e);
}

function chart(e) {
  const box = document.getElementById("chartbox");
  if (!e.curve) { box.style.display = "none"; return; }
  box.style.display = "block";
  document.getElementById("charttitle").textContent = `${e.rep}회 움직임 곡선 — 내 동작 vs VICON 정석 (같은 촬영 각도)`;
  const svg = document.getElementById("chart"), W = 480, H = 220, pad = 22;
  const all = [...e.curve.knee, ...e.curve.hip, ...(e.template ? [...e.template.knee, ...e.template.hip] : [])]
    .filter(x => x !== null);
  const lo = Math.min(...all, 30) - 5, hi = Math.max(...all, 180) + 5;
  const pt = (arr) => arr.map((y, k) => y === null ? null :
    `${pad + (k / (arr.length - 1)) * (W - 2 * pad)},${H - pad - ((y - lo) / (hi - lo)) * (H - 2 * pad)}`)
    .filter(Boolean).join(" ");
  const line = (arr, color, op) => arr ? `<polyline fill="none" stroke="${color}" stroke-opacity="${op}" stroke-width="2.5" points="${pt(arr)}"/>` : "";
  let grid = "";
  for (let a = Math.ceil(lo / 30) * 30; a <= hi; a += 30) {
    const y = H - pad - ((a - lo) / (hi - lo)) * (H - 2 * pad);
    grid += `<line x1="${pad}" x2="${W - pad}" y1="${y}" y2="${y}" stroke="#262c35"/><text x="2" y="${y + 4}" fill="#8b95a3" font-size="10">${a}°</text>`;
  }
  svg.innerHTML = grid +
    line(e.template && e.template.knee, "#4c9dff", 0.35) + line(e.template && e.template.hip, "#ffc83d", 0.35) +
    line(e.curve.knee, "#4c9dff", 1) + line(e.curve.hip, "#ffc83d", 1) +
    `<text x="${pad}" y="${H - 4}" fill="#8b95a3" font-size="10">시작</text>` +
    `<text x="${W - pad - 20}" y="${H - 4}" fill="#8b95a3" font-size="10">끝</text>`;
}

function draw() {
  const v = document.getElementById("v"), c = document.getElementById("c");
  if (!R || !v.videoWidth) return;
  if (c.width !== v.videoWidth) { c.width = v.videoWidth; c.height = v.videoHeight; }
  const g = c.getContext("2d");
  g.clearRect(0, 0, c.width, c.height);
  const f = Math.min(R.frames.length - 1, Math.floor(v.currentTime * R.meta.fps));
  const kp = R.frames[f];
  const dur = R.meta.n_frames / R.meta.fps;
  document.getElementById("cur").style.left = `${(v.currentTime / dur) * 100}%`;
  const cur = R.events.find(e => v.currentTime >= e.start_s && v.currentTime <= e.end_s + 0.6);
  const lw = Math.max(2, c.width / 250);
  if (kp) {
    g.lineWidth = lw; g.strokeStyle = cur ? (COLOR[cur.verdict] || "#fff") : "#ffffff";
    EDGES.forEach(([a, b]) => {
      if (kp[a][2] >= 0.5 && kp[b][2] >= 0.5) { g.beginPath(); g.moveTo(kp[a][0], kp[a][1]); g.lineTo(kp[b][0], kp[b][1]); g.stroke(); }
    });
    kp.forEach(([x, y, s]) => {
      g.fillStyle = s >= 0.5 ? "#3cc864" : s >= 0.3 ? "#ffd400" : "#e6484f";
      g.beginPath(); g.arc(x, y, lw * 1.6, 0, 7); g.fill();
    });
  }
  if (cur) {
    const fs = Math.max(18, c.width / 22);
    g.font = `700 ${fs}px "Malgun Gothic", sans-serif`;
    g.fillStyle = "rgba(0,0,0,.55)"; g.fillRect(0, 0, c.width, fs * (cur.message ? 2.6 : 1.5));
    g.fillStyle = COLOR[cur.verdict] || "#fff";
    g.fillText(`${cur.rep}회 ${cur.verdict}`, fs * 0.5, fs * 1.1);
    if (cur.message) { g.font = `500 ${fs * 0.7}px "Malgun Gothic", sans-serif`; g.fillText(cur.message, fs * 0.5, fs * 2.1); }
  }
}

poll();
