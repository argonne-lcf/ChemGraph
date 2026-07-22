// === Canvas (author) view ===
// Minimum viable authoring surface: agent nodes on a per-HPC swimlane
// grid, drag to move between lanes, drag from a node's edge-handle to
// another node to add allowed_peers, click to select+edit, tool
// palette to drop tools onto selected agent.
//
// All mutations go through POST /api/campaign with action-typed body.
// The server owns validation + copy-on-edit + auto-rsync.
window.__initCanvas = function initCanvas() {
  const $svg = document.getElementById('canvasSvg');
  const $sel = document.getElementById('canvasCampaignSel');
  const $newBtn = document.getElementById('canvasNewBtn');
  const $isUser = document.getElementById('canvasIsUserCopy');
  const $status = document.getElementById('canvasSaveStatus');
  const $agentList = document.getElementById('canvasAgentList');
  const $addAgent = document.getElementById('canvasAddAgentBtn');
  const $editor = document.getElementById('canvasAgentEditor');
  const $palette = document.getElementById('canvasToolPalette');

  let doc = null;         // current campaign JSON
  let currentName = null; // current campaign name
  let selected = null;    // selected agent name
  let sites = [];         // ordered list of site names (columns)
  let siteOfAgent = {};   // agent -> site (from window.__launchCfg or 'unassigned')
  let posOverride = {};   // agent -> {laneIndex, y} laid out from siteOfAgent
  let edgeDrag = null;    // {source, x, y} while drawing a new edge
  // Agent-node drag state, IIFE-scoped so it survives every render().
  // Previously lived inside wireSvgHandlers, which meant each render()
  // opened a fresh closure -- and every render also re-registered
  // window mousemove/mouseup handlers, leaking N pairs by drag N. Old
  // handlers held stale `drag` references pointing at detached SVG
  // nodes; new mousedowns landed in the wrong closure's `drag` slot
  // so the visible node stopped tracking the cursor.
  let drag = null;

  const LANE_W = 220;
  const NODE_W = 160;
  const NODE_H = 60;
  const LANE_HEADER_H = 30;

  function esc(s) { return String(s).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c])); }

  // Persist canvas selection across page refreshes. Scoped by run_id
  // (from window.__launchCfg.run_id) so switching runs doesn't clobber.
  // Refresh mid-run was previously a footgun: campaign choice + agent
  // site assignments both reset even though the compute jobs kept
  // running -- operator had to re-drag agents to lanes and rely on the
  // canvas matching reality by luck. Save state on every mutation,
  // rehydrate at boot.
  function _cvKey() {
    // Prefer run_id from launch config (published by launch.js after
    // GET /api/launch-config). Fall back to location.origin so an
    // early-init race still gets a stable key rather than 'default'
    // (which would collide across runs opened in the same browser).
    const runId = (window.__launchCfg && window.__launchCfg.run_id) || window.location.pathname;
    return `swarm.canvas.v1.${runId}`;
  }
  function saveCanvasState() {
    try {
      localStorage.setItem(_cvKey(), JSON.stringify({
        campaign: currentName || null,
        siteOfAgent: siteOfAgent || {},
      }));
    } catch (_) { /* quota/private-mode: silently skip */ }
  }
  function loadCanvasState() {
    try {
      const raw = localStorage.getItem(_cvKey());
      return raw ? JSON.parse(raw) : null;
    } catch (_) { return null; }
  }

  async function fetchList() {
    const r = await fetch('/api/campaigns');
    return r.ok ? await r.json() : {shipped: [], user_copies: []};
  }
  async function fetchCampaign(name) {
    const r = await fetch('/api/campaign/' + encodeURIComponent(name));
    return r.ok ? await r.json() : {error: 'fetch failed'};
  }
  async function postAction(body) {
    // Auto-fill campaign name from the currently-loaded canvas doc so
    // per-action call sites do not each have to remember. Explicit
    // ``body.campaign`` (e.g. create_blank passes a new name) wins.
    const finalBody = {...body};
    if (!finalBody.campaign && currentName) {
      finalBody.campaign = currentName;
    }
    const r = await fetch('/api/campaign', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(finalBody),
    });
    return await r.json();
  }

  async function refreshCampaignList() {
    const idx = await fetchList();
    const names = Array.from(new Set([...(idx.shipped || []), ...(idx.user_copies || [])])).sort();
    const placeholder = '<option value="" disabled selected>-- select campaign --</option>';
    $sel.innerHTML = placeholder + names.map(n => `<option value="${esc(n)}">${esc(n)}${(idx.user_copies || []).includes(n) ? ' *' : ''}</option>`).join('');
    if (currentName && names.includes(currentName)) $sel.value = currentName;
  }

  async function loadCampaign(name) {
    const data = await fetchCampaign(name);
    if (data.error) {
      $status.textContent = 'error: ' + data.error;
      return;
    }
    doc = data.campaign;
    currentName = name;
    $isUser.textContent = data.is_user_copy ? '(user copy)' : '(shipped, will copy on save)';
    // Cross-tab affordance: Observability detail panel enriches per-
    // agent info with campaign spec when this is set.
    window.__campaignDoc = doc;
    computeSiteAssignments();
    layoutAgents();
    render();
  }

  function computeSiteAssignments() {
    // Systems come from window.__launchCfg.systems (dashboard startup
    // arg: which HPCs THIS laptop can talk to). Placement is per-launch
    // canvas state -- each agent starts in the first swimlane, operator
    // drags to reassign, and Launch snapshots the current mapping.
    // Restore any saved placement from localStorage so a page refresh
    // mid-run doesn't wipe operator's lane assignments.
    const lcfg = window.__launchCfg;
    sites = (lcfg && lcfg.systems && lcfg.systems.length) ? [...lcfg.systems] : ['unassigned'];
    const saved = loadCanvasState();
    const savedForThisCampaign = (saved && saved.campaign === currentName)
      ? (saved.siteOfAgent || {}) : {};
    siteOfAgent = {};
    (doc.agents || []).forEach(a => {
      const savedSite = savedForThisCampaign[a.name];
      siteOfAgent[a.name] = (savedSite && sites.includes(savedSite))
        ? savedSite : sites[0];
    });
    saveCanvasState();
  }

  function publishSelection() {
    // Handshake with initLaunchControls (via window.__canvasSelection):
    // Launch/Bootstrap buttons up top read this to know what to POST.
    if (!doc || !currentName) {
      window.__canvasSelection = null;
      return;
    }
    const placement = {};
    for (const s of sites) placement[s] = [];
    (doc.agents || []).forEach(a => {
      const s = siteOfAgent[a.name] || sites[0];
      if (!placement[s]) placement[s] = [];
      placement[s].push(a.name);
    });
    // Drop empty lanes so a single-site launch does not carry ghost sites.
    for (const s of Object.keys(placement)) {
      if (!placement[s].length) delete placement[s];
    }
    window.__canvasSelection = {
      campaign: currentName,
      site_agents: placement,
    };
  }

  function layoutAgents() {
    // Vertical stack per lane, in doc order for stable layout.
    const perLane = {};
    sites.forEach(s => { perLane[s] = 0; });
    posOverride = {};
    (doc.agents || []).forEach(a => {
      const site = siteOfAgent[a.name] || sites[0];
      const laneIndex = sites.indexOf(site);
      const y = LANE_HEADER_H + 20 + perLane[site] * (NODE_H + 16);
      perLane[site] += 1;
      posOverride[a.name] = {laneIndex: Math.max(0, laneIndex), y};
    });
  }

  function nodeXY(agentName) {
    const p = posOverride[agentName] || {laneIndex: 0, y: LANE_HEADER_H + 20};
    const x = 20 + p.laneIndex * LANE_W + (LANE_W - NODE_W) / 2;
    return {x, y: p.y};
  }

  function render() {
    const w = Math.max(900, sites.length * LANE_W + 40);
    const maxY = Math.max(
      ...Object.values(posOverride).map(p => p.y + NODE_H + 20),
      500,
    );
    $svg.setAttribute('viewBox', `0 0 ${w} ${maxY}`);
    let out = '';
    // Swimlane bands + headers
    sites.forEach((s, i) => {
      const x = 20 + i * LANE_W;
      const isKnown = s !== 'unassigned';
      out += `<rect x="${x}" y="${LANE_HEADER_H}" width="${LANE_W - 12}" height="${maxY - LANE_HEADER_H - 10}" fill="${isKnown ? '#f0f6ff' : '#f5f5f5'}" stroke="#dbe5f0" data-lane="${esc(s)}"/>`;
      out += `<text x="${x + 10}" y="20" font-size="12" fill="#334" font-weight="600">${esc(s)}</text>`;
    });
    // Arrow marker
    out += `<defs><marker id="canvasArrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="#64748b"/></marker></defs>`;
    // Edges: draw a fat transparent hit-target underneath each visible
    // line so click-to-remove has a forgiving click area. data-edge on
    // both so the click handler can identify the pair.
    (doc.agents || []).forEach(a => {
      const from = nodeXY(a.name);
      (a.allowed_peers || []).forEach(p => {
        if (!posOverride[p]) return;
        const to = nodeXY(p);
        const x1 = from.x + NODE_W / 2;
        const y1 = from.y + NODE_H / 2;
        const x2 = to.x + NODE_W / 2;
        const y2 = to.y + NODE_H / 2;
        const key = `${a.name}::${p}`;
        out += `<line data-edge="${esc(key)}" x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" stroke="transparent" stroke-width="14" style="cursor:pointer;"/>`;
        out += `<line data-edge="${esc(key)}" x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" stroke="#64748b" stroke-width="1.5" marker-end="url(#canvasArrow)" style="pointer-events:none;"/>`;
      });
    });
    // Nodes
    (doc.agents || []).forEach(a => {
      const {x, y} = nodeXY(a.name);
      const isSel = a.name === selected;
      const isInitial = a.name === doc.initial_agent;
      out += `<g data-agent="${esc(a.name)}" transform="translate(${x} ${y})" style="cursor:grab;">
        <rect width="${NODE_W}" height="${NODE_H}" rx="8" fill="${isSel ? '#dbeafe' : '#fff'}" stroke="${isSel ? '#2563eb' : '#94a3b8'}" stroke-width="${isSel ? 2 : 1}"/>
        <text x="10" y="20" font-size="13" font-weight="600" fill="#0f172a">${esc(a.name)}${isInitial ? ' *' : ''}</text>
        <text x="10" y="40" font-size="11" fill="#475569">${esc((a.role || '').slice(0, 22))}</text>
        <text x="10" y="54" font-size="10" fill="#94a3b8">tools: ${(a.allowed_tools || []).length} · peers: ${(a.allowed_peers || []).length}</text>
        <circle data-edge-handle="${esc(a.name)}" cx="${NODE_W}" cy="${NODE_H / 2}" r="6" fill="#2563eb" style="cursor:crosshair;"/>
      </g>`;
    });
    // Ghost line for in-progress edge drag. Kept as a stable
    // element so mousemove can mutate its x2/y2 directly without
    // touching the rest of the canvas (which is what made drag
    // feel laggy -- previously every mousemove re-serialized
    // the whole SVG).
    out += `<line id="canvasEdgeGhost" x1="0" y1="0" x2="0" y2="0" stroke="#2563eb" stroke-dasharray="4,4" stroke-width="1.5" style="display:none; pointer-events:none;"/>`;
    $svg.innerHTML = out;
    wireSvgHandlers();
    renderAgentList();
    renderEditor();
    renderPalette();
    publishSelection();
  }

  function wireSvgHandlers() {
    // Per-node handlers only. The window mousemove/mouseup listeners
    // live in wireGlobalDragHandlers() called ONCE from the IIFE init
    // so re-renders don't leak listener pairs.
    $svg.querySelectorAll('[data-agent]').forEach(g => {
      const name = g.dataset.agent;
      g.addEventListener('mousedown', ev => {
        if (ev.target.dataset && ev.target.dataset.edgeHandle) return;
        // Prevent the browser from starting a text selection while
        // the operator drags. Without this, mouse-down + move
        // highlights the agent-name text inside the node group and
        // the drag feels stuck.
        ev.preventDefault();
        drag = {name, node: g, startX: ev.clientX, startY: ev.clientY, moved: false};
      });
      g.addEventListener('click', ev => {
        if (drag && drag.moved) { drag = null; return; }
        selected = name;
        render();
      });
    });
    $svg.querySelectorAll('[data-edge-handle]').forEach(circle => {
      circle.addEventListener('mousedown', ev => {
        ev.stopPropagation();
        const source = circle.dataset.edgeHandle;
        const pt = svgPoint(ev);
        edgeDrag = {source, x: pt.x, y: pt.y};
        // Prime the ghost line at the source anchor + current cursor.
        const from = nodeXY(source);
        const ghost = document.getElementById('canvasEdgeGhost');
        if (ghost) {
          ghost.setAttribute('x1', String(from.x + NODE_W));
          ghost.setAttribute('y1', String(from.y + NODE_H / 2));
          ghost.setAttribute('x2', String(pt.x));
          ghost.setAttribute('y2', String(pt.y));
          ghost.style.display = '';
        }
      });
    });
    // Click an edge (its fat transparent hit-target) to remove it.
    // Confirm because the click can be accidental when nodes overlap.
    $svg.querySelectorAll('[data-edge]').forEach(line => {
      line.addEventListener('click', async ev => {
        ev.stopPropagation();
        const key = line.dataset.edge;
        const [source, target] = key.split('::');
        if (!confirm(`Remove edge ${source} -> ${target}?`)) return;
        $status.textContent = 'removing edge...';
        const res = await postAction({
          action: 'set_edge',
          params: {source, target, add: false},
        });
        if (res.ok) {
          doc = res.campaign;
          layoutAgents();
          render();
          $status.textContent = `removed ${source} -> ${target}`;
        } else {
          $status.textContent = 'error: ' + (res.error || 'unknown');
        }
      });
    });
  }

  // Registered ONCE at IIFE init below. Reads the IIFE-scoped
  // `drag` and `edgeDrag` so every render() sees the same state.
  function windowMove(ev) {
    if (drag) {
      const dx = ev.clientX - drag.startX;
      const dy = ev.clientY - drag.startY;
      if (Math.abs(dx) + Math.abs(dy) > 4) drag.moved = true;
      // Live translate the node group under the cursor. SVG scale
      // factor comes from the viewBox width vs client width so the
      // node tracks the cursor in SVG coords, not screen pixels.
      if (drag.moved && drag.node) {
        const rect = $svg.getBoundingClientRect();
        const scale = $svg.viewBox.baseVal.width / rect.width;
        drag.node.setAttribute(
          'transform',
          `translate(${dx * scale}, ${dy * scale})`,
        );
      }
    }
    if (edgeDrag) {
      const pt = svgPoint(ev);
      edgeDrag.x = pt.x;
      edgeDrag.y = pt.y;
      const ghost = document.getElementById('canvasEdgeGhost');
      if (ghost) {
        ghost.setAttribute('x2', String(pt.x));
        ghost.setAttribute('y2', String(pt.y));
      }
    }
  }

  async function windowUp(ev) {
    // Clear the live-drag transform so the upcoming render() draws
    // the node at its final grid slot instead of at the offset.
    if (drag && drag.node) drag.node.removeAttribute('transform');
    if (drag && drag.moved) {
      // Determine target lane by cursor X against swimlane bands.
      const pt = svgPoint(ev);
      const laneIndex = Math.max(0, Math.min(sites.length - 1, Math.floor((pt.x - 20) / LANE_W)));
      const newSite = sites[laneIndex];
      if (newSite && siteOfAgent[drag.name] !== newSite) {
        siteOfAgent[drag.name] = newSite;
        saveCanvasState();
        layoutAgents();
        render();
      }
    }
    if (edgeDrag) {
      const pt = svgPoint(ev);
      const target = hitTestAgentAt(pt);
      const dragged = edgeDrag;
      edgeDrag = null;
      const ghost = document.getElementById('canvasEdgeGhost');
      if (ghost) ghost.style.display = 'none';
      if (target && target !== dragged.source) {
        $status.textContent = 'wiring...';
        const res = await postAction({
          action: 'set_edge',
          params: {source: dragged.source, target, add: true},
        });
        if (res.ok) {
          doc = res.campaign;
          layoutAgents();
          render();
          $status.textContent = 'wired ' + dragged.source + ' -> ' + target;
        } else {
          $status.textContent = 'error: ' + (res.error || 'unknown');
        }
      }
    }
    drag = null;
  }

  // One-shot registration.
  window.addEventListener('mousemove', windowMove);
  window.addEventListener('mouseup', windowUp);

  function svgPoint(ev) {
    const rect = $svg.getBoundingClientRect();
    const vb = $svg.viewBox.baseVal;
    const sx = vb.width / rect.width;
    const sy = vb.height / rect.height;
    return {x: (ev.clientX - rect.left) * sx, y: (ev.clientY - rect.top) * sy};
  }

  function hitTestAgentAt(pt) {
    for (const a of (doc.agents || [])) {
      const {x, y} = nodeXY(a.name);
      if (pt.x >= x && pt.x <= x + NODE_W && pt.y >= y && pt.y <= y + NODE_H) {
        return a.name;
      }
    }
    return null;
  }

  function renderAgentList() {
    if (!doc.agents || !doc.agents.length) {
      $agentList.innerHTML = '<li style="color:#94a3b8; font-style:italic;">No agents. Click + add.</li>';
      return;
    }
    $agentList.innerHTML = doc.agents.map(a => `
      <li style="padding:4px 6px; margin:2px 0; border-radius:3px; cursor:pointer; background:${a.name === selected ? '#dbeafe' : 'transparent'};" data-agent="${esc(a.name)}">
        <span>${esc(a.name)}</span>
        <button data-remove="${esc(a.name)}" title="Remove agent" style="float:right; font-size:10px; padding:0 4px; color:#b42318;">×</button>
      </li>
    `).join('');
    $agentList.querySelectorAll('[data-agent]').forEach(li => {
      li.addEventListener('click', ev => {
        if (ev.target.dataset && ev.target.dataset.remove) return;
        selected = li.dataset.agent;
        render();
      });
    });
    $agentList.querySelectorAll('[data-remove]').forEach(btn => {
      btn.addEventListener('click', async ev => {
        ev.stopPropagation();
        const name = btn.dataset.remove;
        if (!confirm(`Remove agent ${name}?`)) return;
        const res = await postAction({action: 'remove_agent', params: {agent: name}});
        if (res.ok) {
          doc = res.campaign;
          if (selected === name) selected = null;
          layoutAgents();
          render();
        } else {
          $status.textContent = 'error: ' + res.error;
        }
      });
    });
  }

  // --- Engine picker --------------------------------------------------
  // Behavior_graph was removed. Each agent now just picks one reasoning
  // engine from the registry; every wake calls that engine once.
  function _wireEngineEditor(a, saveField, flashHint) {
    const registry = window.__engineRegistry || {engines: [], default: 'chemgraph.single_agent'};
    const current = a.engine || registry.default;
    const $sel = document.getElementById('edEngine');
    if (!$sel) return;
    $sel.innerHTML = registry.engines.map(name =>
      `<option value="${esc(name)}" ${name === current ? 'selected' : ''}>${esc(name)}${name === registry.default ? ' (default)' : ''}</option>`
    ).join('');
    $sel.addEventListener('change', () => {
      saveField('engine', $sel.value);
    });
  }

  function renderEditor() {
    if (!selected) {
      $editor.innerHTML = '<div style="color:#94a3b8;">Select an agent to edit.</div>';
      return;
    }
    const a = (doc.agents || []).find(x => x.name === selected);
    if (!a) {
      selected = null;
      $editor.innerHTML = '<div style="color:#94a3b8;">Select an agent to edit.</div>';
      return;
    }
    // Peers: check/uncheck instead of comma-typing. Closed set of
    // known agent names -- typos are impossible by construction.
    const otherAgents = (doc.agents || [])
      .map(x => x.name)
      .filter(n => n !== a.name);
    const currentPeers = new Set(a.allowed_peers || []);
    const peerChecklist = otherAgents.length
      ? otherAgents.map(n => `
          <label style="display:block; font-size:12px; padding:2px 0;">
            <input type="checkbox" class="edPeerCb" value="${esc(n)}" ${currentPeers.has(n) ? 'checked' : ''}/>
            ${esc(n)}
          </label>
        `).join('')
      : '<div style="color:#94a3b8; font-size:11px;">(no other agents to peer with)</div>';

    $editor.innerHTML = `
      <div style="display:flex; align-items:center; justify-content:space-between; margin-bottom:8px;">
        <div style="font-weight:600;">${esc(a.name)}</div>
        <span id="edAutosaveHint" style="font-size:11px; color:#94a3b8;"></span>
      </div>
      <label style="display:block; font-size:11px; color:#64748b;">Role</label>
      <input id="edRole" class="edAutosave" data-field="role" style="width:100%; box-sizing:border-box; margin-bottom:8px; font-size:12px; padding:3px 5px;" value="${esc(a.role || '')}"/>
      <label style="display:block; font-size:11px; color:#64748b;">Mission</label>
      <textarea id="edMission" class="edAutosave" data-field="mission" style="width:100%; box-sizing:border-box; min-height:100px; font-family:ui-monospace,SFMono-Regular,Menlo,monospace; font-size:11px; padding:5px; margin-bottom:8px;">${esc(a.mission || '')}</textarea>
      <label style="display:block; font-size:11px; color:#64748b;">Allowed peers</label>
      <div id="edPeers" style="margin-bottom:8px; max-height:120px; overflow-y:auto; border:1px solid var(--line); border-radius:4px; padding:4px 6px; background:#fbfcfd;">${peerChecklist}</div>
      <button id="edMakeInitial" style="font-size:11px;" ${a.name === doc.initial_agent ? 'disabled' : ''}>${a.name === doc.initial_agent ? 'is initial' : 'make initial'}</button>
      <div style="margin-top:12px; padding:6px 8px; background:#f8fafc; border:1px dashed #cbd5e1; border-radius:4px; font-size:11px; color:#475569;">
        <strong style="font-size:10px; color:#64748b; text-transform:uppercase;">Built-in actions</strong>
        <div style="margin-top:3px;">
          ${BUILTIN_ACTIONS.map(a => `<span title="${esc(a.desc)}" style="display:inline-block; padding:1px 6px; margin:1px 2px; font-size:10px; border:1px dashed #94a3b8; border-radius:8px; background:#fff; color:#475569;">${esc(a.name)}</span>`).join('')}
        </div>
        <div style="margin-top:3px; font-size:10px; color:#94a3b8;">Every agent has these; you cannot opt out.</div>
      </div>
      <hr style="margin:14px 0 8px; border:none; border-top:1px solid var(--line);"/>
      <label style="display:block; font-size:11px; color:#64748b;">Reasoning engine</label>
      <select id="edEngine" style="width:100%; box-sizing:border-box; margin-bottom:10px; font-size:12px; padding:3px 5px;"></select>
      <hr style="margin:14px 0 8px; border:none; border-top:1px solid var(--line);"/>
      <div style="font-size:11px; color:#64748b; margin-bottom:4px;">Inject a message to this agent (operator-originated; use during a run or between them):</div>
      <textarea id="edInjectContent" placeholder="Message content..." style="width:100%; box-sizing:border-box; min-height:60px; font-family:ui-monospace,SFMono-Regular,Menlo,monospace; font-size:11px; padding:5px; margin-bottom:4px;"></textarea>
      <input id="edInjectTldr" placeholder="tldr (optional)" style="width:100%; box-sizing:border-box; margin-bottom:4px; font-size:11px; padding:3px 5px;"/>
      <button id="edInjectSend" style="font-size:11px;">Inject to ${esc(a.name)}</button>
    `;

    // Autosave: role/mission debounce on typing; peers save on toggle.
    // No Save button -- eliminates the "did I save?" ambiguity flagged
    // by users (tools + edges + drag-lane already auto-save).
    const $hint = document.getElementById('edAutosaveHint');
    function flashHint(text, isError) {
      $hint.textContent = text;
      $hint.style.color = isError ? '#b42318' : '#15803d';
      if (!isError) {
        setTimeout(() => {
          if ($hint.textContent === text) {
            $hint.textContent = '';
          }
        }, 1500);
      }
    }
    async function saveField(field, value) {
      flashHint('saving...', false);
      const res = await postAction({
        action: 'edit_agent_field',
        params: {agent: a.name, field, value},
      });
      if (!res.ok) {
        flashHint(`error: ${res.error}`, true);
        return false;
      }
      doc = res.campaign;
      flashHint('✓ saved', false);
      return true;
    }
    let debounceTimer = null;
    function scheduleSave(field, valueFn) {
      if (debounceTimer) clearTimeout(debounceTimer);
      debounceTimer = setTimeout(() => saveField(field, valueFn()), 400);
    }
    document.getElementById('edRole').addEventListener('input', () => {
      scheduleSave('role', () => document.getElementById('edRole').value);
    });
    document.getElementById('edMission').addEventListener('input', () => {
      scheduleSave('mission', () => document.getElementById('edMission').value);
    });
    _wireEngineEditor(a, saveField, flashHint);
    document.querySelectorAll('.edPeerCb').forEach(cb => {
      cb.addEventListener('change', () => {
        const peers = Array.from(document.querySelectorAll('.edPeerCb'))
          .filter(x => x.checked).map(x => x.value);
        saveField('allowed_peers', peers).then(ok => {
          if (ok) { layoutAgents(); render(); }
        });
      });
    });
    document.getElementById('edMakeInitial').addEventListener('click', async () => {
      const res = await postAction({
        action: 'edit_campaign_field',
        params: {field: 'initial_agent', value: a.name},
      });
      if (res.ok) { doc = res.campaign; render(); $status.textContent = 'initial set'; }
      else { $status.textContent = 'error: ' + res.error; }
    });
    document.getElementById('edInjectSend').addEventListener('click', async () => {
      const content = document.getElementById('edInjectContent').value.trim();
      if (!content) { $status.textContent = 'inject: content is empty'; return; }
      const tldr = document.getElementById('edInjectTldr').value.trim() || 'operator-injected';
      $status.textContent = `injecting to ${a.name}...`;
      const r = await fetch('/api/inject-message', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({
          campaign: currentName,
          recipient: a.name,
          content,
          tldr,
          sender: 'operator',
          kind: 'message',
        }),
      });
      const data = await r.json();
      if (!data.ok) { $status.textContent = `inject refused: ${data.detail}`; return; }
      document.getElementById('edInjectContent').value = '';
      document.getElementById('edInjectTldr').value = '';
      $status.textContent = `injected to ${a.name}`;
    });
  }

  // Built-in action tools every agent gets for free from
  // build_chemgraph_reasoning_tools(). Shown in the palette so operators
  // know what an agent can already do; NOT draggable (you can't opt out).
  const BUILTIN_ACTIONS = [
    {name: 'send_message', desc: 'Send message to an allowed peer'},
    {name: 'submit_result', desc: 'Submit a final result for the campaign'},
    {name: 'finish_turn',   desc: 'End this reasoning round'},
  ];

  function mcpName(entry) {
    // Shipped campaigns use {name, command}; user copies could contain
    // either. Also tolerate a bare string as a name-only reference.
    if (typeof entry === 'string') return entry;
    if (entry && typeof entry === 'object') return entry.name || '';
    return '';
  }

  // Cache: (site::name) -> {tools, loading, error, site}. Keyed on site
  // so switching HPC selection doesn't clobber another site's tool list.
  window.__mcpToolsCache = window.__mcpToolsCache || {};

  function _pickDiscoverHpc() {
    // Discover on the HPC where the currently-selected agent lives.
    // Falls back to the first declared HPC in the campaign, so the
    // laptop dashboard never needs the tool's own venv installed.
    if (selected && siteOfAgent[selected] && siteOfAgent[selected] !== 'unassigned') {
      return siteOfAgent[selected];
    }
    const declared = (window.__launchCfg?.systems || []);
    return declared[0] || null;
  }

  function _cacheKey(name, hpc) {
    return `${hpc || '__local__'}::${name}`;
  }

  async function discoverMcp(name, command, refresh) {
    const hpc = _pickDiscoverHpc();
    const key = _cacheKey(name, hpc);
    window.__mcpToolsCache[key] = {loading: true, site: hpc || '__local__'};
    render();
    try {
      const r = await fetch('/api/mcp/discover', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({name, command, hpc, refresh: !!refresh}),
      });
      const data = await r.json();
      if (data.ok) {
        window.__mcpToolsCache[key] = {tools: data.tools, site: data.site || hpc || '__local__'};
      } else {
        window.__mcpToolsCache[key] = {error: data.error || 'unknown', site: data.site || hpc || '__local__'};
      }
    } catch (e) {
      window.__mcpToolsCache[key] = {error: String(e), site: hpc || '__local__'};
    }
    render();
  }

  // Bottom-drawer palette: MCP servers as horizontal chips; click a
  // chip to expand its tool checklist inline. Drag a chip onto an
  // agent to attach. Built-in actions live in the agent editor now
  // (informational per-agent, not a global palette entry).
  let expandedMcp = null;  // name of the MCP server whose tools are expanded
  function renderPalette() {
    const mcps = (doc.mcp_servers || []).filter(m => mcpName(m));
    const selectedAgent = selected ? (doc.agents || []).find(a => a.name === selected) : null;
    const selectedMcpSet = new Set(selectedAgent ? (selectedAgent.mcp_servers || []) : []);
    const selectedToolSet = new Set(selectedAgent ? (selectedAgent.allowed_tools || []) : []);

    // Hint line: what the operator should do next in the palette.
    const $hint = document.getElementById('canvasPaletteHint');
    if ($hint) {
      if (!selectedAgent) {
        $hint.textContent = '(select an agent to see attach / tool controls)';
      } else if (mcps.length === 0) {
        $hint.textContent = 'no MCP servers -- click + Add MCP server';
      } else {
        $hint.textContent = `editing tools for ${selected}`;
      }
    }

    const mcpChipsHtml = mcps.length ? mcps.map(m => {
      const name = mcpName(m);
      const cmd = (m && typeof m === 'object') ? (m.command || '') : '';
      const attached = selectedMcpSet.has(name);
      const isExpanded = expandedMcp === name;
      const attachedCount = (doc.agents || []).filter(a => (a.mcp_servers || []).includes(name)).length;
      return `
        <div style="border:1px solid ${isExpanded ? '#2563eb' : '#dbe5f0'}; border-radius:6px; background:${isExpanded ? '#eff6ff' : '#fbfdff'}; padding:6px 8px; min-width:180px;">
          <div style="display:flex; align-items:center; gap:6px;">
            <span draggable="true" data-mcp="${esc(name)}" style="font-weight:600; font-size:12px; cursor:grab; color:#1e3a8a;" title="drag onto an agent to attach">${esc(name)}</span>
            <span style="font-size:10px; color:${attached ? '#15803d' : '#94a3b8'};" title="attached to N agents">
              ${attached ? '● on ' + esc(selected) : (attachedCount ? `${attachedCount} agents` : '(unused)')}
            </span>
            <button data-toggle-mcp="${esc(name)}" style="font-size:10px; padding:0 4px; margin-left:auto;" title="show tool list">${isExpanded ? '▲' : '▼'}</button>
            <button data-remove-mcp="${esc(name)}" title="Remove MCP server from campaign" style="font-size:10px; padding:0 4px; color:#b42318;">×</button>
          </div>
          ${cmd ? `<div style="font-size:10px; color:#64748b; margin-top:2px; font-family:ui-monospace,SFMono-Regular,Menlo,monospace; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; max-width:260px;" title="${esc(cmd)}">${esc(cmd)}</div>` : ''}
          ${isExpanded ? _renderMcpToolBlock(name, cmd, selectedAgent, selectedToolSet, attached) : ''}
        </div>
      `;
    }).join('') : `
      <div style="color:#94a3b8; font-size:11px;">
        No MCP servers declared yet.
      </div>
    `;

    $palette.innerHTML = mcpChipsHtml;

    // Drag mcp chips onto an agent.
    document.querySelectorAll('#canvasToolPalette [data-mcp]').forEach(chip => {
      chip.addEventListener('dragstart', ev => {
        ev.dataTransfer.setData('text/plain', chip.dataset.mcp);
      });
    });
    // Reset drop handlers each render so we do not stack them.
    $svg.ondragover = ev => ev.preventDefault();
    $svg.ondrop = async ev => {
      ev.preventDefault();
      const mcp = ev.dataTransfer.getData('text/plain');
      if (!mcp) return;
      const pt = svgPoint(ev);
      const target = hitTestAgentAt(pt);
      if (!target) return;
      const agent = (doc.agents || []).find(a => a.name === target);
      if (!agent) return;
      const current = new Set(agent.mcp_servers || []);
      if (current.has(mcp)) { $status.textContent = `${target} already has ${mcp}`; return; }
      current.add(mcp);
      const res = await postAction({
        action: 'edit_agent_field',
        params: {agent: target, field: 'mcp_servers', value: Array.from(current)},
      });
      if (res.ok) { doc = res.campaign; render(); $status.textContent = `added ${mcp} to ${target}`; }
      else { $status.textContent = 'error: ' + res.error; }
    };

    // Toggle expand.
    document.querySelectorAll('#canvasToolPalette [data-toggle-mcp]').forEach(btn => {
      btn.addEventListener('click', () => {
        const name = btn.dataset.toggleMcp;
        expandedMcp = expandedMcp === name ? null : name;
        render();
      });
    });

    // Remove MCP server.
    document.querySelectorAll('#canvasToolPalette [data-remove-mcp]').forEach(btn => {
      btn.addEventListener('click', async () => {
        const name = btn.dataset.removeMcp;
        if (!confirm(`Remove MCP server ${name} from the campaign (and unlink from every agent)?`)) return;
        const res = await postAction({action: 'remove_mcp_server', params: {name}});
        if (res.ok) {
          // Wipe all cached entries for this MCP name across every site.
          Object.keys(window.__mcpToolsCache)
            .filter(k => k.endsWith(`::${name}`))
            .forEach(k => delete window.__mcpToolsCache[k]);
          if (expandedMcp === name) expandedMcp = null;
          doc = res.campaign;
          render();
          $status.textContent = `removed mcp ${name}`;
        }
        else { $status.textContent = 'error: ' + res.error; }
      });
    });

    // Discover / retry buttons (rendered inside expanded tool block).
    document.querySelectorAll('#canvasToolPalette [data-discover-mcp]').forEach(btn => {
      btn.addEventListener('click', () => {
        discoverMcp(btn.dataset.discoverMcp, btn.dataset.discoverCmd, !!btn.dataset.refresh);
      });
    });

    // Tool checkbox toggles (rendered inside expanded tool block).
    document.querySelectorAll('#canvasToolPalette .edToolCb').forEach(cb => {
      cb.addEventListener('change', async () => {
        if (!selected) return;
        const agent = (doc.agents || []).find(a => a.name === selected);
        const current = new Set(agent.allowed_tools || []);
        const tool = cb.dataset.tool;
        if (cb.checked) current.add(tool); else current.delete(tool);
        const res = await postAction({
          action: 'edit_agent_field',
          params: {agent: selected, field: 'allowed_tools', value: Array.from(current)},
        });
        if (res.ok) { doc = res.campaign; render(); $status.textContent = `${cb.checked ? '+' : '-'}${tool}`; }
        else { $status.textContent = 'error: ' + res.error; cb.checked = !cb.checked; }
      });
    });
  }

  function _renderMcpToolBlock(name, cmd, selectedAgent, selectedToolSet, attached) {
    // Rendered inline in the drawer when a chip is expanded.
    // Only shows the tool checklist when the server IS attached to
    // the selected agent -- otherwise there is no target for check
    // toggles. Shows a "not attached" hint when it is not.
    if (!selectedAgent) {
      return `<div style="margin-top:6px; font-size:11px; color:#94a3b8;">Select an agent to see this server's tools.</div>`;
    }
    if (!attached) {
      return `<div style="margin-top:6px; font-size:11px; color:#94a3b8;">Drag this chip onto ${esc(selectedAgent.name)} to expose its tools.</div>`;
    }
    const discoverHpc = _pickDiscoverHpc();
    const cacheEntry = window.__mcpToolsCache[_cacheKey(name, discoverHpc)];
    const siteBadge = discoverHpc
      ? `<span style="font-size:10px; color:#64748b; margin-left:6px;">(discovering on <b>${esc(discoverHpc)}</b>)</span>`
      : `<span style="font-size:10px; color:#94a3b8; margin-left:6px;">(no HPC selected — local discovery)</span>`;
    if (!cacheEntry) {
      return `<div style="margin-top:6px;">${siteBadge}<div style="margin-top:3px;"><button data-discover-mcp="${esc(name)}" data-discover-cmd="${esc(cmd)}" style="font-size:11px;">Discover tools</button></div></div>`;
    }
    if (cacheEntry.loading) {
      return `<div style="margin-top:6px; font-size:11px; color:#64748b;">discovering on ${esc(cacheEntry.site || '?')}...</div>`;
    }
    if (cacheEntry.error) {
      const err = String(cacheEntry.error || '');
      return `<div style="margin-top:6px;">${siteBadge}</div>
        <div style="margin-top:4px; font-size:11px; color:#b42318;">discover failed: ${esc(err)}</div>
        <button data-discover-mcp="${esc(name)}" data-discover-cmd="${esc(cmd)}" data-refresh="1" style="font-size:11px; margin-top:4px;">Retry</button>`;
    }
    const tools = cacheEntry.tools || [];
    if (!tools.length) {
      return `<div style="margin-top:6px; font-size:11px; color:#94a3b8;">(server advertises no tools)</div>`;
    }
    const allExposed = selectedToolSet.size === 0;
    return `
      <div style="margin-top:6px; font-size:11px; color:#64748b;">
        <button data-discover-mcp="${esc(name)}" data-discover-cmd="${esc(cmd)}" data-refresh="1" style="font-size:10px;" title="re-discover">↻ refresh</button>
      </div>
      <div style="max-height:160px; overflow-y:auto; border:1px solid var(--line); border-radius:3px; padding:4px 6px; background:#fff; margin-top:3px;">
        ${allExposed ? '<div style="font-size:10px; color:#15803d; margin-bottom:3px;">(all tools exposed -- check any to restrict)</div>' : ''}
        ${tools.map(t => `
          <label style="display:block; font-size:11px; padding:1px 0;" title="${esc(t.description)}">
            <input type="checkbox" class="edToolCb" data-tool="${esc(t.name)}" ${selectedToolSet.has(t.name) ? 'checked' : ''}/>
            ${esc(t.name)}
          </label>
        `).join('')}
      </div>
    `;
  }

  $addAgent.addEventListener('click', async () => {
    const name = prompt('New agent name (letters, digits, _-):');
    if (!name) return;
    const res = await postAction({action: 'add_agent', params: {agent: name}});
    if (res.ok) {
      doc = res.campaign;
      siteOfAgent[name] = sites[0];
      saveCanvasState();
      layoutAgents();
      selected = name;
      render();
      $status.textContent = `added ${name}`;
    } else {
      $status.textContent = 'error: ' + res.error;
    }
  });

  // + Add MCP server: lives in the palette drawer header now (was
  // rendered inside the palette body per commit that introduced the
  // action-dispatcher). Consolidated here so the palette body only
  // contains chips + tool checklists.
  const $addMcp = document.getElementById('canvasAddMcpBtn');
  if ($addMcp) {
    $addMcp.addEventListener('click', async () => {
      const name = prompt('New MCP server name (letters, digits, _-):');
      if (!name) return;
      const command = prompt('Command to launch it (e.g. "python -m my_pkg.mcp_server"):');
      if (!command) return;
      const res = await postAction({
        action: 'add_mcp_server',
        params: {name, command},
      });
      if (res.ok) { doc = res.campaign; render(); $status.textContent = `added mcp ${name}`; }
      else { $status.textContent = 'error: ' + res.error; }
    });
  }

  $newBtn.addEventListener('click', async () => {
    const name = prompt('New campaign name (letters, digits, _-):');
    if (!name) return;
    // ponytail: prompt() is browser-native ugly but zero JS cost;
    // upgrade to modal if the choose-source flow needs richer UX.
    const idx = await fetchList();
    const knownAll = Array.from(new Set([...(idx.shipped || []), ...(idx.user_copies || [])])).sort();
    let source = prompt(
      `Optional: seed from an existing campaign. Type a name to clone, or leave blank for a fresh empty campaign.\n\nAvailable:\n  ${knownAll.join('\n  ')}`,
      '',
    );
    // prompt returns null on Cancel -- treat as "abort", not "blank".
    if (source === null) return;
    source = (source || '').trim();
    const body = source
      ? {action: 'clone', params: {source}, campaign: name}
      : {action: 'create_blank', params: {}, campaign: name};
    const res = await postAction(body);
    if (res.ok) {
      await refreshCampaignList();
      $sel.value = name;
      await loadCampaign(name);
      $status.textContent = source
        ? `created ${name} (cloned from ${source})`
        : `created ${name}`;
    } else {
      $status.textContent = 'error: ' + res.error;
    }
  });

  $sel.addEventListener('change', () => loadCampaign($sel.value));

  // Launch settings modal: shows the exact command + PBS scripts the
  // next Launch would fire (read-only), alongside editable overrides.
  // Autosaves via edit_launch_field / set_pbs_script actions.
  const $launchModal = document.getElementById('canvasLaunchSettingsModal');
  const $launchBtn = document.getElementById('canvasLaunchSettingsBtn');
  const $launchClose = document.getElementById('canvasLaunchSettingsClose');
  const $launchRefresh = document.getElementById('canvasLaunchSettingsRefresh');
  const $launchHint = document.getElementById('canvasLaunchSettingsHint');
  const $launchExtraArgv = document.getElementById('edLaunchExtraArgv');
  const $launchArgvPreview = document.getElementById('edLaunchArgvPreview');
  const $launchPerSite = document.getElementById('edLaunchPerSite');
  const $launchVarsHelp = document.getElementById('edLaunchVarsHelp');

  let launchSaveTimer = null;
  let launchPreview = null;  // most-recent /api/launch-preview response

  function flashLaunchHint(text, isError) {
    $launchHint.textContent = text;
    $launchHint.style.color = isError ? '#b42318' : '#15803d';
    if (!isError) setTimeout(() => {
      if ($launchHint.textContent === text) $launchHint.textContent = '';
    }, 1500);
  }
  async function saveLaunchField(field, value) {
    if (!currentName) return;
    flashLaunchHint('saving...', false);
    const res = await postAction({
      action: 'edit_launch_field',
      params: {field, value},
    });
    if (res.ok) { doc = res.campaign; flashLaunchHint('✓ saved', false); await refreshLaunchPreview(); }
    else { flashLaunchHint('error: ' + res.error, true); }
  }
  async function saveSitePbs(site, script) {
    if (!currentName) return;
    flashLaunchHint('saving...', false);
    const res = await postAction({
      action: 'set_pbs_script',
      params: {site, script},
    });
    if (res.ok) { doc = res.campaign; flashLaunchHint('✓ saved', false); await refreshLaunchPreview(); }
    else { flashLaunchHint('error: ' + res.error, true); }
  }
  function scheduleLaunchSave(fn) {
    if (launchSaveTimer) clearTimeout(launchSaveTimer);
    launchSaveTimer = setTimeout(fn, 400);
  }

  // Duplicated from launch.js's getCanvasSelection: after the app.js
  // → per-file split, the two IIFEs are separate closures. Both read
  // the same window.__canvasSelection global so the value stays in
  // sync -- this helper just isolates the null-check.
  function getCanvasSelection() {
    return window.__canvasSelection || null;
  }

  async function refreshLaunchPreview() {
    // Hit the server dry-run with the SAME body /api/launch would
    // receive so the operator sees exactly what would fire.
    let sel = getCanvasSelection();
    // If the operator hasn't dragged anything yet, synthesize a
    // preview by assigning every declared agent to the first
    // swimlane -- lets them see the PBS before committing to a
    // placement. (Real Launch still requires actual placement.)
    if (!sel || !sel.campaign || !sel.site_agents || Object.keys(sel.site_agents).length === 0) {
      const systems = (window.__launchCfg && window.__launchCfg.systems) || [];
      const agents = (doc && doc.agents || []).map(a => a.name).filter(Boolean);
      if (currentName && systems.length && agents.length) {
        sel = {campaign: currentName, site_agents: {[systems[0]]: agents}};
      } else {
        $launchArgvPreview.value = '(add at least one agent to this campaign to preview)';
        launchPreview = null;
        renderPerSite();
        return;
      }
    }
    let data;
    try {
      const r = await fetch('/api/launch-preview', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({
          campaign: sel.campaign,
          site_agents: sel.site_agents,
        }),
      });
      if (!r.ok) {
        const errText = await r.text();
        throw new Error(`HTTP ${r.status}: ${errText.slice(0, 200)}`);
      }
      data = await r.json();
    } catch (err) {
      $launchArgvPreview.value = '(preview crashed: ' + err.message + ')';
      launchPreview = null;
      renderPerSite();
      return;
    }
    if (!data.ok) {
      $launchArgvPreview.value = '(preview failed: ' + (data.error || 'unknown') + ')';
      launchPreview = null;
      renderPerSite();
      return;
    }
    launchPreview = data;
    $launchArgvPreview.value = data.argv_string;
    // Auto-fit height to content.
    $launchArgvPreview.style.height = 'auto';
    $launchArgvPreview.style.height = Math.min(200, $launchArgvPreview.scrollHeight + 4) + 'px';
    renderPerSite();
  }

  function renderPerSite() {
    const ld = (doc && doc.launch_defaults) || {};
    const overrides = ld.per_site_overrides || {};
    const previewSites = (launchPreview && launchPreview.sites) || {};
    const siteNames = Object.keys(previewSites).length
      ? Object.keys(previewSites)
      : ((window.__launchCfg && window.__launchCfg.systems) || []);
    if (siteNames.length === 0) {
      $launchPerSite.innerHTML = '<div style="font-size:11px; color:var(--muted);">Dashboard has no --system flag; no sites to configure.</div>';
      return;
    }
    // Surface preview error so an empty modal doesn't mystify the user.
    const previewErr = $launchArgvPreview.value && $launchArgvPreview.value.startsWith('(') ? $launchArgvPreview.value : '';
    const errBanner = previewErr
      ? `<div style="font-size:11px; color:#b42318; background:#fee; border:1px solid #fecaca; border-radius:4px; padding:6px 8px; margin-bottom:10px;">Preview problem: ${esc(previewErr)}<br><span style="color:#7f1d1d;">Editing below still saves; the left "default" pane just cannot be rendered right now.</span></div>`
      : '';
    $launchPerSite.innerHTML = errBanner + siteNames.map(site => {
      const override = (overrides[site] && overrides[site].pbs_script) || '';
      const hasOverride = !!override;
      const preview = previewSites[site] || {};
      // Left pane: fully-substituted preview so operator sees the
      // exact bash that would qsub RIGHT NOW.
      const pbsDefault = preview.pbs_default || '(no preview available — click Refresh)';
      // Right pane (editable): seed with the un-substituted TEMPLATE
      // so ${RUN_DIR}, ${SPAWN_INVOCATION} etc. get re-substituted
      // per launch. Otherwise saving today's expansion pins today's
      // run_id / run_dir forever, breaking every subsequent launch.
      const pbsDefaultTemplate = preview.pbs_default_template || pbsDefault;
      const editableValue = override || pbsDefaultTemplate;
      return `
        <details style="border:1px solid var(--line); border-radius:4px; padding:6px 8px; margin-bottom:8px; background:#fbfcfd;" ${hasOverride ? 'open' : ''}>
          <summary style="cursor:pointer; font-size:12px; display:flex; align-items:center; gap:8px;">
            <strong>${esc(site)}</strong>
            <span style="font-size:10px; color:${hasOverride ? '#15803d' : 'var(--muted)'};">${hasOverride ? '✓ custom' : 'using built-in'}</span>
            <button data-clear="${esc(site)}" style="font-size:10px; margin-left:auto;" ${hasOverride ? '' : 'disabled'} title="Clear override; revert to built-in">Clear override</button>
          </summary>
          <div style="display:grid; grid-template-columns:1fr 1fr; gap:8px; margin-top:6px;">
            <div>
              <div style="font-size:10px; color:var(--muted); text-transform:uppercase; margin-bottom:2px;">Default script for ${esc(site)}, current expansion (read-only)</div>
              <textarea readonly style="width:100%; box-sizing:border-box; min-height:260px; font-family:ui-monospace,SFMono-Regular,Menlo,monospace; font-size:11px; padding:6px; background:#f8fafc; color:#334155;">${esc(pbsDefault)}</textarea>
            </div>
            <div>
              <div style="font-size:10px; color:var(--muted); text-transform:uppercase; margin-bottom:2px;">Template (edit here) — ${'$'}{VAR}s expand at launch</div>
              <textarea data-pbs="${esc(site)}" spellcheck="false" style="width:100%; box-sizing:border-box; min-height:260px; font-family:ui-monospace,SFMono-Regular,Menlo,monospace; font-size:11px; padding:6px;">${esc(editableValue)}</textarea>
            </div>
          </div>
        </details>
      `;
    }).join('');
    // Wire the per-site editable + clear buttons.
    $launchPerSite.querySelectorAll('textarea[data-pbs]').forEach(ta => {
      ta.addEventListener('input', () => {
        const site = ta.dataset.pbs;
        // Diff-detection: if the operator's text matches the built-in
        // preview verbatim, treat as "no override" (empty saves =
        // clear). Avoids persisting a snapshot that would go stale
        // on the next canvas placement change.
        const preview = (launchPreview && launchPreview.sites && launchPreview.sites[site]) || {};
        // Compare against the TEMPLATE (what we seeded from), not the
        // substituted preview. Otherwise the operator's untouched
        // textarea always looks "modified" from the template and we
        // save a copy every keystroke.
        const seed = preview.pbs_default_template || preview.pbs_default || '';
        const value = (ta.value === seed) ? '' : ta.value;
        scheduleLaunchSave(() => saveSitePbs(site, value));
      });
    });
    $launchPerSite.querySelectorAll('button[data-clear]').forEach(btn => {
      btn.addEventListener('click', async (ev) => {
        ev.preventDefault();
        const site = btn.dataset.clear;
        await saveSitePbs(site, '');
      });
    });
  }

  $launchBtn.addEventListener('click', async () => {
    if (!currentName) { alert('Select or create a campaign first.'); return; }
    // Populate the extra-argv box from the campaign JSON.
    const ld = (doc && doc.launch_defaults) || {};
    $launchExtraArgv.value = ld.extra_launcher_argv || '';
    $launchModal.style.display = 'flex';
    await refreshLaunchPreview();
  });
  $launchRefresh.addEventListener('click', refreshLaunchPreview);
  $launchClose.addEventListener('click', () => { $launchModal.style.display = 'none'; });
  $launchModal.addEventListener('click', (ev) => {
    // Close on backdrop click (not on modal-body clicks).
    if (ev.target === $launchModal) $launchModal.style.display = 'none';
  });
  $launchExtraArgv.addEventListener('input', () => {
    scheduleLaunchSave(() => saveLaunchField('extra_launcher_argv', $launchExtraArgv.value));
  });
  $launchVarsHelp.addEventListener('click', (ev) => {
    ev.preventDefault();
    alert(
      'Supported ${VAR} substitutions in PBS scripts:\n\n' +
      '  ${PROJECT}, ${QUEUE}, ${WALLTIME}, ${NODES}, ${FILESYSTEMS}\n' +
      '  ${RUN_DIR}, ${BUNDLE_ROOT}, ${ENV_SCRIPT}, ${ENV_EXPORTS}\n' +
      '  ${SPAWN_INVOCATION}, ${SITE}, ${RUN_ID}, ${CAMPAIGN}\n\n' +
      'The left pane shows what these expand to for the current launch.\n' +
      'Unknown vars raise an error at launch time so typos surface immediately.'
    );
  });
  const $launchFlagsHelp = document.getElementById('edLaunchFlagsHelp');
  if ($launchFlagsHelp) $launchFlagsHelp.addEventListener('click', (ev) => {
    ev.preventDefault();
    alert(
      'Common launcher flags you might append:\n\n' +
      '  --skip-preflight\n' +
      '     Skip the operator env-var check (ALCF_USER, ARGO_USER, etc).\n' +
      '     Use when you deliberately want to defer to login-node defaults.\n\n' +
      '  --spawn-arg <flag> --spawn-arg <value>\n' +
      '     Extra argv passed through to the remote `chemgraph academy\n' +
      '     spawn-site` invocation. Example:\n' +
      '       --spawn-arg --agents-per-node --spawn-arg 2\n\n' +
      'Full list: `swarm launch --help`.'
    );
  });

  (async () => {
    await refreshCampaignList();
    // Prefetch the reasoning-engine registry so the llm_decide node
    // inspector can render its dropdown synchronously. Cheap: single
    // GET returning ~5 names. Failure is non-fatal -- the dropdown
    // falls back to "(default)" only.
    try {
      const r = await fetch('/api/engines');
      window.__engineRegistry = await r.json();
    } catch (_) {
      window.__engineRegistry = {engines: [], default: 'chemgraph.single_agent'};
    }
    // Rehydrate the last-picked campaign for this run from localStorage
    // (so refreshing the browser mid-run doesn't drop selection + agent
    // lane assignments). Fallback to blank if none saved / no longer
    // available in the campaign list.
    const saved = loadCanvasState();
    const savedName = saved && saved.campaign;
    const availableOptions = Array.from($sel.options).map(o => o.value);
    if (savedName && availableOptions.includes(savedName)) {
      $sel.value = savedName;
      await loadCampaign(savedName);  // computeSiteAssignments picks up saved siteOfAgent
    } else {
      $sel.value = '';
      doc = null;
      currentName = null;
      $isUser.textContent = '';
      $status.textContent = 'Select a campaign or click New... to start.';
      publishSelection();  // clears __canvasSelection so Launch shows a helpful error
    }
  })();
};
