// === Launch controls (opt-in, requires --enable-launch-buttons on dashboard) ===
(function initLaunchControls() {
  let cfg = null;       // result of /api/launch-config
  let lastSeq = -1;     // seen seq on /api/launch-status; cheap polling skip

  function stateBadgeColor(state) {
    switch (state) {
      case 'ready':      return '#15803d';
      case 'running':    return '#2563eb';
      case 'queued':     return '#a16207';
      case 'submitting': return '#a16207';
      case 'failed':     return '#b42318';
      default:           return '#6b7280';
    }
  }

  async function fetchConfig() {
    try {
      const r = await fetch('/api/launch-config');
      if (!r.ok) return null;
      return await r.json();
    } catch (e) {
      return null;
    }
  }

  async function fetchStatus() {
    try {
      const r = await fetch('/api/launch-status');
      if (!r.ok) return null;
      return await r.json();
    } catch (e) {
      return null;
    }
  }

  function renderSiteButtons() {
    const container = document.getElementById('launchSites');
    container.innerHTML = '';
    for (const site of (cfg.systems || [])) {
      const wrap = document.createElement('div');
      wrap.style.cssText = 'display:flex; gap:4px; align-items:center;';
      const btn = document.createElement('button');
      btn.textContent = `Launch ${site}`;
      btn.dataset.site = site;
      btn.addEventListener('click', () => doLaunch([site]));
      const badge = document.createElement('span');
      badge.id = `launchBadge-${site}`;
      badge.style.cssText = 'font-size:11px; padding:2px 6px; border-radius:3px; background:#6b7280; color:#fff; min-width:60px; text-align:center;';
      badge.textContent = 'idle';
      const detail = document.createElement('span');
      detail.id = `launchDetail-${site}`;
      detail.style.cssText = 'font-size:11px; color:#b42318; max-width:320px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; cursor:help;';
      const logLink = document.createElement('a');
      logLink.id = `launchLogLink-${site}`;
      logLink.href = '#';
      logLink.textContent = '[pbs log]';
      logLink.title = `Tail <run_dir>/${site}.pbs.log on the HPC via ssh`;
      logLink.style.cssText = 'font-size:11px; color:#2563eb; text-decoration:underline; cursor:pointer;';
      logLink.addEventListener('click', (e) => {
        e.preventDefault();
        fetchSiteLog(site);
      });
      wrap.appendChild(btn);
      wrap.appendChild(badge);
      wrap.appendChild(detail);
      wrap.appendChild(logLink);
      container.appendChild(wrap);
    }
  }

  async function fetchSiteLog(site) {
    const panel = document.getElementById('siteLogPanel');
    const title = document.getElementById('siteLogTitle');
    const body = document.getElementById('siteLogBody');
    title.textContent = `Loading ${site}.pbs.log...`;
    body.textContent = '';
    panel.style.display = 'block';
    try {
      const r = await fetch(`/api/site-log/${encodeURIComponent(site)}`);
      const data = await r.json();
      if (data.error) {
        title.textContent = `${site} -- error fetching log`;
        body.textContent = data.error + '\n\nPaths tried:\n' +
          (data.remote_paths || []).map(p => '  ' + p).join('\n');
        return;
      }
      title.textContent =
        `${site} via ssh ${data.login_host} ` +
        `(${data.lines.length} lines, paths: ${data.remote_paths.join(', ')})`;
      body.textContent = (data.lines || []).join('\n');
      body.scrollTop = body.scrollHeight;
    } catch (e) {
      title.textContent = `${site} -- fetch failed`;
      body.textContent = String(e);
    }
  }

  function getCanvasSelection() {
    // Populated by initCanvas whenever it loads a campaign or the
    // user drags placement between swimlanes. Absent = canvas has no
    // campaign selected yet.
    return window.__canvasSelection || null;
  }

  async function doLaunch(sites) {
    const sel = getCanvasSelection();
    if (!sel || !sel.campaign) {
      alert('Open the Canvas tab and select or create a campaign before launching.');
      return;
    }
    if (!sel.site_agents || Object.keys(sel.site_agents).length === 0) {
      alert('Place at least one agent onto a swimlane in the Canvas tab.');
      return;
    }
    // Persist for the Observability graph so its site swimlanes match
    // what this launch actually placed.
    window.__lastLaunchPlacement = sel.site_agents;
    // Bootstrap always fires on launch now (kickoff of the campaign is
    // the whole point of Launch; the separate Bootstrap button was
    // legacy from the pre-inject-primitive era). Operator uses inject
    // for anything mid-run.
    const body = {
      campaign: sel.campaign,
      site_agents: sel.site_agents,
      sites: sites,
      auto_bootstrap: true,
    };
    const r = await fetch('/api/launch', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(body),
    });
    const data = await r.json();
    if (!data.ok) {
      alert(`Launch refused: ${data.detail}`);
    }
  }

  async function doQdel() {
    if (!confirm('qdel all known PBS jobs for this run?')) return;
    const r = await fetch('/api/qdel', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({}),
    });
    const data = await r.json();
    const summary = (data.results || []).map(x =>
      `${x.site}: ${x.ok ? 'ok' : 'fail'} ${x.detail || ''}`
    ).join('\n');
    alert(summary || 'no jobs known');
  }

  function applyStatus(s) {
    if (!s) return;
    if (s.seq === lastSeq) return;
    lastSeq = s.seq;

    // Per-site badges + failure detail
    let anyFailed = false;
    for (const site of (cfg.systems || [])) {
      const badge = document.getElementById(`launchBadge-${site}`);
      if (!badge) continue;
      const ss = s.sites[site];
      if (!ss) continue;
      const label = ss.pbs_state
        ? `${ss.state} (PBS:${ss.pbs_state})`
        : ss.state;
      badge.textContent = label;
      badge.style.background = stateBadgeColor(ss.state);

      // Failure detail next to the badge
      const detail = document.getElementById(`launchDetail-${site}`);
      if (detail) {
        if (ss.state === 'failed' && ss.last_event) {
          detail.textContent = ss.last_event;
          detail.title = ss.last_event;  // full text on hover
          anyFailed = true;
        } else if (ss.last_event && ss.state !== 'idle') {
          // For non-failure states, show a faint hint of progress
          detail.textContent = ss.last_event;
          detail.title = ss.last_event;
          detail.style.color = '#6b7280';
        } else {
          detail.textContent = '';
          detail.title = '';
        }
      }
    }

    // Auto-open the launcher log <details> when something failed,
    // so the operator doesn't miss diagnostic output buried in a
    // collapsed section.
    if (anyFailed) {
      const det = document.querySelector('#launchPanel details');
      if (det) det.open = true;
    }

    // Overall status: bootstrap now always auto-fires with launch, so
    // its state is a launch subphase rather than a separate operator
    // step. Surface as inline text but not a separate button.
    const overall = document.getElementById('launchOverallStatus');
    const bits = [];
    if (s.launcher_running) bits.push('launcher running');
    if (s.bootstrap && s.bootstrap !== 'idle') {
      bits.push(`bootstrap=${s.bootstrap}`);
    }
    overall.textContent = bits.join(' · ');
    overall.style.color = s.bootstrap === 'failed' ? '#b42318' : '#6b7280';

    // Log buffer
    const logEl = document.getElementById('launchLog');
    logEl.textContent = (s.log_lines || []).join('\n');
    logEl.scrollTop = logEl.scrollHeight;

    // Per-site buttons: disable ONLY the sites that already have a
    // launch subprocess in flight. Sites can be launched independently
    // (aurora queuing does not block clicking Launch crux).
    // Launch-all disables when EVERY site is already running.
    const runningSites = new Set(
      Object.entries(s.sites || {}).filter(([, ss]) => ss.running).map(([n]) => n)
    );
    for (const site of (cfg.systems || [])) {
      const btn = document.querySelector(`#launchSites button[data-site="${site}"]`);
      if (btn) btn.disabled = runningSites.has(site);
    }
    const launchAll = document.getElementById('launchAllBtn');
    launchAll.disabled = (cfg.systems || []).length > 0
      && (cfg.systems || []).every(x => runningSites.has(x));
    // Publish for the mission-save button (rendered inside renderDetail).
    // Kept true as long as ANY site is live so the Save button knows
    // not to fire a canvas edit into a running run.
    window.__launcherRunning = runningSites.size > 0;
  }

  async function pollLoop() {
    const s = await fetchStatus();
    applyStatus(s);
  }

  (async () => {
    cfg = await fetchConfig();
    if (!cfg) return;  // launch buttons not enabled; leave panel hidden
    // Publish for the main app's site-merge code so it has authoritative
    // agent->site mapping from the operator's launch config.
    window.__launchCfg = cfg;
    document.getElementById('launchPanel').style.display = 'block';
    renderSiteButtons();
    document.getElementById('launchAllBtn').addEventListener(
      'click', () => doLaunch(null)  // null sites = launcher defaults to all
    );
    document.getElementById('qdelAllBtn').addEventListener(
      'click', doQdel
    );
    // Copy-to-clipboard for the launcher log. Button lives inside
    // the <summary>, so we need to prevent the click from toggling
    // the <details> open/close.
    const $logCopy = document.getElementById('launchLogCopyBtn');
    if ($logCopy) {
      $logCopy.addEventListener('click', async ev => {
        ev.preventDefault();
        ev.stopPropagation();
        const text = document.getElementById('launchLog').textContent || '';
        try {
          await navigator.clipboard.writeText(text);
          $logCopy.textContent = 'copied';
          setTimeout(() => { $logCopy.textContent = 'copy'; }, 1200);
        } catch (e) {
          $logCopy.textContent = 'error';
        }
      });
    }
    // Copy-to-clipboard for the per-site PBS log panel.
    const $siteCopy = document.getElementById('siteLogCopyBtn');
    if ($siteCopy) {
      $siteCopy.addEventListener('click', async () => {
        const text = document.getElementById('siteLogBody').textContent || '';
        try {
          await navigator.clipboard.writeText(text);
          $siteCopy.textContent = 'copied';
          setTimeout(() => { $siteCopy.textContent = 'copy'; }, 1200);
        } catch (e) {
          $siteCopy.textContent = 'error';
        }
      });
    }
    await pollLoop();
    setInterval(pollLoop, 2000);
  })();
})();
