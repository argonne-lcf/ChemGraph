    let snapshot = null;
    let snapshotIdentity = null;
    let selectedAgent = null;
    let selectedEdgeKey = null;
    let selectedActivityEventKey = null;
    let timelineIndex = null;
    let followLatest = true;
    let graphMode = 'recent';
    let isReplaying = false;
    let replayTimer = null;
    let replayStartedAtMs = 0;
    let replayStartTimestamp = null;
    let replayStartIndex = 0;
    let graphView = null;
    let graphPanDrag = null;
    let embeddedWorkflowView = null;
    let embeddedWorkflowPanDrag = null;
    let selectedEmbeddedWorkflowEventKey = null;
    let embeddedWorkflowAnchorKey = null;
    let workflowPanelOpen = true;
    let workflowPanelFrame = {x: 28, y: 76, width: 1120, height: 640};
    let workflowPanelDrag = null;
    let workflowPanelResizeDrag = null;
    let detailResizeDrag = null;
    let lastRenderedDetailIdentity = null;
    let lastEmbeddedWorkflowInspectorIdentity = null;
    const recentMessageWindow = 4;
    const actionToolNames = new Set(['send_message', 'submit_result', 'finish_turn']);
    const renderedHtmlCache = new WeakMap();

    const esc = (s) => String(s ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
    const trunc = (s, n=180) => {
      s = String(s ?? '');
      return s.length > n ? s.slice(0, n - 1) + '…' : s;
    };
    const formatTime = (timestamp) => timestamp ? new Date(timestamp * 1000).toLocaleTimeString() : '-';
    const eventTimestamp = (event) => {
      const value = Number(event?.timestamp);
      return Number.isFinite(value) ? value : null;
    };
    const replaySpeed = () => Number(document.getElementById('replaySpeed')?.value || 25);
    const eventHoldSeconds = () => Number(document.getElementById('eventHold')?.value || 8);

    function estimateLabelWidth(text) {
      return Math.min(460, Math.max(20, String(text || '').length * 6.2 + 14));
    }

    function labelBox(x, y, text) {
      const width = estimateLabelWidth(text);
      const height = 18;
      return {
        x1: x - width / 2,
        y1: y - height + 5,
        x2: x + width / 2,
        y2: y + 7,
      };
    }

    function boxesOverlap(a, b, pad = 5) {
      return !(a.x2 + pad < b.x1 || b.x2 + pad < a.x1 || a.y2 + pad < b.y1 || b.y2 + pad < a.y1);
    }

    function placeEdgeLabel(baseX, baseY, rawLabel, occupiedBoxes, force = false) {
      const text = trunc(String(rawLabel || ''), force ? 96 : 72);
      if (!text) return null;
      const candidates = [
        [0, 0], [0, -18], [0, 18], [30, -12], [-30, 12],
        [44, 18], [-44, -18], [0, -34], [0, 34],
      ];
      for (const [dx, dy] of candidates) {
        const x = baseX + dx;
        const y = baseY + dy;
        const box = labelBox(x, y, text);
        const collision = occupiedBoxes.some(other => boxesOverlap(box, other));
        if (force || !collision) {
          occupiedBoxes.push(box);
          return {x, y, text};
        }
      }
      return null;
    }

    async function load() {
      const [statusRes, eventsRes] = await Promise.all([
        fetch('/api/status'),
        fetch('/api/events'),
      ]);
      const statusData = await statusRes.json();
      const eventsData = await eventsRes.json();
      const nextSnapshot = {...statusData, events: eventsData.events || []};
      // Federated detection: server-side B.4c.2 wraps per-site state
      // under ``sites: {<name>: {status, placement, summary, updated, schema}}``
      // for multi-site runs and tags every event with ``site``. Build
      // a flat agent->site index so renderers can ask "which site does
      // this agent live on?" without re-walking sites every time.
      nextSnapshot.federated = !!(nextSnapshot.sites && Object.keys(nextSnapshot.sites).length);
      nextSnapshot.siteNames = nextSnapshot.federated
        ? Object.keys(nextSnapshot.sites).sort()
        : [];
      nextSnapshot.sitesByAgent = {};
      // Per-launch placement, seeded from window.__lastLaunchPlacement
      // which the canvas sets when POST /api/launch fires. Without a
      // launch yet, agent->site mapping falls through to the per-event
      // ``site`` backfill below.
      if (window.__lastLaunchPlacement) {
        for (const [siteName, agentList] of Object.entries(window.__lastLaunchPlacement)) {
          (agentList || []).forEach(a => { nextSnapshot.sitesByAgent[a] = siteName; });
        }
      }
      if (nextSnapshot.federated) {
        // Merge per-site status/placement/summary up to the top level so
        // the rest of the app (agents(), renderMetrics(), workflow code)
        // can read snapshot.status / snapshot.placement exactly like in
        // single-site mode. Without this merge `snapshot.status?.agents`
        // is undefined in federated runs and the entire graph + metrics
        // panel renders empty even though events stream in.
        const mergedAgents = [];
        const mergedPlacements = {};
        const seenAgentIds = new Set();
        let mergedSchema = null;
        const mergedStatusExtras = {};
        for (const [siteName, siteData] of Object.entries(nextSnapshot.sites)) {
          const agents = (siteData?.status?.agents) || [];
          agents.forEach(spec => {
            const agentId = spec.agent_id || spec.agent_name || spec.name;
            if (agentId && !seenAgentIds.has(agentId)) {
              seenAgentIds.add(agentId);
              mergedAgents.push({...spec, site: siteName});
            }
          });
          // Note: not using siteData.local_agents because /eagle is
          // mounted on every ALCF site, so the per-site rsync mirrors
          // of agent_status/ end up identical. The launch-cfg mapping
          // above is the authoritative source.
          const placements = (siteData?.placement?.agents) || {};
          Object.entries(placements).forEach(([agentId, placement]) => {
            if (!(agentId in mergedPlacements)) {
              mergedPlacements[agentId] = placement;
            }
          });
          // Carry a representative schema + common status scalars so
          // isWorkflowMode() and "campaign / mode" lookups behave the
          // same as single-site. Last-write-wins is fine here -- all
          // sites in a federated run share the same campaign/mode.
          if (siteData?.schema) mergedSchema = siteData.schema;
          const siteStatus = siteData?.status || {};
          ['campaign', 'campaign_kind', 'mode', 'converged', 'query',
            'workflow_type', 'model_name'].forEach(key => {
            if (siteStatus[key] !== undefined && mergedStatusExtras[key] === undefined) {
              mergedStatusExtras[key] = siteStatus[key];
            }
          });
        }
        // Backfill from events too, since the per-event ``site`` tag
        // is authoritative for the agent that emitted each event.
        (nextSnapshot.events || []).forEach(event => {
          if (event.site && event.agent_id && !(event.agent_id in nextSnapshot.sitesByAgent)) {
            nextSnapshot.sitesByAgent[event.agent_id] = event.site;
          }
        });
        nextSnapshot.status = {
          ...mergedStatusExtras,
          agents: mergedAgents,
        };
        nextSnapshot.placement = {agents: mergedPlacements};
        if (mergedSchema && !nextSnapshot.schema) {
          nextSnapshot.schema = mergedSchema;
        }
      }
      const nextIdentity = identityForSnapshot(nextSnapshot);
      const previousEventCount = snapshot?.events?.length || 0;
      const nextEventCount = nextSnapshot.events.length;
      if (
        snapshotIdentity !== null
        && (
          nextIdentity !== snapshotIdentity
          || nextEventCount < previousEventCount
        )
      ) {
        resetInteractionState();
      }
      snapshotIdentity = nextIdentity;
      snapshot = nextSnapshot;
      const latest = allEvents().length - 1;
      if (followLatest || timelineIndex === null) {
        timelineIndex = latest;
      } else {
        timelineIndex = Math.min(timelineIndex, latest);
      }
      render();
    }

    function identityForSnapshot(data) {
      return [
        data?.schema || '',
        data?.run_dir || '',
      ].join('|');
    }

    function resetInteractionState() {
      stopReplay(false);
      selectedAgent = null;
      selectedEdgeKey = null;
      selectedActivityEventKey = null;
      timelineIndex = null;
      followLatest = true;
      graphView = null;
      graphPanDrag = null;
      embeddedWorkflowView = null;
      embeddedWorkflowPanDrag = null;
      lastRenderedDetailIdentity = null;
      lastEmbeddedWorkflowInspectorIdentity = null;
      selectedEmbeddedWorkflowEventKey = null;
      embeddedWorkflowAnchorKey = null;
    }

    function allEvents() {
      return snapshot?.events || [];
    }

    function isWorkflowMode() {
      return snapshot?.schema === 'chemgraph_workflow';
    }

    function currentEventIndex() {
      const events = allEvents();
      if (!events.length) return -1;
      if (timelineIndex === null) return events.length - 1;
      return Math.max(0, Math.min(timelineIndex, events.length - 1));
    }

    function visibleEvents() {
      const index = currentEventIndex();
      return index < 0 ? [] : allEvents().slice(0, index + 1);
    }

    function currentEvent() {
      const index = currentEventIndex();
      return index < 0 ? null : allEvents()[index];
    }

    function eventKey(event) {
      if (!event) return '';
      if (event.event_id) return String(event.event_id);
      const payload = JSON.stringify(event.payload || {});
      return [
        event.timestamp ?? '',
        event.event || '',
        event.agent_id || '',
        event.correlation_id || '',
        payload.slice(0, 220),
      ].join('|');
    }

    function firstTimestamp() {
      const first = allEvents().find(event => eventTimestamp(event) !== null);
      return eventTimestamp(first);
    }

    function currentTimestamp() {
      return eventTimestamp(currentEvent());
    }

    function eventIndexAtTimestamp(timestamp) {
      const events = allEvents();
      if (!events.length) return -1;
      if (timestamp === null || timestamp === undefined || Number.isNaN(timestamp)) {
        return Math.min(events.length - 1, replayStartIndex + 1);
      }
      let index = 0;
      for (let i = 0; i < events.length; i += 1) {
        const ts = eventTimestamp(events[i]);
        if (ts === null) {
          index = i;
          continue;
        }
        if (ts <= timestamp) index = i;
        else break;
      }
      return index;
    }

    function activeWindowEvents(multiplier = 1) {
      const events = visibleEvents();
      const now = currentTimestamp();
      if (now === null) return events.slice(-Math.max(1, recentMessageWindow));
      const hold = eventHoldSeconds() * multiplier;
      return events.filter(event => {
        const ts = eventTimestamp(event);
        return ts !== null && ts <= now && now - ts <= hold;
      });
    }

    function eventsOf(type) {
      return visibleEvents().filter(e => e.event === type);
    }

    function graphMessageEvents() {
      const sent = eventsOf('message_sent');
      if (graphMode === 'cumulative') return sent;
      if (graphMode === 'current') {
        const event = currentEvent();
        const activeMessages = activeWindowEvents(1).filter(e => e.event === 'message_sent');
        if (!event) return activeMessages;
        if (activeMessages.length) return activeMessages;
        if (event.event === 'message_sent') return [event];
        if (event.event === 'message_received') {
          const messageId = event.payload?.message_id;
          return sent.filter(item => item.payload?.message_id === messageId).slice(-1);
        }
        return [];
      }
      const now = currentTimestamp();
      if (now !== null) {
        const windowSeconds = Math.max(eventHoldSeconds() * 2, 8);
        const windowed = sent.filter(event => {
          const ts = eventTimestamp(event);
          return ts !== null && ts <= now && now - ts <= windowSeconds;
        });
        if (windowed.length) return windowed;
      }
      return sent.slice(-recentMessageWindow);
    }

    function graphModeLabel() {
      if (graphMode === 'current') return `showing active events for ${eventHoldSeconds()}s`;
      if (graphMode === 'cumulative') return 'showing all prior communication';
      return `showing recent communication window`;
    }

    function latestEventOf(type, agentId = null) {
      const matches = visibleEvents().filter(e => e.event === type && (!agentId || e.agent_id === agentId));
      return matches.length ? matches[matches.length - 1] : null;
    }

    function agents() {
      if (!snapshot) return [];
      const specs = snapshot.status?.agents || [];
      const currentEvents = visibleEvents();
      const visiblePlacements = {};
      currentEvents.forEach(event => {
        if (event.event !== 'agent_started' || !event.agent_id) return;
        const placement = event.payload?.placement;
        if (placement) visiblePlacements[event.agent_id] = placement;
      });
      const finalPlacements = snapshot.placement?.agents || {};
      return specs.map(spec => {
        const agentId = spec.agent_id || spec.agent_name || spec.name;
        return {
          ...spec,
          agent_id: agentId,
          agent_name: spec.agent_name || agentId,
          ...agentStateAt(agentId, currentEvents),
          placement: visiblePlacements[agentId] || finalPlacements[agentId] || spec.placement || {},
        };
      }).filter(agent => agent.agent_id);
    }

    function agentStateAt(agentId, events) {
      const state = {
        started: false,
        last_error: null,
        decision_count: 0,
        received_message_count: 0,
        outbox_count: 0,
        tool_started_count: 0,
        tool_finished_count: 0,
      };
      events.forEach(event => {
        if (event.agent_id !== agentId) return;
        if (event.event === 'agent_started') state.started = true;
        if (event.event === 'agent_error') state.last_error = event.payload?.error || 'agent_error';
        if (event.event === 'agent_decision') state.decision_count += 1;
        if (event.event === 'message_received') state.received_message_count += 1;
        if (event.event === 'message_sent') state.outbox_count += 1;
        if (event.event === 'tool_call_started') state.tool_started_count += 1;
        if (event.event === 'tool_call_finished' || event.event === 'tool_call_failed') state.tool_finished_count += 1;
      });
      return state;
    }

    function agentHost(agent) {
      return agent?.placement?.short_hostname || agent?.placement?.hostname || (agent?.started ? 'unknown host' : 'pending');
    }

    function agentSite(agent) {
      // In federated runs the meaningful grouping is "which HPC"
      // (aurora vs crux), not individual compute hostnames. The
      // server tags every event with ``site`` and we built a
      // sitesByAgent index in load(); fall back to "pending" so
      // not-yet-registered agents still render.
      if (!snapshot?.federated) return null;
      const agentId = agent?.agent_id || agent?.agent_name || agent?.name;
      return snapshot.sitesByAgent[agentId] || 'pending';
    }

    function agentGroup(agent) {
      // Single source of truth for "what bucket does this agent
      // belong to in the current view": site (federated) or host
      // (single-machine). Renderers that ask "what color / what
      // label" go through here so federated vs single-site rendering
      // diverges in exactly one place.
      return agentSite(agent) || agentHost(agent);
    }

    function renderSitesBadge() {
      // Header-bar federation indicator. Hidden in single-site runs
      // so the existing single-site UI looks unchanged; in federated
      // runs it shows e.g. "Sites: aurora · crux (2 agents on aurora,
      // 1 on crux, 0 events from aurora)" so operators / demo
      // viewers can confirm at a glance that the campaign really
      // is spanning multiple HPCs.
      const badge = document.getElementById('sitesBadge');
      if (!badge) return;
      if (!snapshot?.federated || !snapshot.siteNames.length) {
        badge.style.display = 'none';
        badge.textContent = '';
        return;
      }
      const eventCounts = {};
      (snapshot.events || []).forEach(e => {
        if (e.site) eventCounts[e.site] = (eventCounts[e.site] || 0) + 1;
      });
      const agentCounts = {};
      Object.values(snapshot.sitesByAgent || {}).forEach(site => {
        agentCounts[site] = (agentCounts[site] || 0) + 1;
      });
      const parts = snapshot.siteNames.map(site => {
        const a = agentCounts[site] || 0;
        const e = eventCounts[site] || 0;
        return `${site} (${a}🤖 / ${e}📨)`;
      });
      badge.textContent = 'Sites: ' + parts.join(' · ');
      badge.style.display = '';
    }

    function hostColor(index) {
      const colors = ['#dbeafe', '#dcfce7', '#fef3c7', '#fce7f3', '#e0e7ff', '#ccfbf1', '#fee2e2', '#ede9fe'];
      return colors[index % colors.length];
    }

    function hostStroke(index) {
      const colors = ['#2563eb', '#16a34a', '#d97706', '#db2777', '#4f46e5', '#0f766e', '#dc2626', '#7c3aed'];
      return colors[index % colors.length];
    }

    function render() {
      const detailScroll = captureDetailScrollSnapshot();
      document.getElementById('updated').textContent = snapshot.updated ? new Date(snapshot.updated * 1000).toLocaleTimeString() : '';
      document.getElementById('runPath').textContent = snapshot.run_dir || '';
      renderSitesBadge();
      document.getElementById('graphTitle').textContent = isWorkflowMode() ? 'ChemGraph Workflow' : 'Agent Graph';
      renderTimeline();
      renderMetrics();
      renderGraph();
      renderAgentPicker();
      renderDetail();
      renderEmbeddedWorkflowPanel();
      restoreDetailScrollSnapshot(detailScroll);
      lastRenderedDetailIdentity = currentDetailIdentity();
    }

    function renderTimeline() {
      const events = allEvents();
      const slider = document.getElementById('timeSlider');
      const index = currentEventIndex();
      slider.max = String(Math.max(0, events.length - 1));
      slider.value = String(Math.max(0, index));
      slider.disabled = events.length === 0;
      const event = index >= 0 ? events[index] : null;
      const mode = isReplaying ? 'replay' : followLatest ? 'latest' : `event ${index + 1}`;
      document.getElementById('timeLabel').textContent = `${mode} / ${events.length}`;
      document.getElementById('timeEvent').textContent = event
        ? `${formatTime(event.timestamp)} ${event.event}${event.agent_id ? ` · ${event.agent_id}` : ''} · ${graphModeLabel()}`
        : '';
      document.getElementById('playReplay').textContent = isReplaying ? 'Pause' : 'Replay';
      document.querySelectorAll('#graphMode button').forEach(button => {
        button.classList.toggle('active', button.dataset.mode === graphMode);
      });
    }

    function renderMetrics() {
      if (isWorkflowMode()) {
        renderWorkflowMetrics();
        return;
      }
      const events = visibleEvents();
      const counts = {};
      events.forEach(event => { counts[event.event] = (counts[event.event] || 0) + 1; });
      const currentAgents = agents();
      const startedAgents = currentAgents.filter(agent => agent.started);
      // Group by site in federated mode, by host otherwise. The
      // "cross-node messages" metric below stays meaningful either
      // way -- in federated mode it becomes "messages that crossed
      // the HPC boundary," which is exactly what we want to surface.
      const hostByAgent = new Map(currentAgents.map(agent => [agent.agent_id, agentGroup(agent)]));
      const hosts = new Set(startedAgents.map(agentGroup).filter(host => host && host !== 'pending'));
      const finishEvent = latestEventOf('campaign_finished');
      const finish = finishEvent?.payload || {};
      const messageEvents = events.filter(event => event.event === 'message_sent');
      const crossNodeMessages = messageEvents.filter(event => {
        const p = event.payload || {};
        const senderHost = hostByAgent.get(p.sender);
        const recipientHost = hostByAgent.get(p.recipient);
        return senderHost && recipientHost && senderHost !== recipientHost;
      });
      const maceResults = events.filter(event => (
        ['tool_call_finished', 'chemgraph_job_result'].includes(event.event)
        && event.payload?.tool_name === 'run_mace_ensemble'
      ));
      const values = [
        ['Finish', finishEvent
          ? (finish.all_agents_done ? 'done' : 'stopped')
          : (startedAgents.length ? 'running' : 'idle')],
        ['Decisions', counts.agent_decision || 0],
        ['Agents / Hosts', `${startedAgents.length} / ${hosts.size}`],
        ['Errors', counts.agent_error || 0],
        ['Messages', messageEvents.length],
        ['Cross-node', crossNodeMessages.length],
        ['Tool calls', counts.tool_call_started || 0],
        ['Workflows', counts.workflow_started || 0],
        ['Graph transitions', (counts.graph_transition || 0)],
        ['Graph terminals', (counts.graph_terminal || 0)],
      ];
      document.getElementById('metrics').innerHTML = values.map(([k,v]) => `
        <div class="metric"><div class="label">${esc(k)}</div><div class="value">${esc(v)}</div></div>
      `).join('');
      document.getElementById('proof').innerHTML = crossNodeMessages.length
        ? `<span class="pill ok">cross-node messages=${crossNodeMessages.length}</span>`
        : '';
      renderCorrelations(events);
    }

    // ponytail: correlation panel was tied to graph_round_finished events
    // from the removed behavior_graph runner. Keep the no-op so callers
    // don't have to change; delete the DOM anchor from index.html if we
    // reclaim the space later.
    function renderCorrelations(_events) {
      const container = document.getElementById('correlations');
      if (container) container.innerHTML = '';
    }

    function renderWorkflowMetrics() {
      const events = visibleEvents();
      const counts = {};
      events.forEach(event => { counts[event.event] = (counts[event.event] || 0) + 1; });
      const status = snapshot.status || {};
      const finish = events.filter(event => event.event === 'workflow_finished').slice(-1)[0]?.payload || {};
      const toolResults = events.filter(event => event.event === 'tool_call_finished' && event.payload?.runtime);
      const tokenEvents = workflowTokenEvents(events);
      const tokenTotals = summedTokenCounts(events);
      const values = [
        ['Status', finish.status || status.status || 'running'],
        ['Workflow', status.workflow_type || finish.workflow_type || '-'],
        ['Events', events.length],
        ['LM calls', tokenEvents.length || (counts.llm_decision || 0)],
        ['LM tokens', tokenTotals ? formatTokenCount(tokenTotals.total) : '-'],
        ['Tool results', toolResults.length],
        ['Errors', finish.status === 'failed' ? 1 : 0],
        ['Model', status.model_name || '-'],
        ['Span', trunc(status.workflow_span_id || finish.span_id || '-', 18)],
      ];
      document.getElementById('metrics').innerHTML = values.map(([k,v]) => `
        <div class="metric"><div class="label">${esc(k)}</div><div class="value">${esc(v)}</div></div>
      `).join('');
      document.getElementById('proof').innerHTML = '<span class="pill">local ChemGraph workflow</span>';
    }

    function workflowGraphEvents() {
      return visibleEvents().filter(event => isWorkflowEvent(event));
    }

    function activeEmbeddedWorkflowContext() {
      if (isWorkflowMode()) return null;
      const activity = selectedActivityEvent();
      if (activity) {
        const events = workflowEventsForSelection(activity);
        if (events.length) {
          return {
            events,
            anchorKey: eventKey(activity),
            title: isWorkflowEvent(activity)
              ? `ChemGraph: ${workflowAgentId(activity) || activity.agent_id || 'workflow'}`
              : `ChemGraph: ${activity.agent_id || 'agent'}`,
            meta: embeddedWorkflowMeta(events),
          };
        }
      }
      return null;
    }

    function embeddedWorkflowMeta(events) {
      const flow = workflowFlowGraph(events);
      const tokenEventCount = workflowTokenEvents(events).length;
      const llmCount = tokenEventCount || events.filter(event => event.event === 'llm_decision').length;
      const toolCount = flow.nodes.filter(node => node.type === 'tool').length;
      const tokenTotals = summedTokenCounts(events);
      const first = events[0];
      const p = first?.payload || {};
      return [
        p.thread_id || (p.round !== undefined ? `round ${p.round}` : ''),
        `${llmCount} LM`,
        tokenTotals ? `${formatTokenCount(tokenTotals.total)} tok` : '',
        `${toolCount} tools/actions`,
        `${events.length} events`,
      ].filter(Boolean).join(' · ');
    }

    function renderEmbeddedWorkflowPanel() {
      const context = activeEmbeddedWorkflowContext();
      const panel = document.getElementById('workflowFloatingPanel');
      const tab = document.getElementById('workflowFloatingTab');
      if (!context || !context.events.length) {
        panel.classList.add('hidden');
        tab.classList.add('hidden');
        return;
      }
      if (!workflowPanelOpen) {
        panel.classList.add('hidden');
        tab.classList.remove('hidden');
        return;
      }
      tab.classList.add('hidden');
      panel.classList.remove('hidden');
      applyWorkflowPanelFrame();
      if (embeddedWorkflowAnchorKey !== context.anchorKey) {
        embeddedWorkflowAnchorKey = context.anchorKey;
        selectedEmbeddedWorkflowEventKey = null;
        embeddedWorkflowView = null;
      }
      if (
        selectedEmbeddedWorkflowEventKey
        && !context.events.some(event => eventKey(event) === selectedEmbeddedWorkflowEventKey)
      ) {
        selectedEmbeddedWorkflowEventKey = null;
      }
      document.getElementById('workflowFloatingTitle').textContent = context.title;
      document.getElementById('workflowFloatingMeta').textContent = context.meta;
      renderEmbeddedWorkflowGraph(context.events);
      renderEmbeddedWorkflowInspector(context.events);
    }

    function currentEmbeddedWorkflowInspectorIdentity(events) {
      const selected = selectedEmbeddedWorkflowEvent(events);
      return [
        embeddedWorkflowAnchorKey || '',
        selected ? eventKey(selected) : 'summary',
      ].join('|');
    }

    function applyWorkflowPanelFrame() {
      const panel = document.getElementById('workflowFloatingPanel');
      const maxWidth = Math.max(520, window.innerWidth - 24);
      const maxHeight = Math.max(320, window.innerHeight - 24);
      workflowPanelFrame.width = Math.min(Math.max(workflowPanelFrame.width, 520), maxWidth);
      workflowPanelFrame.height = Math.min(Math.max(workflowPanelFrame.height, 320), maxHeight);
      workflowPanelFrame.x = Math.min(Math.max(workflowPanelFrame.x, 8), window.innerWidth - 80);
      workflowPanelFrame.y = Math.min(Math.max(workflowPanelFrame.y, 8), window.innerHeight - 56);
      panel.style.left = `${workflowPanelFrame.x}px`;
      panel.style.top = `${workflowPanelFrame.y}px`;
      panel.style.width = `${workflowPanelFrame.width}px`;
      panel.style.height = `${workflowPanelFrame.height}px`;
    }

    function renderEmbeddedWorkflowGraph(events) {
      const svg = document.getElementById('embeddedWorkflowGraph');
      const empty = document.getElementById('embeddedWorkflowEmpty');
      const flow = workflowFlowGraph(events);
      const nodes = flow.nodes;
      const edges = flow.edges;
      if (!nodes.length) {
        svg.innerHTML = '';
        empty.textContent = 'No ChemGraph workflow nodes visible for this selection.';
        empty.classList.remove('hidden');
        return;
      }
      empty.classList.add('hidden');

      const nodeW = 190;
      const nodeH = 66;
      const columnGap = 112;
      const toolLaneGap = 82;
      const toolStartY = 260;
      const maxColumn = Math.max(...nodes.map(node => node.column || 0));
      const maxToolLanes = Math.max(1, ...nodes.filter(node => node.type === 'tool').map(node => node.laneCount || 1));
      const width = Math.max(1120, 120 + (maxColumn + 1) * (nodeW + columnGap));
      const height = Math.max(540, toolStartY + maxToolLanes * toolLaneGap + 100);
      const yByType = {input: 236, lm: 128, output: 236};
      const positions = new Map();
      nodes.forEach(node => {
        const column = node.column || 0;
        const y = node.type === 'tool'
          ? toolStartY + (node.laneIndex || 0) * toolLaneGap
          : (yByType[node.type] || 236);
        positions.set(node.id, {
          x: 96 + nodeW / 2 + column * (nodeW + columnGap),
          y,
        });
      });

      const selectedEvent = selectedEmbeddedWorkflowEvent(events);
      const selectedNodeId = selectedEvent ? workflowFlowNodeId(selectedEvent) : null;
      const current = currentEvent();
      const currentNodeId = current ? workflowFlowNodeId(current) : null;
      const selectedEdgeIds = new Set();
      if (selectedNodeId) {
        edges.forEach(edge => {
          if (edge.from === selectedNodeId || edge.to === selectedNodeId) {
            selectedEdgeIds.add(`${edge.from}->${edge.to}`);
          }
        });
      }
      const nodeById = new Map(nodes.map(node => [node.id, node]));
      const edgeSvg = edges.map(edge => {
        const prev = nodeById.get(edge.from);
        const node = nodeById.get(edge.to);
        const source = positions.get(edge.from);
        const target = positions.get(edge.to);
        if (!prev || !node || !source || !target) return '';
        const startX = source.x + nodeW / 2;
        const endX = target.x - nodeW / 2;
        const midX = (startX + endX) / 2;
        const controlY = Math.min(source.y, target.y) - 54;
        const path = `M ${startX.toFixed(1)} ${source.y.toFixed(1)} Q ${midX.toFixed(1)} ${controlY.toFixed(1)} ${endX.toFixed(1)} ${target.y.toFixed(1)}`;
        const cls = [
          'workflow-edge',
          node.id === currentNodeId || prev.id === currentNodeId ? 'current' : '',
          selectedEdgeIds.has(`${edge.from}->${edge.to}`) ? 'related' : '',
        ].filter(Boolean).join(' ');
        return `
          <path class="${cls}" d="${path}" marker-end="url(#embeddedWorkflowArrow)">
            <title>${esc(prev.title)} -> ${esc(node.title)}</title>
          </path>
        `;
      }).join('');

      const nodeSvg = nodes.map(node => {
        const pos = positions.get(node.id);
        const classes = [
          'workflow-node',
          node.type,
          node.toolClass || '',
          node.failed ? 'error' : '',
          node.id === currentNodeId ? 'current' : '',
          node.id === selectedNodeId ? 'selected' : '',
        ].filter(Boolean).join(' ');
        return `
          <g class="${classes}" data-embedded-workflow-event-key="${esc(node.eventKey)}" transform="translate(${(pos.x - nodeW / 2).toFixed(1)} ${(pos.y - nodeH / 2).toFixed(1)})">
            <rect width="${nodeW}" height="${nodeH}" rx="8"></rect>
            <text class="workflow-node-title" x="12" y="21">${esc(trunc(node.title, 25))}</text>
            <text class="workflow-node-meta" x="12" y="40">${esc(trunc(node.meta, 34))}</text>
            <text class="workflow-node-meta" x="12" y="56">${esc(formatTime(node.event.timestamp))}</text>
            <title>${esc(formatWorkflowEvent(node.event))}</title>
          </g>
        `;
      }).join('');

      ensureEmbeddedWorkflowView(width, height);
      svg.innerHTML = `
        <defs>
          <marker id="embeddedWorkflowArrow" markerWidth="9" markerHeight="7" refX="8" refY="3.5" orient="auto" markerUnits="strokeWidth">
            <path d="M0,0 L9,3.5 L0,7 Z" fill="#7b8794"></path>
          </marker>
        </defs>
        <text class="host-label" x="${(width / 2).toFixed(1)}" y="34" text-anchor="middle">
          ChemGraph turn · ${nodes.length} node(s) · ${events.length} visible event(s)
        </text>
        ${edgeSvg}
        ${nodeSvg}
      `;
      updateEmbeddedWorkflowViewBox();
      svg.querySelectorAll('[data-embedded-workflow-event-key]').forEach(node => {
        node.addEventListener('click', event => {
          selectedEmbeddedWorkflowEventKey = node.dataset.embeddedWorkflowEventKey;
          renderEmbeddedWorkflowPanel();
          event.stopPropagation();
        });
      });
    }

    function selectedEmbeddedWorkflowEvent(events) {
      if (!selectedEmbeddedWorkflowEventKey) return null;
      return events.find(event => eventKey(event) === selectedEmbeddedWorkflowEventKey) || null;
    }

    function renderEmbeddedWorkflowInspector(events) {
      const title = document.getElementById('embeddedWorkflowInspectorTitle');
      const meta = document.getElementById('embeddedWorkflowInspectorMeta');
      const body = document.getElementById('embeddedWorkflowInspectorBody');
      const identity = currentEmbeddedWorkflowInspectorIdentity(events);
      const previousIdentity = lastEmbeddedWorkflowInspectorIdentity;
      const previousScrollTop = body.scrollTop;
      const previousScrollLeft = body.scrollLeft;
      const event = selectedEmbeddedWorkflowEvent(events);
      if (!event) {
        const tokenTotals = summedTokenCounts(events);
        const flow = workflowFlowGraph(events);
        const html = detailRich(
          detailSection('Turn Summary', detailKvGrid([
            ['Visible events', events.length],
            ['Nodes', flow.nodes.length],
            ['LM calls', workflowTokenEvents(events).length || events.filter(item => item.event === 'llm_decision').length],
            ['Input tokens', tokenTotals?.input ?? '-'],
            ['Output tokens', tokenTotals?.output ?? '-'],
            ['Total tokens', tokenTotals?.total ?? '-'],
          ]), 'info'),
          detailSection(
            'Inspect',
            paragraphsHtml('Click a ChemGraph node in this panel to inspect its LM tokens, tool arguments, output, and payload without changing the outer dashboard selection.'),
          ),
        );
        title.textContent = 'ChemGraph Inspector';
        meta.textContent = 'Select an LM, tool, action, or output node.';
        setStableHtml(body, html, identity === previousIdentity);
        if (identity === previousIdentity) {
          body.scrollTop = previousScrollTop;
          body.scrollLeft = previousScrollLeft;
        }
        lastEmbeddedWorkflowInspectorIdentity = identity;
        return;
      }
      const html = detailRich(
        chemgraphNodeDetailHtml(event),
        payloadDetailHtml(event.payload || {}),
        chemgraphNodeContextHtml(event),
      );
      title.textContent = chemgraphNodeDetailTitle(event);
      meta.textContent = `${formatTime(event.timestamp)} · ${event.event}`;
      setStableHtml(body, html, identity === previousIdentity);
      if (identity === previousIdentity) {
        body.scrollTop = previousScrollTop;
        body.scrollLeft = previousScrollLeft;
      }
      lastEmbeddedWorkflowInspectorIdentity = identity;
    }

    function ensureEmbeddedWorkflowView(width, height) {
      const padX = Math.max(180, width * 0.08);
      const padY = Math.max(100, height * 0.12);
      const bounds = {
        x: -padX,
        y: -padY,
        width: width + padX * 2,
        height: height + padY * 2,
      };
      if (!embeddedWorkflowView) {
        embeddedWorkflowView = {
          x: bounds.x,
          y: bounds.y,
          width: bounds.width,
          height: bounds.height,
          layoutWidth: width,
          layoutHeight: height,
          boundsX: bounds.x,
          boundsY: bounds.y,
          boundsWidth: bounds.width,
          boundsHeight: bounds.height,
        };
        return;
      }
      if (
        embeddedWorkflowView.layoutWidth !== width
        || embeddedWorkflowView.layoutHeight !== height
      ) {
        const nextView = preserveViewForLayoutChange(
          embeddedWorkflowView,
          bounds,
          width,
          height,
        );
        embeddedWorkflowView = {
          ...embeddedWorkflowView,
          ...nextView,
          layoutWidth: width,
          layoutHeight: height,
          boundsX: bounds.x,
          boundsY: bounds.y,
          boundsWidth: bounds.width,
          boundsHeight: bounds.height,
        };
        clampEmbeddedWorkflowView();
      }
    }

    function updateEmbeddedWorkflowViewBox() {
      const svg = document.getElementById('embeddedWorkflowGraph');
      if (!embeddedWorkflowView) return;
      svg.setAttribute(
        'viewBox',
        `${embeddedWorkflowView.x.toFixed(1)} ${embeddedWorkflowView.y.toFixed(1)} ${embeddedWorkflowView.width.toFixed(1)} ${embeddedWorkflowView.height.toFixed(1)}`
      );
    }

    function clampEmbeddedWorkflowView() {
      if (!embeddedWorkflowView) return;
      const boundsX = embeddedWorkflowView.boundsX ?? 0;
      const boundsY = embeddedWorkflowView.boundsY ?? 0;
      const boundsWidth = embeddedWorkflowView.boundsWidth ?? embeddedWorkflowView.layoutWidth;
      const boundsHeight = embeddedWorkflowView.boundsHeight ?? embeddedWorkflowView.layoutHeight;
      embeddedWorkflowView.width = Math.min(boundsWidth, Math.max(embeddedWorkflowView.layoutWidth / 12, embeddedWorkflowView.width));
      embeddedWorkflowView.height = Math.min(boundsHeight, Math.max(embeddedWorkflowView.layoutHeight / 12, embeddedWorkflowView.height));
      embeddedWorkflowView.x = Math.min(Math.max(boundsX, embeddedWorkflowView.x), boundsX + boundsWidth - embeddedWorkflowView.width);
      embeddedWorkflowView.y = Math.min(Math.max(boundsY, embeddedWorkflowView.y), boundsY + boundsHeight - embeddedWorkflowView.height);
    }

    function zoomEmbeddedWorkflow(factor) {
      if (!embeddedWorkflowView) return;
      const centerX = embeddedWorkflowView.x + embeddedWorkflowView.width / 2;
      const centerY = embeddedWorkflowView.y + embeddedWorkflowView.height / 2;
      embeddedWorkflowView.width *= factor;
      embeddedWorkflowView.height *= factor;
      embeddedWorkflowView.x = centerX - embeddedWorkflowView.width / 2;
      embeddedWorkflowView.y = centerY - embeddedWorkflowView.height / 2;
      clampEmbeddedWorkflowView();
      updateEmbeddedWorkflowViewBox();
    }

    function resetEmbeddedWorkflowView() {
      embeddedWorkflowView = null;
      renderEmbeddedWorkflowPanel();
    }

    function renderWorkflowGraph() {
      const svg = document.getElementById('graph');
      const events = workflowGraphEvents();
      document.getElementById('hostLegend').innerHTML = `
        <span class="legend-item"><span class="legend-swatch" style="background:#fef3c7;border-color:#d97706"></span>query</span>
        <span class="legend-item"><span class="legend-swatch" style="background:#dbeafe;border-color:#2563eb"></span>LM</span>
        <span class="legend-item"><span class="legend-swatch" style="background:#fffbeb;border-color:#d97706"></span>action tool</span>
        <span class="legend-item"><span class="legend-swatch" style="background:#dcfce7;border-color:#16a34a"></span>science tool</span>
        <span class="legend-item"><span class="legend-swatch" style="background:#ede9fe;border-color:#7c3aed"></span>output</span>
        <span class="legend-item"><span class="legend-swatch" style="background:#fee2e2;border-color:#dc2626"></span>failure</span>
      `;
      if (!events.length) {
        svg.setAttribute('viewBox', '0 0 1000 260');
        svg.innerHTML = '<text x="500" y="130" text-anchor="middle" class="host-label">No ChemGraph workflow events yet.</text>';
        return;
      }

      const flow = workflowFlowGraph(events);
      const nodes = flow.nodes;
      const edges = flow.edges;
      if (!nodes.length) {
        svg.setAttribute('viewBox', '0 0 1000 260');
        svg.innerHTML = '<text x="500" y="130" text-anchor="middle" class="host-label">Waiting for ChemGraph workflow execution events.</text>';
        return;
      }

      const nodeW = 184;
      const nodeH = 64;
      const columnGap = 96;
      const toolLaneGap = 76;
      const toolStartY = 230;
      const maxColumn = Math.max(...nodes.map(node => node.column || 0));
      const maxToolLanes = Math.max(1, ...nodes.filter(node => node.type === 'tool').map(node => node.laneCount || 1));
      const width = Math.max(1040, 120 + (maxColumn + 1) * (nodeW + columnGap));
      const height = Math.max(500, toolStartY + maxToolLanes * toolLaneGap + 80);
      const yByType = {input: 220, lm: 126, output: 220};
      const positions = new Map();
      nodes.forEach(node => {
        const column = node.column || 0;
        const y = node.type === 'tool'
          ? toolStartY + (node.laneIndex || 0) * toolLaneGap
          : (yByType[node.type] || 220);
        positions.set(node.id, {
          x: 96 + nodeW / 2 + column * (nodeW + columnGap),
          y,
        });
      });
      const current = currentEvent();
      const selectedEvent = selectedActivityEvent();
      const currentNodeId = current ? workflowFlowNodeId(current) : null;
      const selectedNodeId = selectedEvent ? workflowFlowNodeId(selectedEvent) : null;
      const selectedEdgeIds = new Set();
      if (selectedNodeId) {
        edges.forEach(edge => {
          if (edge.from === selectedNodeId || edge.to === selectedNodeId) {
            selectedEdgeIds.add(`${edge.from}->${edge.to}`);
          }
        });
      }

      const nodeById = new Map(nodes.map(node => [node.id, node]));
      const edgeSvg = edges.map(edge => {
        const prev = nodeById.get(edge.from);
        const node = nodeById.get(edge.to);
        const source = positions.get(edge.from);
        const target = positions.get(edge.to);
        if (!prev || !node || !source || !target) return '';
        const startX = source.x + nodeW / 2;
        const endX = target.x - nodeW / 2;
        const midX = (startX + endX) / 2;
        const controlY = Math.min(source.y, target.y) - 46;
        const path = `M ${startX.toFixed(1)} ${source.y.toFixed(1)} Q ${midX.toFixed(1)} ${controlY.toFixed(1)} ${endX.toFixed(1)} ${target.y.toFixed(1)}`;
        const cls = [
          'workflow-edge',
          node.id === currentNodeId || prev.id === currentNodeId ? 'current' : '',
          selectedEdgeIds.has(`${edge.from}->${edge.to}`) ? 'related' : '',
        ].filter(Boolean).join(' ');
        return `
          <path class="${cls}" d="${path}" marker-end="url(#workflowArrow)">
            <title>${esc(prev.title)} -> ${esc(node.title)}</title>
          </path>
        `;
      }).join('');

      const nodeSvg = nodes.map(node => {
        const pos = positions.get(node.id);
        const classes = [
          'workflow-node',
          node.type,
          node.toolClass || '',
          node.failed ? 'error' : '',
          node.id === currentNodeId ? 'current' : '',
          node.id === selectedNodeId ? 'selected' : '',
        ].filter(Boolean).join(' ');
        return `
          <g class="${classes}" data-activity-event-key="${esc(node.eventKey)}" transform="translate(${(pos.x - nodeW / 2).toFixed(1)} ${(pos.y - nodeH / 2).toFixed(1)})">
            <rect width="${nodeW}" height="${nodeH}" rx="8"></rect>
            <text class="workflow-node-title" x="12" y="21">${esc(trunc(node.title, 24))}</text>
            <text class="workflow-node-meta" x="12" y="40">${esc(trunc(node.meta, 32))}</text>
            <text class="workflow-node-meta" x="12" y="55">${esc(formatTime(node.event.timestamp))}</text>
            <title>${esc(formatWorkflowEvent(node.event))}</title>
          </g>
        `;
      }).join('');

      svg.style.minHeight = `${height}px`;
      ensureGraphView(width, height);
      svg.innerHTML = `
        <defs>
          <marker id="workflowArrow" markerWidth="9" markerHeight="7" refX="8" refY="3.5" orient="auto" markerUnits="strokeWidth">
            <path d="M0,0 L9,3.5 L0,7 Z" fill="#7b8794"></path>
          </marker>
        </defs>
        <text class="host-label" x="${(width / 2).toFixed(1)}" y="34" text-anchor="middle">
          ${esc(snapshot.status?.workflow_type || 'ChemGraph workflow')} · ${nodes.length} flow node(s) · ${events.length} visible event(s)
        </text>
        ${edgeSvg}
        ${nodeSvg}
      `;
      updateGraphViewBox();
      svg.querySelectorAll('[data-activity-event-key]').forEach(activityEl => {
        activityEl.addEventListener('click', event => {
          selectedActivityEventKey = activityEl.dataset.activityEventKey;
          selectedAgent = null;
          selectedEdgeKey = null;
          event.stopPropagation();
          render();
        });
      });
    }

    function workflowFlowGraph(events) {
      const nodes = [];
      const edges = [];
      const toolNodeIndexes = new Map();
      const hasGraphWork = events.some(event => (
        event.event === 'llm_decision'
        || event.event === 'workflow_output'
        || event.event === 'run_finished'
        || event.event.startsWith('tool_call_')
      ));
      const runStart = events.find(event => event.event === 'run_started')
        || events.find(event => event.event === 'workflow_started');
      let lastColumn = -1;
      let lastNodeIds = [];
      let currentLmId = null;
      let currentToolBatchIds = [];
      let currentToolBatchColumn = null;
      let lmTurn = 0;

      function addEdge(from, to) {
        if (!from || !to || from === to) return;
        if (!edges.some(edge => edge.from === from && edge.to === to)) {
          edges.push({from, to});
        }
      }

      function addEdges(fromIds, to) {
        Array.from(new Set(fromIds.filter(Boolean))).forEach(from => addEdge(from, to));
      }

      function updateCurrentToolBatchLanes() {
        currentToolBatchIds.forEach((id, laneIndex) => {
          const index = toolNodeIndexes.get(id);
          if (index === undefined) return;
          nodes[index].laneIndex = laneIndex;
          nodes[index].laneCount = currentToolBatchIds.length;
        });
      }

      if (runStart && hasGraphWork) {
        const node = workflowFlowNode(runStart, 0);
        node.column = 0;
        nodes.push(node);
        lastColumn = 0;
        lastNodeIds = [node.id];
      }

      events.forEach(event => {
        if (event.event === 'llm_decision') {
          lmTurn += 1;
          const afterToolBatch = currentToolBatchIds.length > 0;
          const node = workflowFlowNode(event, nodes.length, {lmTurn});
          node.column = afterToolBatch
            ? currentToolBatchColumn + 1
            : lastColumn + 1;
          nodes.push(node);
          addEdges(afterToolBatch ? currentToolBatchIds : lastNodeIds, node.id);
          lastColumn = node.column;
          lastNodeIds = [node.id];
          currentLmId = node.id;
          currentToolBatchIds = [];
          currentToolBatchColumn = null;
          return;
        }
        if (event.event.startsWith('tool_call_')) {
          if (currentToolBatchColumn === null) {
            currentToolBatchColumn = lastColumn + 1;
          }
          const node = workflowFlowNode(event, nodes.length);
          if (toolNodeIndexes.has(node.id)) {
            const index = toolNodeIndexes.get(node.id);
            node.column = nodes[index].column;
            node.laneIndex = nodes[index].laneIndex;
            node.laneCount = nodes[index].laneCount;
            nodes[index] = node;
          } else {
            node.column = currentToolBatchColumn;
            node.laneIndex = currentToolBatchIds.length;
            node.laneCount = currentToolBatchIds.length + 1;
            toolNodeIndexes.set(node.id, nodes.length);
            currentToolBatchIds.push(node.id);
            nodes.push(node);
            addEdges(currentLmId ? [currentLmId] : lastNodeIds, node.id);
            updateCurrentToolBatchLanes();
          }
        }
      });

      const output = events.filter(event => event.event === 'workflow_output').slice(-1)[0]
        || (
          nodes.length
            ? events.filter(event => event.event === 'workflow_finished' || event.event === 'run_finished').slice(-1)[0]
            : null
        );
      if (output) {
        const afterToolBatch = currentToolBatchIds.length > 0;
        const node = workflowFlowNode(output, nodes.length);
        node.column = afterToolBatch
          ? currentToolBatchColumn + 1
          : lastColumn + 1;
        nodes.push(node);
        addEdges(afterToolBatch ? currentToolBatchIds : lastNodeIds, node.id);
      }
      return {nodes, edges};
    }

    function workflowFlowNodeId(event) {
      const p = event?.payload || {};
      if (!event) return null;
      if (event.event === 'run_started' || event.event === 'workflow_started') return 'workflow-query';
      if (event.event === 'workflow_output' || event.event === 'workflow_finished' || event.event === 'run_finished') return 'workflow-output';
      if (event.event.startsWith('tool_call_')) {
        return p.tool_call_id ? `workflow-tool-${p.tool_call_id}` : (p.span_id || eventKey(event));
      }
      return p.span_id || eventKey(event);
    }

    function toolCallSummary(toolCalls) {
      const counts = new Map();
      toolCalls
        .map(call => call?.name || call?.id || 'tool')
        .filter(Boolean)
        .forEach(name => counts.set(name, (counts.get(name) || 0) + 1));
      const parts = Array.from(counts.entries()).map(([name, count]) => count > 1 ? `${name} x${count}` : name);
      return parts.join(', ');
    }

    function numberOrNull(value) {
      const number = Number(value);
      return Number.isFinite(number) ? number : null;
    }

    function tokenField(raw, names) {
      if (!raw || typeof raw !== 'object') return null;
      for (const name of names) {
        if (raw[name] !== undefined && raw[name] !== null) {
          const value = numberOrNull(raw[name]);
          if (value !== null) return value;
        }
      }
      return null;
    }

    function llmTokenCounts(event) {
      const raw = event?.payload?.token_counts;
      if (!raw || typeof raw !== 'object') return null;
      const input = tokenField(raw, ['input_tokens', 'prompt_tokens']);
      const output = tokenField(raw, ['output_tokens', 'completion_tokens']);
      let total = tokenField(raw, ['total_tokens']);
      if (total === null && (input !== null || output !== null)) {
        total = (input || 0) + (output || 0);
      }
      if (input === null && output === null && total === null) return null;
      return {
        input,
        output,
        total,
        source: raw.source || 'unknown',
        estimateScope: raw.estimate_scope || '',
        rawUsage: raw.raw_usage,
        raw,
      };
    }

    function workflowTokenEvents(events) {
      return events
        .map(event => ({event, counts: llmTokenCounts(event)}))
        .filter(item => item.counts);
    }

    function summedTokenCounts(events) {
      const items = workflowTokenEvents(events);
      if (!items.length) return null;
      let input = 0;
      let output = 0;
      let total = 0;
      let sawInput = false;
      let sawOutput = false;
      let sawTotal = false;
      const sources = new Set();
      items.forEach(({counts}) => {
        if (counts.input !== null) {
          input += counts.input;
          sawInput = true;
        }
        if (counts.output !== null) {
          output += counts.output;
          sawOutput = true;
        }
        if (counts.total !== null) {
          total += counts.total;
          sawTotal = true;
        }
        if (counts.source) sources.add(counts.source);
      });
      if (!sawTotal && (sawInput || sawOutput)) total = input + output;
      return {
        input: sawInput ? input : null,
        output: sawOutput ? output : null,
        total: sawTotal || sawInput || sawOutput ? total : null,
        source: Array.from(sources).join('+') || 'unknown',
      };
    }

    function formatTokenCount(value) {
      const number = numberOrNull(value);
      if (number === null) return '-';
      if (Math.abs(number) >= 1000000) return `${(number / 1000000).toFixed(1)}M`;
      if (Math.abs(number) >= 10000) return `${Math.round(number / 1000)}k`;
      if (Math.abs(number) >= 1000) return `${(number / 1000).toFixed(1)}k`;
      return String(Math.round(number));
    }

    function tokenSourceLabel(source) {
      if (source === 'local_estimate') return 'estimate';
      if (source === 'provider') return 'provider';
      return source || 'unknown';
    }

    function tokenSummary(event, {compact = false} = {}) {
      const counts = llmTokenCounts(event);
      if (!counts) return '';
      const source = tokenSourceLabel(counts.source);
      if (compact) {
        const sourceSuffix = source === 'estimate' ? ' · est' : source === 'provider' ? '' : ` · ${source}`;
        return `tok ${formatTokenCount(counts.input)} in / ${formatTokenCount(counts.output)} out${sourceSuffix}`;
      }
      return [
        `input ${formatTokenCount(counts.input)}`,
        `output ${formatTokenCount(counts.output)}`,
        `total ${formatTokenCount(counts.total)}`,
        source,
      ].filter(Boolean).join(' · ');
    }

    function promptDisclosureHtml(event) {
      const messages = event?.payload?.prompt_messages;
      if (!Array.isArray(messages) || !messages.length) return '';
      return `
        <details class="detail-raw detail-prompt">
          <summary>Show full prompt (${messages.length} messages)</summary>
          <pre>${esc(formatJson(messages))}</pre>
        </details>
      `;
    }

    function promptDetailSection(event) {
      if (!llmTokenCounts(event)) return '';
      const disclosure = promptDisclosureHtml(event);
      return detailSection(
        'Prompt',
        disclosure || paragraphsHtml('Full prompt was not captured for this event. Rerun with the updated ChemGraph observability code to populate this field.'),
        disclosure ? 'info' : 'warn',
      );
    }

    function tokenDetailSection(event, title = 'Token Counts') {
      const counts = llmTokenCounts(event);
      if (!counts) return '';
      const body = [
        detailKvGrid([
          ['Input tokens', counts.input === null ? '-' : Math.round(counts.input)],
          ['Output tokens', counts.output === null ? '-' : Math.round(counts.output)],
          ['Total tokens', counts.total === null ? '-' : Math.round(counts.total)],
          ['Source', tokenSourceLabel(counts.source)],
          ['Estimate scope', counts.estimateScope || ''],
        ]),
      ].filter(Boolean).join('');
      return [
        detailSection(title, body, counts.source === 'provider' ? 'ok' : 'info'),
        counts.rawUsage ? detailSection('Provider Usage', detailValueHtml(counts.rawUsage)) : '',
      ].filter(Boolean).join('');
    }

    function workflowToolKind(event) {
      const name = toolDisplayName(event);
      return actionToolNames.has(name) ? 'action-tool' : 'science-tool';
    }

    function workflowToolKindLabel(event) {
      return workflowToolKind(event) === 'action-tool' ? 'action' : 'science';
    }

    function workflowFlowNode(event, index, options = {}) {
      const p = event.payload || {};
      const key = eventKey(event);
      if (event.event === 'run_started' || event.event === 'workflow_started') {
        return {
          id: workflowFlowNodeId(event),
          type: 'input',
          title: p.nested ? 'Prompt' : 'Query',
          meta: p.query || p.thread_id || (p.round !== undefined ? `round ${p.round}` : snapshot.status?.query || 'user request'),
          event,
          eventKey: key,
          failed: false,
        };
      }
      if (event.event === 'llm_decision') {
        const calls = Array.isArray(p.tool_calls) ? p.tool_calls : [];
        const callNames = toolCallSummary(calls);
        const tokens = tokenSummary(event, {compact: true});
        return {
          id: workflowFlowNodeId(event),
          type: 'lm',
          title: `LM turn ${options.lmTurn || index}`,
          meta: [tokens, callNames ? `calls: ${callNames}` : 'response'].filter(Boolean).join(' · '),
          event,
          eventKey: key,
          failed: false,
        };
      }
      if (event.event.startsWith('tool_call_')) {
        const failed = event.event === 'tool_call_failed' || p.status === 'failed';
        const kind = workflowToolKind(event);
        // ponytail: prefix title with a serial ("#N ") once the batch
        // has more than one lane so operators don't misread stacked
        // tool nodes as parallel invocations. Multi_agent's executor
        // calls tools sequentially within one LM turn but the graph
        // draws them side-by-side in lanes; the number restores order.
        return {
          id: workflowFlowNodeId(event),
          type: 'tool',
          toolClass: kind,
          title: toolDisplayName(event),
          meta: `${workflowToolKindLabel(event)} · ${failed ? 'failed' : event.event === 'tool_call_started' ? 'running' : 'finished'} · serial (not parallel)`,
          event,
          eventKey: key,
          failed,
        };
      }
      return {
        id: workflowFlowNodeId(event),
        type: 'output',
        title: 'Output',
        meta: tokenSummary(event, {compact: true}) || p.content_preview || p.status || snapshot.status?.status || 'finished',
        event,
        eventKey: key,
        failed: p.status === 'failed',
      };
    }

    function renderGraph() {
      if (isWorkflowMode()) {
        renderWorkflowGraph();
        return;
      }
      const svg = document.getElementById('graph');
      const currentAgents = agents();
      if (!currentAgents.length) {
        svg.setAttribute('viewBox', '0 0 1000 260');
        svg.innerHTML = '<text x="500" y="130" text-anchor="middle" class="host-label">No agents yet.</text>';
        document.getElementById('hostLegend').innerHTML = '<span class="muted">Waiting for placement.</span>';
        return;
      }

      // Group by site (federated) or hostname (single-site). The
      // graph layout is unchanged either way -- only the swimlane
      // labels and the band per-agent-group differ. In federated
      // mode you see "aurora" and "crux" swimlanes instead of
      // "x4708..." and "x1000...". Same nodes, clearer story.
      const byHost = new Map();
      currentAgents.forEach(agent => {
        const host = agentGroup(agent);
        if (!byHost.has(host)) byHost.set(host, []);
        byHost.get(host).push(agent);
      });
      const hosts = Array.from(byHost.keys()).sort();
      const maxPerHost = Math.max(...hosts.map(host => byHost.get(host).length));
      const radial = currentAgents.length >= 2;
      const width = radial
        ? Math.max(1200, currentAgents.length * 220, hosts.length * 220)
        : 1000;
      const height = radial
        ? Math.max(760, currentAgents.length * 135)
        : Math.max(340, 110 + maxPerHost * 88);
      const marginX = 42;
      const top = 58;
      const bottom = 34;
      const laneGap = radial ? 8 : 14;
      const laneWidth = (width - marginX * 2 - laneGap * Math.max(0, hosts.length - 1)) / hosts.length;
      const nodeW = radial ? 132 : Math.min(154, Math.max(82, laneWidth - 18));
      const nodeH = radial ? 58 : 58;
      const positions = new Map();
      svg.style.minWidth = '';
      svg.style.minHeight = `${height}px`;

      const hostIndex = new Map(hosts.map((host, index) => [host, index]));
      const legendPrefix = `
        <span class="legend-item"><span class="legend-swatch" style="background:#ccfbf1;border-color:#0f766e"></span>ChemGraph turn chip</span>
        ${radial ? '<span class="legend-item"><span class="muted">radial host layout</span></span>' : ''}
      `;
      const legend = legendPrefix + hosts.map((host, index) => `
        <span class="legend-item">
          <span class="legend-swatch" style="background:${hostColor(index)};border-color:${hostStroke(index)}"></span>
          ${esc(host)} <span class="muted">(${byHost.get(host).length})</span>
        </span>
      `).join('');
      document.getElementById('hostLegend').innerHTML = legend;

      let bands = '';
      if (radial) {
        const centerX = width / 2;
        const centerY = height / 2;
        const radiusScale = currentAgents.length < 5 ? 0.31 : 0.36;
        const radiusX = width * radiusScale;
        const radiusY = height * (currentAgents.length < 5 ? 0.30 : 0.34);
        const hostCenters = new Map();
        bands = `
          <ellipse cx="${centerX}" cy="${centerY}" rx="${radiusX}" ry="${radiusY}" fill="#fff" stroke="#d7dde5" stroke-width="1"></ellipse>
          <ellipse cx="${centerX}" cy="${centerY}" rx="${(radiusX * 0.62).toFixed(1)}" ry="${(radiusY * 0.62).toFixed(1)}" fill="none" stroke="#edf1f5" stroke-width="1"></ellipse>
          <text class="host-label" x="${centerX}" y="34" text-anchor="middle">${currentAgents.length} daemon agents across ${hosts.length} host(s)</text>
        `;
        hosts.forEach((host, index) => {
          const angle = -Math.PI / 2 + (2 * Math.PI * index) / hosts.length;
          const x = centerX + Math.cos(angle) * radiusX;
          const y = centerY + Math.sin(angle) * radiusY;
          hostCenters.set(host, {x, y, angle});
          const fill = hostColor(index);
          const stroke = hostStroke(index);
          bands += `
            <circle cx="${x.toFixed(1)}" cy="${y.toFixed(1)}" r="${Math.max(nodeW * 0.72, 72).toFixed(1)}" fill="${fill}" fill-opacity="0.18" stroke="${stroke}" stroke-opacity="0.38"></circle>
            <text class="host-label" x="${x.toFixed(1)}" y="${(y - nodeH / 2 - 16).toFixed(1)}" text-anchor="middle">${esc(trunc(host, 28))}</text>
          `;
        });
        hosts.forEach((host, hIndex) => {
          const list = byHost.get(host).slice().sort((a, b) => a.agent_id.localeCompare(b.agent_id));
          const center = hostCenters.get(host);
          const spread = list.length > 1 ? Math.max(nodeH + 22, 84) : 0;
          list.forEach((agent, index) => {
            const offset = (index - (list.length - 1) / 2) * spread;
            const tangentX = -Math.sin(center.angle);
            const tangentY = Math.cos(center.angle);
            const x = center.x + tangentX * offset;
            const y = center.y + tangentY * offset;
            positions.set(agent.agent_id, {x, y, host, hostIndex: hIndex, agent});
          });
        });
      } else {
        bands = hosts.map((host, index) => {
          const x = marginX + index * (laneWidth + laneGap);
          const fill = hostColor(index);
          const stroke = hostStroke(index);
          const label = trunc(host, 34);
          return `
            <rect class="host-band" x="${x.toFixed(1)}" y="18" width="${laneWidth.toFixed(1)}" height="${(height - 34).toFixed(1)}" rx="8" fill="${fill}" fill-opacity="0.34" stroke="${stroke}" stroke-opacity="0.42"></rect>
            <text class="host-label" x="${(x + 12).toFixed(1)}" y="39">${esc(label)}</text>
          `;
        }).join('');

        hosts.forEach((host, hIndex) => {
          const list = byHost.get(host).sort((a, b) => a.agent_id.localeCompare(b.agent_id));
          const x = marginX + hIndex * (laneWidth + laneGap) + laneWidth / 2;
          list.forEach((agent, index) => {
            const usable = height - top - bottom;
            const y = top + ((index + 1) * usable) / (list.length + 1);
            positions.set(agent.agent_id, {x, y, host, hostIndex: hIndex, agent});
          });
        });
      }

      const sent = graphMessageEvents();
      const recentIds = new Set(sent.slice(-10).map(e => e.payload?.message_id).filter(Boolean));
      const edgeMap = new Map();
      sent.forEach((event, index) => {
        const p = event.payload || {};
        if (!positions.has(p.sender) || !positions.has(p.recipient)) return;
        const key = `${p.sender}->${p.recipient}`;
        const prev = edgeMap.get(key) || {
          key,
          sender: p.sender,
          recipient: p.recipient,
          count: 0,
          latestIndex: -1,
          latestMessageId: null,
          latestTldr: '',
          latestContent: '',
          messages: [],
        };
        prev.count += 1;
        prev.latestIndex = index;
        prev.latestMessageId = p.message_id;
        prev.latestTldr = p.tldr || '';
        prev.latestContent = p.content || '';
        prev.messages.push(event);
        edgeMap.set(key, prev);
      });
      const edges = Array.from(edgeMap.values()).sort((a, b) => a.latestIndex - b.latestIndex);
      if (selectedEdgeKey && !edgeMap.has(selectedEdgeKey)) {
        selectedEdgeKey = null;
      }
      const allVisibleMessages = eventsOf('message_sent').length;
      const showEdgeLabels = Boolean(selectedEdgeKey) || (graphMode !== 'cumulative' && edges.length <= 8);

      const labelBoxes = [];
      const edgeSvg = edges.map((edge, index) => {
        const source = positions.get(edge.sender);
        const target = positions.get(edge.recipient);
        const cross = source.host !== target.host;
        const selectedByAgent = selectedAgent && (edge.sender === selectedAgent || edge.recipient === selectedAgent);
        const selectedByEdge = selectedEdgeKey === edge.key;
        const dimmed = (selectedAgent && !selectedByAgent) || (selectedEdgeKey && !selectedByEdge);
        const recent = recentIds.has(edge.latestMessageId);
        const start = edgeEndpoint(source, target, nodeW, nodeH, true);
        const end = edgeEndpoint(source, target, nodeW, nodeH, false);
        const hasReverse = edgeMap.has(`${edge.recipient}->${edge.sender}`);
        const route = curvedRoute(start, end, edge, index, radial, width / 2, height / 2, hasReverse);
        const path = route.path;
        const cls = ['edge', cross ? 'cross-node' : '', recent ? 'recent' : '', selectedByEdge ? 'selected' : '', dimmed ? 'dimmed' : ''].filter(Boolean).join(' ');
        const marker = cross ? 'url(#arrowCross)' : 'url(#arrow)';
        const labelX = route.labelX;
        const labelY = route.labelY;
        const edgeSummary = edge.latestTldr || '';
        const shouldShowSummary = edgeSummary && !dimmed && (selectedByEdge || recent || showEdgeLabels);
        const rawLabel = shouldShowSummary ? edgeSummary : (selectedByEdge || showEdgeLabels ? edge.count : '');
        const placedLabel = placeEdgeLabel(labelX, labelY, rawLabel, labelBoxes, selectedByEdge);
        const titleText = edgeSummary || edge.latestContent;
        return `
          <path class="${cls}" d="${path}" marker-end="${marker}" data-edge-key="${esc(edge.key)}">
            <title>${esc(edge.sender)} -> ${esc(edge.recipient)} (${edge.count}) ${cross ? 'cross-node' : 'same-node'}&#10;${esc(trunc(titleText, 180))}</title>
          </path>
          <path class="edge-hit" d="${path}" data-edge-key="${esc(edge.key)}"></path>
          ${placedLabel ? `<text class="edge-label ${dimmed ? 'dimmed' : ''}" x="${placedLabel.x.toFixed(1)}" y="${placedLabel.y.toFixed(1)}" text-anchor="middle" data-edge-key="${esc(edge.key)}">${esc(placedLabel.text)}</text>` : ''}
        `;
      }).join('');

      const edgeHint = !edges.length ? `
        <text class="graph-hint" x="${(width / 2).toFixed(1)}" y="${(height / 2).toFixed(1)}" text-anchor="middle">
          ${allVisibleMessages ? 'No routes visible in this graph mode. Try Recent or All.' : 'No messages visible at this time point.'}
        </text>
      ` : '';
      const bubbleSvg = renderActivityBubbles(currentAgents, positions, nodeW, nodeH, width, height);

      const nodeSvg = currentAgents.map(agent => {
        const pos = positions.get(agent.agent_id);
        const current = currentEvent();
        const hIndex = hostIndex.get(pos.host) || 0;
        const x = pos.x - nodeW / 2;
        const y = pos.y - nodeH / 2;
        const classes = [
          'agent-node',
          agent.agent_id === selectedAgent ? 'selected' : '',
          current?.agent_id === agent.agent_id ? 'current' : '',
          agent.last_error ? 'error' : '',
          !agent.started ? 'pending' : '',
          selectedEdgeKey && !selectedEdgeKey.split('->').includes(agent.agent_id) ? 'dimmed' : '',
        ].filter(Boolean).join(' ');
        const status = !agent.started ? 'pending' : agent.last_error ? 'error' : `${agent.decision_count || 0} decisions`;
        return `
          <g class="${classes}" data-agent="${esc(agent.agent_id)}" tabindex="0" transform="translate(${x.toFixed(1)} ${y.toFixed(1)})">
            <rect width="${nodeW}" height="${nodeH}" rx="7" fill="${hostColor(hIndex)}" stroke="${hostStroke(hIndex)}"></rect>
            <text class="agent-name" x="10" y="18">${esc(trunc(agent.agent_id, 20))}</text>
            <text class="agent-role" x="10" y="35">${esc(trunc(agent.role || '', 24))}</text>
            <text class="agent-host" x="10" y="51">${esc(status)}</text>
            <title>${esc(agent.agent_id)}&#10;host: ${esc(pos.host)}&#10;role: ${esc(agent.role || '')}</title>
          </g>
        `;
      }).join('');

      ensureGraphView(width, height);
      svg.innerHTML = `
        <defs>
          <marker id="arrow" markerWidth="9" markerHeight="7" refX="8" refY="3.5" orient="auto" markerUnits="strokeWidth">
            <path d="M0,0 L9,3.5 L0,7 Z" fill="#7b8794"></path>
          </marker>
          <marker id="arrowCross" markerWidth="9" markerHeight="7" refX="8" refY="3.5" orient="auto" markerUnits="strokeWidth">
            <path d="M0,0 L9,3.5 L0,7 Z" fill="#0f766e"></path>
          </marker>
        </defs>
        ${bands}
        ${edgeSvg}
        ${edgeHint}
        ${nodeSvg}
        ${bubbleSvg}
      `;
      updateGraphViewBox();
      svg.querySelectorAll('[data-edge-key]').forEach(edgeEl => {
        edgeEl.addEventListener('click', () => {
          selectedEdgeKey = edgeEl.dataset.edgeKey;
          selectedAgent = null;
          selectedActivityEventKey = null;
          render();
        });
      });
      svg.querySelectorAll('.agent-node').forEach(node => {
        node.addEventListener('click', () => {
          selectedAgent = node.dataset.agent;
          selectedEdgeKey = null;
          selectedActivityEventKey = null;
          render();
        });
        node.addEventListener('keydown', event => {
          if (event.key === 'Enter' || event.key === ' ') {
            selectedAgent = node.dataset.agent;
            selectedEdgeKey = null;
            selectedActivityEventKey = null;
            render();
          }
        });
      });
      svg.querySelectorAll('[data-activity-event-key]').forEach(activityEl => {
        activityEl.addEventListener('click', event => {
          selectedActivityEventKey = activityEl.dataset.activityEventKey;
          selectedAgent = null;
          selectedEdgeKey = null;
          event.stopPropagation();
          render();
        });
      });
    }

    function edgeEndpoint(source, target, nodeW, nodeH, isStart) {
      const from = isStart ? source : target;
      const to = isStart ? target : source;
      const dx = to.x - from.x;
      const dy = to.y - from.y;
      if (Math.abs(dx) > Math.abs(dy)) {
        return {
          x: from.x + Math.sign(dx || 1) * nodeW / 2,
          y: from.y + (dy / Math.max(Math.abs(dx), 1)) * nodeH * 0.18,
        };
      }
      return {
        x: from.x + (dx / Math.max(Math.abs(dy), 1)) * nodeW * 0.18,
        y: from.y + Math.sign(dy || 1) * nodeH / 2,
      };
    }

    function stableHash(text) {
      let hash = 0;
      for (const ch of String(text || '')) {
        hash = ((hash << 5) - hash + ch.charCodeAt(0)) | 0;
      }
      return Math.abs(hash);
    }

    function curvedRoute(start, end, edge, index, radial, centerX, centerY, hasReverse) {
      const dx = end.x - start.x;
      const dy = end.y - start.y;
      const distance = Math.max(Math.hypot(dx, dy), 1);
      const midX = (start.x + end.x) / 2;
      const midY = (start.y + end.y) / 2;
      const perpX = -dy / distance;
      const perpY = dx / distance;
      const direction = edge.sender < edge.recipient ? 1 : -1;
      const jitter = ((stableHash(edge.key) % 5) - 2) * (radial ? 11 : 8);
      let curve = Math.min(radial ? 230 : 130, Math.max(radial ? 78 : 42, distance * (radial ? 0.24 : 0.17)));
      if (!hasReverse) curve *= 0.62;
      curve = curve * direction + jitter;

      let controlX = midX + perpX * curve;
      let controlY = midY + perpY * curve;
      if (radial) {
        const outX = midX - centerX;
        const outY = midY - centerY;
        const outDistance = Math.max(Math.hypot(outX, outY), 1);
        const outward = Math.min(130, Math.max(36, distance * 0.16));
        controlX += (outX / outDistance) * outward;
        controlY += (outY / outDistance) * outward;
      } else {
        controlX += ((index % 3) - 1) * 18;
      }

      return {
        path: `M ${start.x.toFixed(1)} ${start.y.toFixed(1)} Q ${controlX.toFixed(1)} ${controlY.toFixed(1)} ${end.x.toFixed(1)} ${end.y.toFixed(1)}`,
        labelX: (midX * 0.65 + controlX * 0.35),
        labelY: (midY * 0.65 + controlY * 0.35) - 4,
      };
    }

    function renderActivityBubbles(currentAgents, positions, nodeW, nodeH, layoutWidth, layoutHeight) {
      const grouped = new Map();
      function addGrouped(agentId, item) {
        if (!agentId || !positions.has(agentId)) return;
        if (!grouped.has(agentId)) grouped.set(agentId, []);
        grouped.get(agentId).push(item);
      }
      activeWindowEvents(1).forEach(event => {
        if (isWorkflowEvent(event)) return;
        if (event.event.startsWith('tool_call_') || event.event === 'chemgraph_job_result') return;
        const bubble = bubbleInfo(event);
        if (!bubble) return;
        addGrouped(event.agent_id, {
          event,
          bubble,
          selected: eventKey(event) === selectedActivityEventKey,
          sortTime: eventTimestamp(event) ?? 0,
        });
      });
      currentAgents.forEach(agent => {
        reasoningTurnsForAgent(agent.agent_id).slice(-3).forEach(turn => {
          const representative = turn.representative;
          if (!representative) return;
          const selected = turn.events.some(event => eventKey(event) === selectedActivityEventKey);
          addGrouped(agent.agent_id, {
            event: representative,
            bubble: chemgraphTurnBubbleInfo(turn),
            selected,
            sortTime: turn.lastTimestamp ?? turn.firstTimestamp ?? 0,
          });
        });
      });
      const rows = [];
      grouped.forEach((items, agentId) => {
        const pos = positions.get(agentId);
        items
          .slice()
          .sort((a, b) => (a.sortTime || 0) - (b.sortTime || 0))
          .slice(-4)
          .forEach((item, index) => {
          const key = eventKey(item.event);
          const selected = item.selected || key === selectedActivityEventKey;
          const width = Math.max(64, Math.min(156, item.bubble.label.length * 7 + 18));
          let x = pos.x + nodeW / 2 + 8;
          if (x + width > layoutWidth - 10) x = pos.x - nodeW / 2 - width - 8;
          const y = Math.max(12, Math.min(layoutHeight - 24, pos.y - nodeH / 2 + index * 24));
          rows.push(`
            <g class="activity-bubble ${item.bubble.className} ${selected ? 'selected' : ''}" data-activity-event-key="${esc(key)}" transform="translate(${x.toFixed(1)} ${y.toFixed(1)})">
              <rect width="${width}" height="20" rx="10"></rect>
              <text x="${(width / 2).toFixed(1)}" y="14" text-anchor="middle">${esc(item.bubble.label)}</text>
              <title>${esc(item.bubble.title)}</title>
            </g>
          `);
        });
      });
      return rows.join('');
    }

    function chemgraphTurnBubbleInfo(turn) {
      const round = turn.round === null || turn.round === undefined ? '-' : turn.round;
      const toolCount = (turn.scienceToolCount || 0) + (turn.actionToolCount || 0);
      const status = turn.status || 'running';
      const failed = String(status).toLowerCase() === 'failed';
      const label = `CG r${round} ${toolCount}t`;
      const title = [
        `Open ChemGraph turn for ${workflowAgentId(turn.representative) || '-'}`,
        `Round: ${round}`,
        `Status: ${status}`,
        `LM calls: ${turn.lmCount || 0}`,
        `Science tools: ${turn.scienceToolCount || 0}`,
        `Message actions: ${turn.actionToolCount || 0}`,
        `Span: ${turn.spanId || '-'}`,
      ].join('\n');
      return {
        className: failed ? 'bubble-error' : 'bubble-chemgraph',
        label,
        title,
      };
    }

    function toolDisplayName(event) {
      const p = event.payload || {};
      return (
        p.tool_name
        || p.tool
        || p.name
        || p.result?.tool_name
        || p.result?.name
        || p.tool_result?.tool_name
        || 'tool'
      );
    }

    function bubbleInfo(event) {
      const p = event.payload || {};
      if (event.event === 'agent_decision') {
        const actions = Array.isArray(p.actions) ? p.actions.length : 0;
        return {
          className: 'bubble-decision',
          label: actions ? `decide ${actions}` : 'decide',
          title: `${event.agent_id} decision\n${trunc(p.rationale || p.wake_reason || '', 220)}`,
        };
      }
      if (event.event === 'belief_updated') {
        return {
          className: 'bubble-belief',
          label: 'belief',
          title: `${event.agent_id} belief\n${formatBelief(p)}`,
        };
      }
      if (event.event === 'agent_error') {
        return {
          className: 'bubble-error',
          label: 'error',
          title: `${event.agent_id} error\n${p.error || formatJson(p)}`,
        };
      }
      return null;
    }

    function ensureGraphView(width, height) {
      const padX = width >= 1180 ? Math.max(420, width * 0.28) : 140;
      const padY = width >= 1180 ? Math.max(220, height * 0.22) : 120;
      const bounds = {
        x: -padX,
        y: -padY,
        width: width + padX * 2,
        height: height + padY * 2,
      };
      if (!graphView) {
        graphView = {
          x: bounds.x,
          y: bounds.y,
          width: bounds.width,
          height: bounds.height,
          layoutWidth: width,
          layoutHeight: height,
          boundsX: bounds.x,
          boundsY: bounds.y,
          boundsWidth: bounds.width,
          boundsHeight: bounds.height,
        };
        return;
      }
      if (
        graphView.layoutWidth !== width
        || graphView.layoutHeight !== height
      ) {
        const nextView = preserveViewForLayoutChange(
          graphView,
          bounds,
          width,
          height,
        );
        graphView = {
          ...graphView,
          ...nextView,
          layoutWidth: width,
          layoutHeight: height,
          boundsX: bounds.x,
          boundsY: bounds.y,
          boundsWidth: bounds.width,
          boundsHeight: bounds.height,
        };
        clampGraphView();
      }
    }

    function preserveViewForLayoutChange(view, nextBounds, nextLayoutWidth, nextLayoutHeight) {
      const previousBoundsWidth = view.boundsWidth || view.layoutWidth || nextLayoutWidth;
      const previousBoundsHeight = view.boundsHeight || view.layoutHeight || nextLayoutHeight;
      const zoomX = previousBoundsWidth / Math.max(view.width || previousBoundsWidth, 1);
      const zoomY = previousBoundsHeight / Math.max(view.height || previousBoundsHeight, 1);
      const centerXRatio = (
        (view.x || 0) + (view.width || previousBoundsWidth) / 2 - (view.boundsX || 0)
      ) / Math.max(previousBoundsWidth, 1);
      const centerYRatio = (
        (view.y || 0) + (view.height || previousBoundsHeight) / 2 - (view.boundsY || 0)
      ) / Math.max(previousBoundsHeight, 1);
      const width = nextBounds.width / Math.max(zoomX, 1e-6);
      const height = nextBounds.height / Math.max(zoomY, 1e-6);
      const centerX = nextBounds.x + centerXRatio * nextBounds.width;
      const centerY = nextBounds.y + centerYRatio * nextBounds.height;
      return {
        x: centerX - width / 2,
        y: centerY - height / 2,
        width,
        height,
      };
    }

    function updateGraphViewBox() {
      const svg = document.getElementById('graph');
      if (!graphView) return;
      svg.setAttribute(
        'viewBox',
        `${graphView.x.toFixed(1)} ${graphView.y.toFixed(1)} ${graphView.width.toFixed(1)} ${graphView.height.toFixed(1)}`
      );
    }

    function clampGraphView() {
      if (!graphView) return;
      const boundsX = graphView.boundsX ?? 0;
      const boundsY = graphView.boundsY ?? 0;
      const boundsWidth = graphView.boundsWidth ?? graphView.layoutWidth;
      const boundsHeight = graphView.boundsHeight ?? graphView.layoutHeight;
      graphView.width = Math.min(boundsWidth, Math.max(graphView.layoutWidth / 10, graphView.width));
      graphView.height = Math.min(boundsHeight, Math.max(graphView.layoutHeight / 10, graphView.height));
      graphView.x = Math.min(Math.max(boundsX, graphView.x), boundsX + boundsWidth - graphView.width);
      graphView.y = Math.min(Math.max(boundsY, graphView.y), boundsY + boundsHeight - graphView.height);
    }

    function zoomGraph(factor) {
      if (!graphView) return;
      const centerX = graphView.x + graphView.width / 2;
      const centerY = graphView.y + graphView.height / 2;
      graphView.width *= factor;
      graphView.height *= factor;
      graphView.x = centerX - graphView.width / 2;
      graphView.y = centerY - graphView.height / 2;
      clampGraphView();
      updateGraphViewBox();
    }

    function resetGraphView() {
      graphView = null;
      renderGraph();
    }

    function renderAgentPicker() {
      const picker = document.getElementById('agentSelect');
      if (isWorkflowMode()) {
        picker.innerHTML = '<option value="">Workflow events</option>';
        picker.value = '';
        picker.disabled = true;
        return;
      }
      picker.disabled = false;
      const options = ['<option value="">Select agent</option>'].concat(
        agents().map(agent => `<option value="${esc(agent.agent_id)}">${esc(agent.agent_id)}</option>`)
      ).join('');
      picker.innerHTML = options;
      picker.value = selectedAgent || '';
    }

    function selectedState() {
      return agents().find(agent => agent.agent_id === selectedAgent) || null;
    }

    function selectedActivityEvent() {
      if (!selectedActivityEventKey) return null;
      return allEvents().find(event => eventKey(event) === selectedActivityEventKey) || null;
    }

    function currentDetailIdentity() {
      if (selectedActivityEventKey) return `activity:${selectedActivityEventKey}`;
      if (selectedEdgeKey) return `edge:${selectedEdgeKey}`;
      if (selectedAgent) return `agent:${selectedAgent}`;
      if (isWorkflowMode()) {
        const event = currentEvent();
        return event ? `workflow-event:${eventKey(event)}` : 'workflow-empty';
      }
      const event = currentEvent();
      return event ? `timeline-event:${eventKey(event)}` : 'empty';
    }

    function captureDetailScrollSnapshot() {
      const blockIds = ['detailPrimary', 'detailSecondary', 'detailTertiary'];
      const blocks = {};
      blockIds.forEach(id => {
        const el = document.getElementById(id);
        if (!el) return;
        blocks[id] = {
          scrollTop: el.scrollTop,
          scrollLeft: el.scrollLeft,
        };
      });
      return {
        identity: lastRenderedDetailIdentity,
        blocks,
      };
    }

    function restoreDetailScrollSnapshot(snapshot) {
      if (!snapshot || snapshot.identity !== currentDetailIdentity()) return;
      Object.entries(snapshot.blocks || {}).forEach(([id, pos]) => {
        const el = document.getElementById(id);
        if (!el) return;
        el.scrollTop = pos.scrollTop || 0;
        el.scrollLeft = pos.scrollLeft || 0;
      });
    }

    function renderDetail() {
      const activityEvent = selectedActivityEvent();
      if (activityEvent) {
        renderTimelineEventDetail(
          activityEvent,
          allEvents().findIndex(event => eventKey(event) === selectedActivityEventKey),
        );
        return;
      }
      if (isWorkflowMode()) {
        const event = currentEvent();
        if (event) {
          renderTimelineEventDetail(event);
        } else {
          renderEmptyDetail();
        }
        return;
      }
      if (selectedEdgeKey) {
        renderEdgeDetail(selectedEdgeKey);
        return;
      }
      const agent = selectedState();
      if (!agent) {
        renderEmptyDetail();
        return;
      }
      selectedAgent = agent.agent_id;
      document.getElementById('agentSelect').value = selectedAgent;
      document.getElementById('detailTitle').textContent = agent.agent_id;
      // Look up the campaign-config entry for this agent (if loaded).
      // Static fields (mission, allowed_peers, mcp_servers, allowed_tools)
      // live there rather than in the runtime status -- surfacing them
      // is the first read-only slice of authoring-mode. Step 6 turns
      // these into editable inputs.
      const campaignAgentSpec = (window.__campaignDoc?.agents || [])
        .find(a => a?.name === agent.agent_id) || null;
      const cards = [
        ['Role', agent.role || campaignAgentSpec?.role || '-'],
        ['Host', agentHost(agent)],
        ['Decisions', agent.decision_count || 0],
        ['Received / Sent', `${agent.received_message_count || 0} / ${agent.outbox_count || 0}`],
        ['Tools', `${agent.tool_finished_count || 0} / ${agent.tool_started_count || 0}`],
        ['State', agent.last_error ? 'error' : agent.started ? 'active' : 'pending'],
      ];
      if (campaignAgentSpec) {
        const peers = campaignAgentSpec.allowed_peers || [];
        const mcps = campaignAgentSpec.mcp_servers || [];
        const tools = campaignAgentSpec.allowed_tools || [];
        cards.push(['Allowed peers', peers.length ? peers.join(', ') : '(any)']);
        cards.push(['MCP servers', mcps.length ? mcps.join(', ') : '(none)']);
        if (tools.length) cards.push(['Allowed tools', tools.join(', ')]);
      }
      document.getElementById('detailCards').innerHTML = detailCards(cards);
      // Mission field: editable textarea + Save button. First slice
      // of authoring-mode; expanded per slice as A2 feedback lands.
      // Save is disabled when a launcher subprocess is running --
      // mid-run edits would race with rank-0 reading the campaign
      // config at spawn time.
      if (campaignAgentSpec) {
        const missionHost = document.createElement('div');
        missionHost.style.cssText = 'margin-top:14px;';
        const initialText = campaignAgentSpec.mission || '';
        missionHost.innerHTML = `
          <div style="display:flex; justify-content:space-between; align-items:baseline; margin-bottom:4px;">
            <strong style="font-size:12px; color:var(--muted); letter-spacing:.05em; text-transform:uppercase;">Mission</strong>
            <div>
              <span id="missionStatus-${esc(agent.agent_id)}" style="font-size:11px; color:var(--muted); margin-right:6px;"></span>
              <button id="missionSave-${esc(agent.agent_id)}" style="font-size:11px; padding:2px 8px;">Save</button>
            </div>
          </div>
          <textarea id="missionInput-${esc(agent.agent_id)}"
                    style="width:100%; min-height:140px; font-family:ui-monospace,SFMono-Regular,Menlo,monospace; font-size:12px; padding:8px; box-sizing:border-box; border:1px solid var(--line); border-radius:4px; resize:vertical; background:#fbfcfd;"
          >${esc(initialText)}</textarea>
        `;
        document.getElementById('detailCards').appendChild(missionHost);
        const btn = missionHost.querySelector('button');
        const input = missionHost.querySelector('textarea');
        const status = missionHost.querySelector('span');
        // Reflect launcher state (disable during runs)
        const launcherBusy = !!window.__launcherRunning;
        btn.disabled = launcherBusy;
        if (launcherBusy) status.textContent = 'launcher running';
        btn.addEventListener('click', async () => {
          btn.disabled = true;
          status.textContent = 'saving...';
          try {
            const r = await fetch('/api/campaign', {
              method: 'POST',
              headers: {'Content-Type': 'application/json'},
              body: JSON.stringify({
                action: 'edit_agent_field',
                params: {
                  agent: agent.agent_id,
                  field: 'mission',
                  value: input.value,
                },
              }),
            });
            const data = await r.json();
            if (!data.ok) {
              status.textContent = `error: ${data.error || 'unknown'}`;
              return;
            }
            // Update the client-side cache so the panel keeps showing
            // the edited value on the next re-render.
            const cd = window.__campaignDoc;
            if (cd && Array.isArray(cd.agents)) {
              const entry = cd.agents.find(a => a?.name === agent.agent_id);
              if (entry) entry.mission = input.value;
            }
            const failed = (data.rsync || []).filter(x => !x.ok);
            status.textContent = failed.length
              ? `saved local, rsync failed: ${failed.map(x => x.site).join(', ')}`
              : `saved (rsync: ${(data.rsync || []).map(x => x.site).join(', ') || 'no sites'})`;
          } catch (e) {
            status.textContent = `error: ${String(e)}`;
          } finally {
            btn.disabled = false;
          }
        });
      }
      const current = currentEvent();
      if (current?.event === 'agent_decision' && current.agent_id === agent.agent_id) {
        setDetailBlock('detailPrimaryTitle', 'Current Decision', 'detailPrimary', formatDecisionEvent(current));
        setDetailBlock('detailSecondaryTitle', 'Wake Context', 'detailSecondary', formatWakeEvents(current));
        const turnEvents = workflowEventsForSelection(current);
        if (turnEvents.length) {
          setDetailHtmlBlock(
            'detailTertiaryTitle',
            'ChemGraph Turn',
            'detailTertiary',
            detailRich(detailSection(
              'ChemGraph Panel',
              paragraphsHtml('Open in the floating ChemGraph panel. Click inner graph nodes there to inspect LM, tool, action, and output details inside the panel.'),
              'info',
            )),
          );
        } else {
          const received = eventsOf('message_received').filter(e => e.agent_id === agent.agent_id).slice(-4);
          setDetailBlock('detailTertiaryTitle', 'Recent Received Messages', 'detailTertiary', received.length
            ? received.map(formatMessageEvent).join('\n\n')
            : 'No received messages at this point in the timeline.');
        }
        return;
      }
      const beliefEvents = eventsOf('belief_updated').filter(e => e.agent_id === agent.agent_id);
      const latestBelief = beliefEvents.length ? beliefEvents[beliefEvents.length - 1].payload : null;
      setDetailBlock('detailPrimaryTitle', 'Current Belief', 'detailPrimary', latestBelief
        ? formatBelief(latestBelief)
        : 'No belief recorded at this point in the timeline.');
      const received = eventsOf('message_received').filter(e => e.agent_id === agent.agent_id).slice(-6);
      setDetailHtmlBlock('detailSecondaryTitle', 'Received Messages', 'detailSecondary', received.length
        ? messageHistoryHtml(received)
        : '<div class="detail-rich"><div class="muted">No received messages at this point in the timeline.</div></div>');
      const turns = reasoningTurnsForAgent(agent.agent_id);
      setDetailHtmlBlock(
        'detailTertiaryTitle',
        'ChemGraph Turn Entries',
        'detailTertiary',
        reasoningTurnListHtml(turns) + (turns.length
          ? detailRich(detailSection(
              'Open Turn',
              paragraphsHtml('Click a ChemGraph turn chip attached to this agent in the graph, or click a turn row above. The inner ChemGraph graph opens in the floating panel.'),
              'info',
            ))
          : '<div class="muted">No ChemGraph turns for this agent at this point in the timeline.</div>'),
      );
    }

    function renderEdgeDetail(edgeKey) {
      const [sender, recipient] = edgeKey.split('->');
      const currentAgents = agents();
      const senderAgent = currentAgents.find(agent => agent.agent_id === sender);
      const recipientAgent = currentAgents.find(agent => agent.agent_id === recipient);
      // "Group" = site in federated mode, host in single-machine.
      // Route label becomes "cross-site" or "cross-node" accordingly,
      // which is what the operator actually wants to see at a glance.
      const senderGroup = agentGroup(senderAgent);
      const recipientGroup = agentGroup(recipientAgent);
      const groupLabel = snapshot?.federated ? 'site' : 'host';
      const messages = eventsOf('message_sent').filter(e => {
        const p = e.payload || {};
        return p.sender === sender && p.recipient === recipient;
      });
      const latest = messages.length ? messages[messages.length - 1] : null;
      const latestPayload = latest?.payload || {};
      const crossGroup = senderGroup && recipientGroup && senderGroup !== recipientGroup;
      const route = crossGroup
        ? (snapshot?.federated ? 'cross-site' : 'cross-node')
        : 'same-' + groupLabel;
      document.getElementById('detailTitle').textContent = `${sender} -> ${recipient}`;
      document.getElementById('detailCards').innerHTML = detailCards([
        ['Route', route],
        ['Messages', messages.length],
        [`From ${groupLabel}`, senderGroup],
        [`To ${groupLabel}`, recipientGroup],
      ]);
      setDetailHtmlBlock('detailPrimaryTitle', 'Latest Message', 'detailPrimary', latest
        ? messageDetailHtml(latest)
        : '<div class="detail-rich"><div class="muted">No message visible at this point in the timeline.</div></div>');
      const history = messages.slice(-8);
      setDetailHtmlBlock('detailSecondaryTitle', 'Message History', 'detailSecondary', history.length
        ? messageHistoryHtml(history)
        : '<div class="detail-rich"><div class="muted">No messages visible at this point in the timeline.</div></div>');
      const messageIds = new Set(messages.map(e => e.payload?.message_id).filter(Boolean));
      const relatedBeliefs = eventsOf('belief_updated').filter(e => {
        const refs = e.payload?.supporting_message_ids || [];
        return refs.some(ref => messageIds.has(ref));
      });
      setDetailBlock('detailTertiaryTitle', 'Beliefs Citing This Edge', 'detailTertiary', relatedBeliefs.length
        ? relatedBeliefs.slice(-6).map(e => `${formatTime(e.timestamp)} ${e.agent_id}\n${formatBelief(e.payload)}`).join('\n\n')
        : 'No belief cites this relationship at this point in the timeline.');
    }

    function renderEmptyDetail() {
      const event = currentEvent();
      if (event) {
        renderTimelineEventDetail(event);
        return;
      }
      document.getElementById('detailTitle').textContent = 'Timeline Event';
      document.getElementById('detailCards').innerHTML = '';
      setDetailBlock('detailPrimaryTitle', 'State', 'detailPrimary', 'No events yet.');
      setDetailBlock('detailSecondaryTitle', 'Evidence', 'detailSecondary', '');
      setDetailBlock('detailTertiaryTitle', 'History', 'detailTertiary', '');
    }

    function renderTimelineEventDetail(event, indexOverride = null) {
      const index = indexOverride ?? currentEventIndex();
      const isToolEvent = ['tool_call_started', 'tool_call_finished', 'tool_call_failed', 'chemgraph_job_result'].includes(event.event);
      const isNestedWorkflowEvent = isWorkflowEvent(event);
      document.getElementById('detailTitle').textContent = isNestedWorkflowEvent
        ? chemgraphNodeDetailTitle(event)
        : isToolEvent
          ? `Tool: ${toolDisplayName(event)}`
          : `Timeline Event ${index + 1}`;
      const cards = [
        ['Event', event.event],
        ['Time', formatTime(event.timestamp)],
        ['Agent', event.agent_id || '-'],
        ['Role', event.role || '-'],
      ];
      if (isToolEvent) cards.push(['Tool', toolDisplayName(event)]);
      if (isNestedWorkflowEvent) cards.push(['Runtime', event.payload?.runtime || '-']);
      document.getElementById('detailCards').innerHTML = detailCards(cards);
      if (isNestedWorkflowEvent) {
        setDetailHtmlBlock('detailPrimaryTitle', chemgraphNodeDetailTitle(event), 'detailPrimary', chemgraphNodeDetailHtml(event));
        setDetailHtmlBlock('detailSecondaryTitle', 'Node Payload', 'detailSecondary', payloadDetailHtml(event.payload || {}));
        setDetailHtmlBlock('detailTertiaryTitle', 'ChemGraph Context', 'detailTertiary', chemgraphNodeContextHtml(event));
        return;
      }
      if (event.event === 'agent_decision') {
        setDetailBlock('detailPrimaryTitle', 'Agent Decision', 'detailPrimary', formatDecisionEvent(event));
        setDetailBlock('detailSecondaryTitle', 'Wake Context', 'detailSecondary', formatWakeEvents(event));
        const turnEvents = workflowEventsForSelection(event);
        if (turnEvents.length) {
          setDetailHtmlBlock('detailTertiaryTitle', 'ChemGraph Turn', 'detailTertiary', detailRich(detailSection(
            'ChemGraph Panel',
            paragraphsHtml('The ChemGraph turn for this decision is shown in the floating panel. Click an inner node to inspect it inside the panel.'),
            'info',
          )));
        } else {
          setDetailBlock('detailTertiaryTitle', 'Raw Action Count', 'detailTertiary', `${event.payload?.actions?.length || 0} action(s) returned by LM.`);
        }
        return;
      }
      if (event.event === 'message_sent' || event.event === 'message_received') {
        setDetailHtmlBlock('detailPrimaryTitle', 'Message', 'detailPrimary', messageDetailHtml(event));
        setDetailHtmlBlock('detailSecondaryTitle', 'Route', 'detailSecondary', routeDetailHtml(event));
        setDetailHtmlBlock('detailTertiaryTitle', 'Payload', 'detailTertiary', payloadDetailHtml(event.payload || {}));
        return;
      }
      if (event.event === 'belief_updated') {
        setDetailBlock('detailPrimaryTitle', 'Belief Update', 'detailPrimary', formatBelief(event.payload || {}));
        setDetailBlock('detailSecondaryTitle', 'Supporting Messages', 'detailSecondary', (event.payload?.supporting_message_ids || []).join('\n') || 'No message refs.');
        setDetailBlock('detailTertiaryTitle', 'Supporting Artifacts', 'detailTertiary', (event.payload?.supporting_artifact_ids || []).join('\n') || 'No artifact refs.');
        return;
      }
      if (['tool_call_started', 'tool_call_finished', 'tool_call_failed', 'chemgraph_job_result'].includes(event.event)) {
        setDetailHtmlBlock('detailPrimaryTitle', 'Tool Event', 'detailPrimary', toolDetailHtml(event));
        setDetailHtmlBlock('detailSecondaryTitle', 'Tool Payload', 'detailSecondary', payloadDetailHtml(event.payload || {}));
        const nested = nestedWorkflowEventsForTool(event);
        setDetailHtmlBlock('detailTertiaryTitle', nested.length ? 'ChemGraph Tool Trace' : 'Correlation', 'detailTertiary', nested.length
          ? workflowHistoryHtml(nested)
          : payloadDetailHtml({correlation_id: event.correlation_id || 'No correlation id.'}));
        return;
      }
      setDetailHtmlBlock('detailPrimaryTitle', 'Event Payload', 'detailPrimary', payloadDetailHtml(event.payload || {}));
      setDetailBlock('detailSecondaryTitle', 'Correlation', 'detailSecondary', event.correlation_id || 'No correlation id.');
      setDetailBlock('detailTertiaryTitle', 'Selection', 'detailTertiary', 'Click a node or edge to inspect derived agent or communication state.');
    }

    function detailCards(items) {
      return items.map(([label, value]) => `
        <div class="detail-card">
          <div class="label">${esc(label)}</div>
          <div class="value">${esc(value)}</div>
        </div>
      `).join('');
    }

    function setDetailBlock(titleId, title, bodyId, body) {
      document.getElementById(titleId).textContent = title;
      const el = document.getElementById(bodyId);
      const text = body || '';
      if (el.textContent !== text) {
        el.textContent = text;
        renderedHtmlCache.delete(el);
      }
    }

    function setDetailHtmlBlock(titleId, title, bodyId, bodyHtml) {
      document.getElementById(titleId).textContent = title;
      setStableHtml(document.getElementById(bodyId), bodyHtml || '', true);
    }

    function setStableHtml(el, html, preserveScroll = true) {
      if (!el) return;
      const next = html || '';
      if (renderedHtmlCache.get(el) === next) return;
      const scrollTop = el.scrollTop;
      const scrollLeft = el.scrollLeft;
      el.innerHTML = next;
      renderedHtmlCache.set(el, next);
      if (preserveScroll) {
        el.scrollTop = scrollTop;
        el.scrollLeft = scrollLeft;
      }
    }

    function selectActivityEventKey(key) {
      if (!key) return;
      selectedActivityEventKey = key;
      selectedAgent = null;
      selectedEdgeKey = null;
      const index = allEvents().findIndex(event => eventKey(event) === key);
      if (index >= 0) {
        followLatest = false;
        timelineIndex = index;
        const event = allEvents()[index];
        if (workflowEventsForSelection(event).length) {
          workflowPanelOpen = true;
        }
      }
      render();
    }

    function handleDetailPaneClick(event) {
      const target = event.target.closest('[data-detail-activity-key]');
      if (!target) return;
      event.preventDefault();
      event.stopPropagation();
      selectActivityEventKey(target.dataset.detailActivityKey);
    }

    function detailRich(...parts) {
      return `<div class="detail-rich">${parts.filter(Boolean).join('')}</div>`;
    }

    function detailSection(title, body, tone = '') {
      return `
        <div class="detail-section ${esc(tone)}">
          <div class="detail-section-title">${esc(title)}</div>
          ${body || '<div class="muted">None</div>'}
        </div>
      `;
    }

    function detailKvGrid(rows) {
      const visibleRows = rows.filter(([_, value]) => !isEmptyDetailValue(value));
      if (!visibleRows.length) return '<div class="muted">None</div>';
      return `
        <div class="detail-kv">
          ${visibleRows.map(([label, value, kind]) => `
            <div class="detail-key">${esc(label)}</div>
            <div class="detail-value ${kind === 'mono' ? 'mono' : ''}">${detailValueHtml(value, kind)}</div>
          `).join('')}
        </div>
      `;
    }

    function isEmptyDetailValue(value) {
      return value === undefined
        || value === null
        || value === ''
        || (Array.isArray(value) && value.length === 0);
    }

    function detailValueHtml(value, kind = '') {
      if (value === undefined || value === null || value === '') return '<span class="muted">-</span>';
      if (Array.isArray(value)) {
        if (!value.length) return '<span class="muted">none</span>';
        if (value.every(item => ['string', 'number', 'boolean'].includes(typeof item))) {
          return detailChips(value, kind);
        }
        return collapsedJsonHtml(`Array (${value.length})`, value);
      }
      if (typeof value === 'object') {
        return collapsedJsonHtml(`Object (${Object.keys(value).length})`, value);
      }
      const text = String(value);
      if (kind === 'text') return paragraphsHtml(text);
      return esc(text);
    }

    function detailChips(values, kind = '') {
      const list = Array.isArray(values) ? values : [values];
      if (!list.length) return '<span class="muted">none</span>';
      return `
        <div class="detail-chip-row">
          ${list.map(value => `<span class="detail-chip ${esc(kind)}">${esc(String(value))}</span>`).join('')}
        </div>
      `;
    }

    function paragraphsHtml(text) {
      const value = String(text || '').trim();
      if (!value) return '<div class="muted">None</div>';
      const paragraphs = value.split(/\n{2,}/).map(part => part.trim()).filter(Boolean);
      return `<div class="detail-text">${paragraphs.map(part => `<p>${esc(part)}</p>`).join('')}</div>`;
    }

    function collapsedJsonHtml(summary, value) {
      return `
        <details class="detail-raw">
          <summary>${esc(summary)}</summary>
          <pre>${esc(formatJson(value))}</pre>
        </details>
      `;
    }

    function rawJsonDetails(value) {
      return `
        <details class="detail-raw">
          <summary>Raw JSON</summary>
          <pre>${esc(formatJson(value))}</pre>
        </details>
      `;
    }

    function statusTone(status) {
      const value = String(status || '').toLowerCase();
      if (['ok', 'success', 'completed', 'finished'].includes(value)) return 'ok';
      if (['failed', 'failure', 'error'].includes(value)) return 'error';
      if (['running', 'pending', 'submitted'].includes(value)) return 'warn';
      return 'info';
    }

    function messageDetailHtml(event, {includeRaw = true} = {}) {
      const p = event.payload || {};
      const sender = p.sender || event.agent_id || '-';
      const recipient = p.recipient || '-';
      const refs = [
        ...(Array.isArray(p.evidence_refs) ? p.evidence_refs : []),
        ...(Array.isArray(p.tool_result_ids) ? p.tool_result_ids : []),
        ...(Array.isArray(p.supporting_message_ids) ? p.supporting_message_ids : []),
      ];
      return detailRich(
        detailSection('Route', detailKvGrid([
          ['Direction', `${sender} -> ${recipient}`],
          ['Time', formatTime(event.timestamp)],
          ['Message id', p.message_id || '-', 'mono'],
          ['Event', event.event],
        ]), 'info'),
        p.tldr ? detailSection('TLDR', paragraphsHtml(p.tldr), 'info') : '',
        p.content ? detailSection('Content', paragraphsHtml(p.content)) : '',
        p.reason ? detailSection('Reason', paragraphsHtml(p.reason), 'warn') : '',
        refs.length ? detailSection('References', detailChips(refs, 'action')) : '',
        includeRaw ? rawJsonDetails(p) : '',
      );
    }

    function messageHistoryHtml(messages) {
      if (!messages.length) return detailRich('<div class="muted">No messages visible.</div>');
      return detailRich(messages.map(event => {
        const p = event.payload || {};
        const title = `${p.sender || event.agent_id || '-'} -> ${p.recipient || '-'}`;
        // Always show content when present. If the tldr differs from
        // the content (as it does for operator injects, where tldr =
        // "operator-injected" and content is the real payload), show
        // tldr as a small header AND the content underneath. Previously
        // the tldr suppressed the content field entirely, which hid
        // what the operator actually sent.
        const contentText = trunc(p.content || '', 360);
        const tldrText = p.tldr || '';
        const showBoth = tldrText && contentText && tldrText !== contentText;
        const body = [
          detailKvGrid([
            ['Time', formatTime(event.timestamp)],
            ['Message id', p.message_id || '-', 'mono'],
          ]),
          showBoth
            ? `<div class="muted" style="font-size:11px;">${esc(tldrText)}</div>${paragraphsHtml(contentText)}`
            : (contentText
                ? paragraphsHtml(contentText)
                : paragraphsHtml(tldrText || '(no content)')),
        ].join('');
        return detailSection(title, body, '');
      }).join(''));
    }

    function routeDetailHtml(event) {
      const p = event.payload || {};
      const currentAgents = agents();
      const sender = currentAgents.find(agent => agent.agent_id === p.sender);
      const recipient = currentAgents.find(agent => agent.agent_id === p.recipient);
      const senderGroup = agentGroup(sender);
      const recipientGroup = agentGroup(recipient);
      const groupLabel = snapshot?.federated ? 'site' : 'host';
      const crossGroup = senderGroup && recipientGroup && senderGroup !== recipientGroup;
      const route = crossGroup
        ? (snapshot?.federated ? 'cross-site' : 'cross-node')
        : 'same-' + groupLabel;
      return detailRich(
        detailSection('Route', detailKvGrid([
          ['Type', route],
          [`Sender ${groupLabel}`, senderGroup],
          [`Recipient ${groupLabel}`, recipientGroup],
          ['Message id', p.message_id || '-', 'mono'],
        ]), crossGroup ? 'ok' : 'info'),
      );
    }

    function issueListHtml(issues) {
      if (!Array.isArray(issues) || !issues.length) {
        return '<div class="muted">No issues recorded.</div>';
      }
      return issues.map((issue, index) => detailSection(
        `${index + 1}. ${issue.field || 'field'}`,
        detailKvGrid([
          ['Expected', issue.expected || '-'],
          ['Received type', issue.received_type || '-'],
          ['Received', issue.received ?? '', 'mono'],
          ['Defaulted to', issue.defaulted_to ?? '', 'mono'],
          ['Normalized to', issue.normalized_to ?? '', 'mono'],
          ['Dropped items', issue.dropped_items ?? ''],
          ['Allowed peers', issue.allowed_peers || [], 'action'],
        ]),
      )).join('');
    }

    function workflowDetailHtml(event) {
      const p = event.payload || {};
      const calls = Array.isArray(p.tool_calls)
        ? p.tool_calls.map(call => call.name || call.id || 'tool').filter(Boolean)
        : [];
      return detailRich(
        detailSection('Workflow', detailKvGrid([
          ['Status', p.status || event.event],
          ['Runtime', p.runtime || '-'],
          ['Workflow', p.workflow_type || '-'],
          ['Node', p.workflow_node || '-'],
          ['Round', p.round ?? '-'],
          ['Thread', p.thread_id || '-', 'mono'],
          ['Time', formatTime(event.timestamp)],
        ]), statusTone(p.status)),
        detailSection('Span', detailKvGrid([
          ['Span id', p.span_id || '-', 'mono'],
          ['Parent span', p.parent_span_id || '-', 'mono'],
          ['Correlation', event.correlation_id || '-', 'mono'],
          ['Log dir', p.log_dir || '-', 'mono'],
        ])),
        calls.length ? detailSection('Tool Calls', detailChips(calls, 'science')) : '',
        p.content_preview ? detailSection('Preview', paragraphsHtml(p.content_preview)) : '',
        p.error ? detailSection('Error', paragraphsHtml(p.error), 'error') : '',
      );
    }

    function chemgraphNodeDetailTitle(event) {
      if (event.event === 'llm_decision') return 'LM Decision Node';
      if (event.event.startsWith('tool_call_')) return `Tool Node: ${toolDisplayName(event)}`;
      if (event.event === 'workflow_started' || event.event === 'run_started') return 'Wake Input Node';
      if (event.event === 'workflow_finished' || event.event === 'workflow_output' || event.event === 'run_finished') return 'Output Node';
      return `ChemGraph Node: ${event.event}`;
    }

    function chemgraphNodeDetailHtml(event) {
      const p = event.payload || {};
      if (event.event === 'llm_decision') {
        const calls = Array.isArray(p.tool_calls) ? p.tool_calls : [];
        return detailRich(
          detailSection('LM Decision', detailKvGrid([
            ['Agent', event.agent_id || p.agent_id || '-'],
            ['Role', event.role || p.role || '-'],
            ['Time', formatTime(event.timestamp)],
            ['Message index', p.message_index ?? '-'],
            ['Tool calls', calls.length],
          ]), 'info'),
          tokenDetailSection(event),
          promptDetailSection(event),
          calls.length ? detailSection('Requested Tool Calls', detailChips(calls.map(call => call.name || call.id || 'tool'), 'science')) : '',
          calls.length ? detailSection('Call Arguments', calls.map((call, index) => detailSection(
            `${index + 1}. ${call.name || call.id || 'tool'}`,
            detailValueHtml(call.args || call.arguments || {}),
          )).join('')) : '',
          p.content_preview ? detailSection('Response Preview', paragraphsHtml(p.content_preview)) : '',
        );
      }
      if (event.event.startsWith('tool_call_')) {
        return toolDetailHtml(event);
      }
      if (event.event === 'workflow_started' || event.event === 'run_started') {
        return detailRich(
          detailSection('Wake Input', detailKvGrid([
            ['Agent', event.agent_id || p.agent_id || '-'],
            ['Role', event.role || p.role || '-'],
            ['Round', p.round ?? '-'],
            ['Thread', p.thread_id || '-', 'mono'],
            ['Time', formatTime(event.timestamp)],
            ['Tool count', Array.isArray(p.tool_names) ? p.tool_names.length : '-'],
          ]), 'info'),
          Array.isArray(p.tool_names) && p.tool_names.length
            ? detailSection('Available Tools', detailChips(p.tool_names, 'science'))
            : '',
          p.query ? detailSection('Query', paragraphsHtml(p.query)) : '',
        );
      }
      if (event.event === 'workflow_finished' || event.event === 'workflow_output' || event.event === 'run_finished') {
        return detailRich(
          detailSection('Output', detailKvGrid([
            ['Status', p.status || event.event],
            ['Agent', event.agent_id || p.agent_id || '-'],
            ['Round', p.round ?? '-'],
            ['Time', formatTime(event.timestamp)],
          ]), statusTone(p.status)),
          tokenDetailSection(event),
          promptDetailSection(event),
          p.content_preview ? detailSection('Preview', paragraphsHtml(p.content_preview)) : '',
          p.error ? detailSection('Error', paragraphsHtml(p.error), 'error') : '',
        );
      }
      return workflowDetailHtml(event);
    }

    function chemgraphNodeContextHtml(event) {
      const p = event.payload || {};
      const turnEvents = workflowEventsForSelection(event);
      const flow = workflowFlowGraph(turnEvents);
      return detailRich(
        detailSection('Turn Context', detailKvGrid([
          ['Agent', workflowAgentId(event) || '-'],
          ['Runtime', p.runtime || '-'],
          ['Round', p.round ?? '-'],
          ['Thread', p.thread_id || '-', 'mono'],
          ['Workflow span', workflowRootSpanId(event) || '-', 'mono'],
          ['Selected span', p.span_id || event.correlation_id || '-', 'mono'],
          ['Visible nodes', flow.nodes.length],
          ['Visible events', turnEvents.length],
        ]), 'info'),
      );
    }

    function workflowHistoryHtml(events) {
      if (!events.length) return detailRich('<div class="muted">No related workflow events visible.</div>');
      return detailRich(events.map(event => {
        const p = event.payload || {};
        return detailSection(event.event, detailKvGrid([
          ['Time', formatTime(event.timestamp)],
          ['Status', p.status || '-'],
          ['Round', p.round ?? '-'],
          ['Tool', p.tool_name || '-'],
          ['Span', p.span_id || event.correlation_id || '-', 'mono'],
        ]), statusTone(p.status));
      }).join(''));
    }

    function toolDetailHtml(event) {
      const p = event.payload || {};
      const status = p.status || p.result?.status || event.event;
      return detailRich(
        detailSection('Tool', detailKvGrid([
          ['Tool name', toolDisplayName(event)],
          ['Status', status],
          ['Event', event.event],
          ['Time', formatTime(event.timestamp)],
          ['Agent', event.agent_id || '-'],
          ['Call id', p.tool_result_id || p.correlation_id || event.correlation_id || '-', 'mono'],
        ]), statusTone(status)),
        p.arguments ? detailSection('Arguments', detailValueHtml(p.arguments)) : '',
        p.error ? detailSection('Error', paragraphsHtml(p.error), 'error') : '',
        p.result ? detailSection('Result', detailValueHtml(p.result)) : '',
      );
    }

    function payloadDetailHtml(payload) {
      if (!payload || typeof payload !== 'object' || Array.isArray(payload)) {
        return detailRich(detailSection('Value', detailValueHtml(payload)), rawJsonDetails(payload));
      }
      const priority = [
        'status',
        'tool_name',
        'tool_result_id',
        'message_id',
        'sender',
        'recipient',
        'tldr',
        'content_preview',
        'error',
        'reason',
        'runtime',
        'workflow_type',
        'workflow_node',
        'round',
        'thread_id',
        'span_id',
        'parent_span_id',
        'correlation_id',
        'log_dir',
        'run_dir',
        'model_name',
      ];
      const prioritySet = new Set(priority);
      const priorityRows = priority
        .filter(key => Object.prototype.hasOwnProperty.call(payload, key))
        .map(key => [fieldLabel(key), payload[key], fieldKind(key)]);
      const otherRows = Object.keys(payload)
        .filter(key => !prioritySet.has(key))
        .sort()
        .map(key => [fieldLabel(key), payload[key], fieldKind(key)]);
      return detailRich(
        priorityRows.length ? detailSection('Key Fields', detailKvGrid(priorityRows), statusTone(payload.status)) : '',
        otherRows.length ? detailSection('Additional Fields', detailKvGrid(otherRows)) : '',
        rawJsonDetails(payload),
      );
    }

    function fieldLabel(key) {
      return String(key || '').replaceAll('_', ' ');
    }

    function fieldKind(key) {
      if (/(^|_)(id|dir|path|file|span|thread|correlation)($|_)/.test(String(key))) return 'mono';
      if (['content', 'content_preview', 'reason', 'error', 'tldr'].includes(String(key))) return 'text';
      return '';
    }

    function formatBelief(payload) {
      const lines = [
        `Hypothesis: ${payload.hypothesis || '-'}`,
        `Confidence: ${payload.confidence ?? '-'}`,
      ];
      if (payload.reason) lines.push(`Reason: ${payload.reason}`);
      if (payload.supporting_message_ids?.length) lines.push(`Messages: ${payload.supporting_message_ids.join(', ')}`);
      if (payload.supporting_artifact_ids?.length) lines.push(`Artifacts: ${payload.supporting_artifact_ids.join(', ')}`);
      return lines.join('\n');
    }

    function formatMessageEvent(event) {
      const p = event.payload || {};
      const lines = [
        `${formatTime(event.timestamp)} ${p.sender || event.agent_id || '-'} -> ${p.recipient || '-'}`,
      ];
      if (p.tldr) lines.push(`TLDR: ${p.tldr}`);
      if (p.content) lines.push(p.content);
      if (p.evidence_refs?.length) lines.push(`Evidence: ${p.evidence_refs.join(', ')}`);
      if (p.reason) lines.push(`Reason: ${p.reason}`);
      return lines.filter(Boolean).join('\n');
    }

    function formatMessageRoute(event) {
      const p = event.payload || {};
      const currentAgents = agents();
      const sender = currentAgents.find(agent => agent.agent_id === p.sender);
      const recipient = currentAgents.find(agent => agent.agent_id === p.recipient);
      const senderGroup = agentGroup(sender);
      const recipientGroup = agentGroup(recipient);
      const groupLabel = snapshot?.federated ? 'site' : 'host';
      const crossGroup = senderGroup && recipientGroup && senderGroup !== recipientGroup;
      const route = crossGroup
        ? (snapshot?.federated ? 'cross-site' : 'cross-node')
        : 'same-' + groupLabel;
      return [
        `Route: ${route}`,
        `Sender ${groupLabel}: ${senderGroup}`,
        `Recipient ${groupLabel}: ${recipientGroup}`,
        `Message id: ${p.message_id || '-'}`,
      ].join('\n');
    }

    function formatDecisionEvent(event) {
      const p = event.payload || {};
      const lines = [
        `${formatTime(event.timestamp)} ${event.agent_id || '-'} decision`,
      ];
      if (p.mode) lines.push(`Mode: ${p.mode}`);
      if (p.wake_reason) lines.push(`Wake reason: ${p.wake_reason}`);
      if (p.rationale) lines.push(`\nRationale:\n${p.rationale}`);
      const actions = Array.isArray(p.actions) ? p.actions : [];
      if (actions.length) {
        lines.push('\nActions:');
        actions.forEach((action, index) => {
          lines.push(formatDecisionToolCall(action, index + 1));
        });
      } else {
        lines.push('\nActions: none');
      }
      const ignored = Array.isArray(p.ignored_actions) ? p.ignored_actions : [];
      if (ignored.length) {
        lines.push(`\nIgnored actions: ${ignored.length}`);
      }
      return lines.join('\n');
    }

    function formatDecisionToolCall(action, number) {
      const parts = [`${number}. ${action.action || 'unknown_action'}`];
      if (action.recipient) parts.push(`recipient=${action.recipient}`);
      if (action.tool_name) parts.push(`tool=${action.tool_name}`);
      if (action.confidence !== null && action.confidence !== undefined) parts.push(`confidence=${action.confidence}`);
      let text = parts.join(' | ');
      if (action.question) text += `\n   Question: ${action.question}`;
      if (action.content) text += `\n   Content: ${action.content}`;
      if (action.hypothesis) text += `\n   Hypothesis: ${action.hypothesis}`;
      if (action.reason) text += `\n   Reason: ${action.reason}`;
      if (action.evidence_refs?.length) text += `\n   Evidence: ${action.evidence_refs.join(', ')}`;
      if (action.supporting_message_ids?.length) text += `\n   Messages: ${action.supporting_message_ids.join(', ')}`;
      if (action.supporting_artifact_ids?.length) text += `\n   Artifacts: ${action.supporting_artifact_ids.join(', ')}`;
      if (action.arguments && Object.keys(action.arguments).length) {
        text += `\n   Arguments: ${formatJson(action.arguments)}`;
      }
      return text;
    }

    function formatWakeEvents(event) {
      const wakeEvents = event.payload?.wake_events || [];
      if (!Array.isArray(wakeEvents) || !wakeEvents.length) {
        return 'No wake events recorded for this decision.';
      }
      return wakeEvents.map((wake, index) => {
        const payload = wake.payload || {};
        const summary = [
          `${index + 1}. ${formatTime(wake.timestamp)} ${wake.event}${wake.agent_id ? ` · ${wake.agent_id}` : ''}`,
        ];
        if (payload.message_id) summary.push(`   Message: ${payload.message_id}`);
        if (payload.sender || payload.recipient) summary.push(`   Route: ${payload.sender || '-'} -> ${payload.recipient || '-'}`);
        if (payload.tool_name) summary.push(`   Tool: ${payload.tool_name}`);
        if (payload.content) summary.push(`   Content: ${trunc(payload.content, 500)}`);
        if (payload.result) summary.push(`   Result: ${formatJson(payload.result)}`);
        return summary.join('\n');
      }).join('\n\n');
    }

    function formatToolEvent(event) {
      const p = event.payload || {};
      const status = p.status || p.result?.status || event.event;
      const toolName = toolDisplayName(event);
      const lines = [
        `${formatTime(event.timestamp)} ${event.event}`,
        `Tool: ${toolName}`,
        `Status: ${status}`,
      ];
      if (event.correlation_id) lines.push(`Call: ${event.correlation_id}`);
      const results = p.results || p.result?.results;
      if (Array.isArray(results)) {
        lines.push(`Results: ${results.map(item => `${item.index ?? '-'}:${item.status || item.error_type || '-'}`).join(', ')}`);
      }
      if (p.error) lines.push(`Error: ${p.error}`);
      return lines.join('\n');
    }

    function isWorkflowEvent(event) {
      const p = event.payload || {};
      const workflowNames = [
        'run_started',
        'run_finished',
        'workflow_started',
        'workflow_node_started',
        'workflow_node_finished',
        'workflow_output',
        'workflow_finished',
        'llm_decision',
        'tool_call_started',
        'tool_call_finished',
        'tool_call_failed',
      ];
      if (!workflowNames.includes(event.event)) return false;
      return Boolean(
        p.nested
        || p.runtime
        || p.workflow_type
        || p.workflow_span_id
        || p.span_id
        || p.parent_span_id
        || p.thread_id
        || (event.agent_id && p.round !== undefined && p.round !== null)
      );
    }

    function workflowSpanId(event) {
      const p = event.payload || {};
      return p.span_id || event.correlation_id || null;
    }

    function workflowParentSpanId(event) {
      const p = event.payload || {};
      return p.parent_span_id || null;
    }

    function workflowAgentId(event) {
      const p = event.payload || {};
      return event.agent_id || p.agent_id || p.agent_name || p.agent || null;
    }

    function workflowRootSpanId(event) {
      const p = event.payload || {};
      if (p.workflow_span_id) return p.workflow_span_id;
      if (p.thread_id) return p.thread_id;
      if ((event.event === 'workflow_started' || event.event === 'workflow_finished') && p.span_id) {
        return p.span_id;
      }
      if (p.parent_span_id || p.span_id || event.correlation_id) {
        return p.parent_span_id || p.span_id || event.correlation_id;
      }
      const agentId = workflowAgentId(event);
      if (agentId && p.round !== undefined && p.round !== null) {
        return `${agentId}-round-${p.round}`;
      }
      return null;
    }

    function workflowEventsForSpan(spanId) {
      if (!spanId) return [];
      return visibleEvents().filter(candidate => (
        isWorkflowEvent(candidate) && workflowRootSpanId(candidate) === spanId
      ));
    }

    function workflowEventsForSelection(event) {
      if (!event) return [];
      const p = event.payload || {};
      const spanId = p.workflow_span_id || (isWorkflowEvent(event) ? workflowRootSpanId(event) : null);
      if (spanId) return workflowEventsForSpan(spanId);
      const agentId = event.agent_id || p.agent_id || p.agent_name;
      const round = p.round;
      if (!agentId || round === undefined || round === null) return [];
      return visibleEvents().filter(candidate => (
        isWorkflowEvent(candidate)
        && workflowAgentId(candidate) === agentId
        && candidate.payload?.round === round
      ));
    }

    function reasoningTurnsForAgent(agentId) {
      if (!agentId) return [];
      const bySpan = new Map();
      visibleEvents().forEach(event => {
        if (!isWorkflowEvent(event) || workflowAgentId(event) !== agentId) return;
        const spanId = workflowRootSpanId(event);
        if (!spanId) return;
        if (!bySpan.has(spanId)) {
          bySpan.set(spanId, {
            spanId,
            events: [],
            round: null,
            threadId: '',
            started: null,
            finished: null,
            firstTimestamp: null,
            lastTimestamp: null,
          });
        }
        const turn = bySpan.get(spanId);
        turn.events.push(event);
        const p = event.payload || {};
        if (p.round !== undefined && p.round !== null) turn.round = p.round;
        if (p.thread_id) turn.threadId = p.thread_id;
        if (event.event === 'workflow_started') turn.started = event;
        if (event.event === 'workflow_finished') turn.finished = event;
        const ts = eventTimestamp(event);
        if (ts !== null) {
          turn.firstTimestamp = turn.firstTimestamp === null ? ts : Math.min(turn.firstTimestamp, ts);
          turn.lastTimestamp = turn.lastTimestamp === null ? ts : Math.max(turn.lastTimestamp, ts);
        }
      });
      return Array.from(bySpan.values())
        .map(turn => {
          const toolNodes = workflowFlowGraph(turn.events).nodes.filter(node => node.type === 'tool');
          const actionTools = toolNodes.filter(node => node.toolClass === 'action-tool');
          const scienceTools = toolNodes.filter(node => node.toolClass === 'science-tool');
          return {
            ...turn,
            status: turn.finished?.payload?.status || 'running',
            representative: turn.started || turn.events[0],
            lmCount: workflowTokenEvents(turn.events).length || turn.events.filter(event => event.event === 'llm_decision').length,
            tokenTotals: summedTokenCounts(turn.events),
            actionToolCount: actionTools.length,
            scienceToolCount: scienceTools.length,
          };
        })
        .sort((a, b) => (a.firstTimestamp ?? 0) - (b.firstTimestamp ?? 0));
    }

    function reasoningTurnListHtml(turns) {
      if (!turns.length) return '';
      const rows = turns.slice(-6).reverse().map(turn => {
        const key = eventKey(turn.representative);
        const round = turn.round === null || turn.round === undefined ? '-' : turn.round;
        const title = `Round ${round}`;
        const meta = [
          `${turn.status}`,
          `${turn.lmCount} LM`,
          turn.tokenTotals ? `${formatTokenCount(turn.tokenTotals.total)} tok` : '',
          `${turn.actionToolCount} action`,
          `${turn.scienceToolCount} science`,
          formatTime(turn.firstTimestamp),
        ].filter(Boolean).join(' · ');
        return `
          <button type="button" class="turn-row" data-detail-activity-key="${esc(key)}">
            <span class="turn-title">${esc(title)}</span>
            <span class="turn-meta">${esc(meta)}<br>${esc(trunc(turn.threadId || turn.spanId, 72))}</span>
          </button>
        `;
      }).join('');
      return `<div class="turn-list">${rows}</div>`;
    }

    function nestedWorkflowEventsForTool(event) {
      const p = event.payload || {};
      const callId = p.tool_result_id || p.correlation_id || event.correlation_id;
      if (!callId) return [];
      return allEvents().filter(candidate => {
        if (!isWorkflowEvent(candidate)) return false;
        return workflowParentSpanId(candidate) === callId || candidate.payload?.parent_tool_name === toolDisplayName(event);
      });
    }

    function relatedWorkflowEvents(event) {
      const spanId = workflowSpanId(event);
      const parentSpanId = workflowParentSpanId(event);
      if (!spanId && !parentSpanId) return [];
      return allEvents().filter(candidate => {
        if (!isWorkflowEvent(candidate)) return false;
        const candidateSpan = workflowSpanId(candidate);
        const candidateParent = workflowParentSpanId(candidate);
        return candidateSpan === spanId
          || candidateParent === spanId
          || (parentSpanId && (candidateSpan === parentSpanId || candidateParent === parentSpanId));
      });
    }

    function formatWorkflowEvent(event) {
      const p = event.payload || {};
      const lines = [
        `${formatTime(event.timestamp)} ${event.event}`,
        `Runtime: ${p.runtime || '-'}`,
        `Span: ${p.span_id || '-'}`,
      ];
      if (p.parent_span_id) lines.push(`Parent span: ${p.parent_span_id}`);
      if (p.workflow_type) lines.push(`Workflow: ${p.workflow_type}`);
      if (p.workflow_node) lines.push(`Node: ${p.workflow_node}`);
      if (p.phase) lines.push(`Phase: ${p.phase}`);
      if (p.status) lines.push(`Status: ${p.status}`);
      if (p.model_name) lines.push(`Model: ${p.model_name}`);
      if (p.log_dir) lines.push(`Log dir: ${p.log_dir}`);
      if (Array.isArray(p.tool_calls) && p.tool_calls.length) {
        lines.push(`Tool calls: ${p.tool_calls.map(call => call.name || call.id || 'tool').join(', ')}`);
      }
      if (p.tool_name) lines.push(`Tool: ${p.tool_name}`);
      if (p.content_preview) lines.push(`Preview: ${p.content_preview}`);
      if (p.error) lines.push(`Error: ${p.error}`);
      return lines.join('\n');
    }

    function formatWorkflowHistory(event) {
      const history = relatedWorkflowEvents(event);
      return history.length
        ? history.map(formatWorkflowEvent).join('\n\n')
        : 'No related workflow events visible.';
    }

    function formatJson(value) {
      try {
        return JSON.stringify(value, null, 2);
      } catch {
        return String(value);
      }
    }

    function toggleReplay() {
      if (isReplaying) {
        stopReplay(true);
      } else {
        startReplay();
      }
    }

    function startReplay() {
      const events = allEvents();
      if (!events.length) return;
      followLatest = false;
      isReplaying = true;
      graphMode = 'current';
      selectedEdgeKey = null;
      replayStartIndex = Math.max(0, currentEventIndex());
      replayStartTimestamp = eventTimestamp(events[replayStartIndex]) ?? firstTimestamp();
      replayStartedAtMs = Date.now();
      if (replayTimer) window.clearInterval(replayTimer);
      replayTimer = window.setInterval(tickReplay, 120);
      tickReplay();
    }

    function stopReplay(renderNow = false) {
      isReplaying = false;
      if (replayTimer) {
        window.clearInterval(replayTimer);
        replayTimer = null;
      }
      if (renderNow) render();
    }

    function tickReplay() {
      const events = allEvents();
      if (!events.length) {
        stopReplay(true);
        return;
      }
      if (replayStartTimestamp === null) {
        timelineIndex = Math.min(events.length - 1, currentEventIndex() + 1);
      } else {
        const elapsedSeconds = ((Date.now() - replayStartedAtMs) / 1000) * replaySpeed();
        timelineIndex = Math.max(
          replayStartIndex,
          eventIndexAtTimestamp(replayStartTimestamp + elapsedSeconds)
        );
      }
      render();
      if (timelineIndex >= events.length - 1) stopReplay(false);
    }

    function startGraphPan(event) {
      if (!graphView || event.button !== 0) return;
      if (event.target.closest('.agent-node') || event.target.closest('[data-edge-key]') || event.target.closest('.activity-bubble')) return;
      graphPanDrag = {
        clientX: event.clientX,
        clientY: event.clientY,
        startX: graphView.x,
        startY: graphView.y,
      };
      document.getElementById('graph').classList.add('panning');
      event.preventDefault();
    }

    function moveGraphPan(event) {
      if (!graphPanDrag || !graphView) return;
      const svg = document.getElementById('graph');
      const rect = svg.getBoundingClientRect();
      const dx = (event.clientX - graphPanDrag.clientX) * graphView.width / Math.max(rect.width, 1);
      const dy = (event.clientY - graphPanDrag.clientY) * graphView.height / Math.max(rect.height, 1);
      graphView.x = graphPanDrag.startX - dx;
      graphView.y = graphPanDrag.startY - dy;
      clampGraphView();
      updateGraphViewBox();
    }

    function stopGraphPan() {
      graphPanDrag = null;
      document.getElementById('graph').classList.remove('panning');
    }

    function startEmbeddedWorkflowPan(event) {
      if (!embeddedWorkflowView || event.button !== 0) return;
      if (event.target.closest('.workflow-node')) return;
      embeddedWorkflowPanDrag = {
        clientX: event.clientX,
        clientY: event.clientY,
        startX: embeddedWorkflowView.x,
        startY: embeddedWorkflowView.y,
      };
      document.getElementById('embeddedWorkflowGraph').classList.add('panning');
      event.preventDefault();
    }

    function moveEmbeddedWorkflowPan(event) {
      if (!embeddedWorkflowPanDrag || !embeddedWorkflowView) return;
      const svg = document.getElementById('embeddedWorkflowGraph');
      const rect = svg.getBoundingClientRect();
      const dx = (event.clientX - embeddedWorkflowPanDrag.clientX) * embeddedWorkflowView.width / Math.max(rect.width, 1);
      const dy = (event.clientY - embeddedWorkflowPanDrag.clientY) * embeddedWorkflowView.height / Math.max(rect.height, 1);
      embeddedWorkflowView.x = embeddedWorkflowPanDrag.startX - dx;
      embeddedWorkflowView.y = embeddedWorkflowPanDrag.startY - dy;
      clampEmbeddedWorkflowView();
      updateEmbeddedWorkflowViewBox();
    }

    function stopEmbeddedWorkflowPan() {
      embeddedWorkflowPanDrag = null;
      document.getElementById('embeddedWorkflowGraph').classList.remove('panning');
    }

    function startWorkflowPanelDrag(event) {
      if (event.button !== 0 || event.target.closest('button')) return;
      workflowPanelDrag = {
        clientX: event.clientX,
        clientY: event.clientY,
        startX: workflowPanelFrame.x,
        startY: workflowPanelFrame.y,
      };
      event.preventDefault();
    }

    function moveWorkflowPanelDrag(event) {
      if (!workflowPanelDrag) return;
      workflowPanelFrame.x = workflowPanelDrag.startX + event.clientX - workflowPanelDrag.clientX;
      workflowPanelFrame.y = workflowPanelDrag.startY + event.clientY - workflowPanelDrag.clientY;
      applyWorkflowPanelFrame();
    }

    function stopWorkflowPanelDrag() {
      workflowPanelDrag = null;
    }

    function startWorkflowPanelResize(event) {
      if (event.button !== 0) return;
      workflowPanelResizeDrag = {
        clientX: event.clientX,
        clientY: event.clientY,
        startWidth: workflowPanelFrame.width,
        startHeight: workflowPanelFrame.height,
      };
      event.preventDefault();
      event.stopPropagation();
    }

    function moveWorkflowPanelResize(event) {
      if (!workflowPanelResizeDrag) return;
      workflowPanelFrame.width = workflowPanelResizeDrag.startWidth + event.clientX - workflowPanelResizeDrag.clientX;
      workflowPanelFrame.height = workflowPanelResizeDrag.startHeight + event.clientY - workflowPanelResizeDrag.clientY;
      applyWorkflowPanelFrame();
    }

    function stopWorkflowPanelResize() {
      workflowPanelResizeDrag = null;
    }

    function startDetailResize(event) {
      if (event.button !== 0) return;
      const currentWidth = parseFloat(getComputedStyle(document.documentElement).getPropertyValue('--detail-width')) || 430;
      detailResizeDrag = {
        clientX: event.clientX,
        startWidth: currentWidth,
      };
      document.getElementById('detailResizer').classList.add('active');
      event.preventDefault();
    }

    function moveDetailResize(event) {
      if (!detailResizeDrag) return;
      const width = Math.min(760, Math.max(320, detailResizeDrag.startWidth - (event.clientX - detailResizeDrag.clientX)));
      document.documentElement.style.setProperty('--detail-width', `${width}px`);
    }

    function stopDetailResize() {
      detailResizeDrag = null;
      document.getElementById('detailResizer').classList.remove('active');
    }

    document.getElementById('refresh').addEventListener('click', load);
    document.getElementById('playReplay').addEventListener('click', toggleReplay);
    document.getElementById('timeSlider').addEventListener('input', e => {
      stopReplay(false);
      followLatest = false;
      timelineIndex = Number(e.target.value);
      render();
    });
    document.getElementById('latest').addEventListener('click', () => {
      stopReplay(false);
      followLatest = true;
      timelineIndex = allEvents().length - 1;
      render();
    });
    document.getElementById('eventHold').addEventListener('change', render);
    document.getElementById('zoomIn').addEventListener('click', () => zoomGraph(0.86));
    document.getElementById('zoomOut').addEventListener('click', () => zoomGraph(1.16));
    document.getElementById('zoomReset').addEventListener('click', resetGraphView);
    document.getElementById('graph').addEventListener('mousedown', startGraphPan);
    document.getElementById('graph').addEventListener('wheel', event => {
      if (!graphView) return;
      event.preventDefault();
      zoomGraph(event.deltaY < 0 ? 0.94 : 1.06);
    }, {passive: false});
    document.getElementById('embeddedWorkflowGraph').addEventListener('mousedown', startEmbeddedWorkflowPan);
    document.getElementById('embeddedWorkflowGraph').addEventListener('wheel', event => {
      if (!embeddedWorkflowView) return;
      event.preventDefault();
      zoomEmbeddedWorkflow(event.deltaY < 0 ? 0.94 : 1.06);
    }, {passive: false});
    document.getElementById('workflowZoomIn').addEventListener('click', () => zoomEmbeddedWorkflow(0.86));
    document.getElementById('workflowZoomOut').addEventListener('click', () => zoomEmbeddedWorkflow(1.16));
    document.getElementById('workflowZoomReset').addEventListener('click', resetEmbeddedWorkflowView);
    document.getElementById('workflowPanelClose').addEventListener('click', () => {
      workflowPanelOpen = false;
      renderEmbeddedWorkflowPanel();
    });
    document.getElementById('workflowFloatingTab').addEventListener('click', () => {
      workflowPanelOpen = true;
      renderEmbeddedWorkflowPanel();
    });
    document.getElementById('workflowFloatingHead').addEventListener('mousedown', startWorkflowPanelDrag);
    document.getElementById('workflowPanelResize').addEventListener('mousedown', startWorkflowPanelResize);
    document.getElementById('detailResizer').addEventListener('mousedown', startDetailResize);
    window.addEventListener('mousemove', moveGraphPan);
    window.addEventListener('mousemove', moveEmbeddedWorkflowPan);
    window.addEventListener('mousemove', moveWorkflowPanelDrag);
    window.addEventListener('mousemove', moveWorkflowPanelResize);
    window.addEventListener('mousemove', moveDetailResize);
    window.addEventListener('mouseup', stopGraphPan);
    window.addEventListener('mouseup', stopEmbeddedWorkflowPan);
    window.addEventListener('mouseup', stopWorkflowPanelDrag);
    window.addEventListener('mouseup', stopWorkflowPanelResize);
    window.addEventListener('mouseup', stopDetailResize);
    document.getElementById('detailPrimary').addEventListener('click', handleDetailPaneClick);
    document.getElementById('detailSecondary').addEventListener('click', handleDetailPaneClick);
    document.getElementById('detailTertiary').addEventListener('click', handleDetailPaneClick);
    document.querySelectorAll('#graphMode button').forEach(button => {
      button.addEventListener('click', () => {
        graphMode = button.dataset.mode;
        selectedEdgeKey = null;
        selectedActivityEventKey = null;
        render();
      });
    });
    document.getElementById('agentSelect').addEventListener('change', e => {
      selectedAgent = e.target.value || null;
      selectedEdgeKey = null;
      selectedActivityEventKey = null;
      render();
    });
    load();
    setInterval(load, 2000);

// window.__campaignDoc used to be prefetched from /api/campaign so
// the Observability detail panel could show static agent spec fields
// (mission etc). That endpoint is gone -- campaign selection is now
// canvas-only. The observability detail panel keeps its runtime-only
// view; use the canvas for spec.
