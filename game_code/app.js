(() => {
  const $ = id => document.getElementById(id);
  const canvas = $('world'), ctx = canvas.getContext('2d'), map = $('minimap'), mapctx = map.getContext('2d');
  const SAVE_KEY = 'formworld.block-and-bloom.v3', TICK_MS = 100;
  let state = FWWorld.create(), random = FW.rng(state.seed), size = FWArt.S;
  let selected = FW.PALETTE[0], category = 'Build', tool = 'build', gravity = true, guide = true, gridLines = false;
  let blueprint = FWWorld.plan(), targets = FWWorld.targetCells(blueprint);
  let history = [], redoStack = [], particles = [], moves = [], lastPhysics = 0, lastFrame = 0, lastSave = 0, lastUI = 0;
  let hover = null, pointer = null, lastPaint = '', pan = null, follow = false, demo = null, toastUntil = 0;
  let recording = false, records = [], recordLast = null, jumped = false;
  const keys = new Set();
  const hotItems = FW.PALETTE.slice(0, 9);

  function toast(text) { $('toast').textContent = text; $('toast').classList.add('visible'); toastUntil = performance.now() + 2800; }
  function pack() { return { ...state, g: FW.clone(state.g), back: FW.clone(state.back), player: { ...state.player }, camera: { ...state.camera }, rngState: random.state(), size }; }
  function restore(saved) {
    state = { ...saved, g: FW.clone(saved.g), back: FW.clone(saved.back), player: { ...saved.player }, camera: { ...saved.camera } };
    FW.prune(state.g, state.back);
    random = FW.rng(saved.seed); if (saved.rngState !== undefined) random.restore(saved.rngState);
    size = saved.size || FWArt.S; moves = []; demo = null; particles = []; clampCamera(); updateUI();
    record({ type: 'restore_world' });
  }
  function snapshot() { history.push(pack()); if (history.length > 65) history.shift(); redoStack = []; }
  function undo() { if (!history.length) return; redoStack.push(pack()); restore(history.pop()); toast('Last build undone.'); }
  function redo() { if (!redoStack.length) return; history.push(pack()); restore(redoStack.pop()); toast('Build restored.'); }
  function validate(saved) {
    return saved && saved.version === 3 && Number.isInteger(saved.seed) && saved.g?.length === FWWorld.HEIGHT && saved.back?.length === FWWorld.HEIGHT
      && [saved.g, saved.back].every(layer => layer.every(row => row.length === FWWorld.WIDTH && row.every(v => Number.isInteger(v) && v >= 0 && v < FW.colors.length)))
      && saved.g.at(-1).every(v => v === FW.GROUND) && saved.player && saved.camera
      && [saved.player.x, saved.player.y, saved.player.vx, saved.player.vy, saved.camera.x, saved.camera.y, saved.clock].every(Number.isFinite);
  }
  function save(manual = false) {
    try { localStorage.setItem(SAVE_KEY, JSON.stringify(pack())); $('save-status').textContent = `${manual ? 'Saved' : 'Autosaved'} locally · ${new Date().toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' })}`; if (manual) toast('World saved in this browser.'); }
    catch { $('save-status').textContent = 'Storage unavailable · export JSON to keep a copy'; if (manual) toast('Local save unavailable. Export JSON instead.'); }
  }
  function load() {
    try {
      const saved = JSON.parse(localStorage.getItem(SAVE_KEY));
      if (!validate(saved)) { toast('No compatible saved world yet.'); return; }
      snapshot(); restore(saved); toast('Saved meadow loaded.');
    } catch { toast('This saved world could not be loaded.'); }
  }
  function confirmAction(title, text, fn) {
    $('confirm-title').textContent = title; $('confirm-text').textContent = text;
    $('confirm-dialog').showModal(); $('confirm-yes').onclick = () => { $('confirm-dialog').close(); fn(); };
  }
  $('confirm-cancel').onclick = () => $('confirm-dialog').close();

  function materialButtons() {
    $('materials').replaceChildren();
    for (const m of FW.PALETTE.filter(m => m.group === category)) {
      const b = document.createElement('button'); b.className = 'material' + (m.id === selected.id ? ' active' : '');
      b.title = `${m.name}: ${m.hint}`; b.setAttribute('aria-label', m.name); b.setAttribute('aria-pressed', m.id === selected.id);
      const icon = document.createElement('canvas'); icon.width = icon.height = 30; FWArt.icon(icon, m.id);
      const label = document.createElement('span'); label.textContent = m.name;
      b.append(icon, label); b.onclick = () => choose(m); $('materials').append(b);
    }
  }
  function choose(m) {
    selected = m; category = m.group; tool = 'build'; materialButtons();
    document.querySelectorAll('.tabs button').forEach(b => { b.classList.toggle('active', b.dataset.group === category); b.setAttribute('aria-selected', b.dataset.group === category); });
    $('selected-name').textContent = m.name; $('selected-hint').textContent = m.hint;
    $('selected-layer').textContent = m.layer === 'back' ? 'BACKGROUND LAYER' : 'FOREGROUND BLOCK';
    FWArt.icon($('selected-icon'), m.id); updateTools();
    document.querySelectorAll('.hot-slot').forEach(b => b.classList.toggle('active', +b.dataset.id === selected.id));
  }
  document.querySelectorAll('.tabs button').forEach(b => b.onclick = () => {
    category = b.dataset.group;
    document.querySelectorAll('.tabs button').forEach(tab => { tab.classList.toggle('active', tab === b); tab.setAttribute('aria-selected', tab === b); });
    materialButtons();
  });
  for (const [i, m] of hotItems.entries()) {
    const b = document.createElement('button'); b.className = 'hot-slot'; b.dataset.id = m.id; b.title = `${m.name} (${i + 1})`; b.setAttribute('aria-label', `Quick select ${m.name}`);
    const icon = document.createElement('canvas'); icon.width = icon.height = 28; FWArt.icon(icon, m.id);
    const key = document.createElement('span'); key.textContent = i + 1; b.append(icon, key); b.onclick = () => choose(m); $('hotbar').append(b);
  }
  function updateTools() { document.querySelectorAll('[data-tool]').forEach(b => b.classList.toggle('active', b.dataset.tool === tool)); }
  document.querySelectorAll('[data-tool]').forEach(b => b.onclick = () => { tool = b.dataset.tool; updateTools(); });

  function burst(x, y, id) {
    for (let i = 0; i < 8; i++) particles.push({ x: x + .5, y: y + .5, vx: (Math.random() - .5) * 2.4, vy: -Math.random() * 2.8, size: 2 + Math.floor(Math.random() * 3), color: FW.colors[id], start: performance.now(), life: 450 + Math.random() * 300 });
  }
  function resetRecording() { records = []; recordLast = recording ? layers() : null; }
  function layers() { return { foreground: FW.clone(state.g), background: FW.clone(state.back), player: { ...state.player } }; }
  function record(action) {
    if (!recording) return;
    const after = layers(); records.push({ before: recordLast, action: { ...action }, after }); recordLast = after;
    if (records.length >= 1500) { recording = false; $('record-toggle').textContent = 'Record OFF'; toast('Recording limit reached. Export your 1,500 transitions.'); }
  }
  function hitsPlayer(x, y, material) {
    if (!FW.collidable(material)) return false;
    const p = state.player, top = material === FW.DOOR ? y - 1 : y;
    return x + 1 > p.x - .32 && x < p.x + .32 && y + 1 > p.y && top < p.y + 1.8;
  }
  function backHint(material) {
    if (material === FW.FLOWER) return 'Flowers need grass or dirt directly below.';
    if (material === FW.LADDER) return 'Ladders must touch a solid block, a wall, or another ladder above/below.';
    if (material === FW.TORCH) return 'Torches must touch a solid block or a wall.';
    return 'Walls and windows must touch a solid block or another wall. They cannot float.';
  }
  function action(a, silent = false) {
    const before = state.g[a.y]?.[a.x];
    if (a.layer !== 'back' && a.type === 'place' && hitsPlayer(a.x, a.y, a.material)) {
      if (!silent) toast('Move out of the way before placing a solid block.'); return false;
    }
    if (!FW.applyAction(state.g, a, state.back)) {
      if (!silent && a.material === FW.DOOR && a.type === 'place') toast('A door needs two empty cells above a solid floor. Click its bottom cell.');
      else if (!silent && a.layer === 'back' && a.type === 'place') toast(backHint(a.material));
      return false;
    }
    moves = []; state.builds++; record(a);
    burst(a.x, a.y, a.type === 'remove' ? before || selected.id : a.material || FW.WOOD);
    return true;
  }
  function paint(shift = false) {
    if (!hover || demo) return;
    const { x, y } = hover; if (!FW.inside(state.g, x, y)) return;
    const removing = pointer === 2 || tool === 'erase';
    const key = `${x}:${y}:${removing}:${shift}:${selected.id}`;
    if (lastPaint === key) return; lastPaint = key;
    if (tool === 'pick' && !removing) {
      const id = state.g[y][x] || state.back[y][x], m = FW.PALETTE.find(m => m.id === id || (m.id === FW.DOOR && FW.doorBase(state.g, x, y)));
      if (m) { choose(m); toast(`Picked ${m.name}.`); } return;
    }
    const layer = removing ? (shift || !state.g[y][x] ? 'back' : 'front') : selected.layer || 'front';
    action({ type: removing ? 'remove' : 'place', x, y, material: selected.id, layer });
    updateUI();
  }
  function point(e) {
    const rect = canvas.getBoundingClientRect();
    return { x: Math.floor((e.clientX - rect.left) * canvas.width / rect.width / size + state.camera.x),
      y: Math.floor((e.clientY - rect.top) * canvas.height / rect.height / size + state.camera.y) };
  }
  canvas.addEventListener('pointerdown', e => {
    canvas.focus({ preventScroll: true }); canvas.setPointerCapture(e.pointerId);
    if (e.button === 1) { pan = { x: e.clientX, y: e.clientY, camera: { ...state.camera } }; follow = false; return; }
    if (![0, 2].includes(e.button)) return;
    if (demo) { toast('Stop the demo before editing.'); return; }
    snapshot(); pointer = e.button; hover = point(e); lastPaint = ''; paint(e.shiftKey);
  });
  canvas.addEventListener('pointermove', e => {
    if (pan) {
      const rect = canvas.getBoundingClientRect();
      state.camera.x = pan.camera.x - (e.clientX - pan.x) / rect.width * canvas.width / size;
      state.camera.y = pan.camera.y - (e.clientY - pan.y) / rect.height * canvas.height / size; clampCamera(); return;
    }
    hover = point(e); if (pointer !== null) paint(e.shiftKey);
  });
  canvas.addEventListener('pointerup', () => { pointer = null; pan = null; lastPaint = ''; });
  canvas.addEventListener('pointercancel', () => { pointer = null; pan = null; });
  canvas.addEventListener('pointerleave', () => { if (pointer === null) hover = null; });
  canvas.addEventListener('contextmenu', e => e.preventDefault());
  canvas.addEventListener('wheel', e => { e.preventDefault(); zoom(e.deltaY < 0 ? 1 : -1); }, { passive: false });
  function clampCamera() {
    state.camera.x = Math.max(0, Math.min(state.g[0].length - canvas.width / size, state.camera.x));
    state.camera.y = Math.max(0, Math.min(state.g.length - canvas.height / size, state.camera.y));
  }
  function zoom(direction) {
    const cx = state.camera.x + canvas.width / size / 2, cy = state.camera.y + canvas.height / size / 2;
    size = Math.max(20, Math.min(44, size + direction * 4)); state.camera.x = cx - canvas.width / size / 2; state.camera.y = cy - canvas.height / size / 2;
    clampCamera(); $('zoom-label').textContent = `${Math.round(size / FWArt.S * 100)}%`;
  }
  function center() { state.camera.x = state.player.x - canvas.width / size / 2; state.camera.y = state.player.y - canvas.height / size * .68; clampCamera(); follow = true; }
  map.onclick = e => {
    const rect = map.getBoundingClientRect(); follow = false;
    state.camera.x = (e.clientX - rect.left) / rect.width * FWWorld.WIDTH - canvas.width / size / 2;
    state.camera.y = (e.clientY - rect.top) / rect.height * FWWorld.HEIGHT - canvas.height / size / 2; clampCamera();
  };

  function collide(x, y) {
    for (let iy = Math.floor(y); iy <= Math.floor(y + 1.8 - .001); iy++) for (let ix = Math.floor(x - .32); ix <= Math.floor(x + .32 - .001); ix++) {
      if (ix < 0 || ix >= state.g[0].length || iy >= state.g.length) return true;
      if (iy >= 0 && FW.collidable(state.g[iy][ix])) return true;
    }
    return false;
  }
  function updatePlayer(dt) {
    const p = state.player;
    const horizontal = (keys.has('d') ? 1 : 0) - (keys.has('a') ? 1 : 0);
    if (horizontal) p.facing = horizontal;
    p.vx = horizontal * (p.flying ? 9 : 5.5);
    const onLadder = [Math.floor(p.y + .5), Math.floor(p.y + 1.6)].some(y => state.back[y]?.[Math.floor(p.x)] === FW.LADDER);
    const vertical = (keys.has('s') ? 1 : 0) - (keys.has('w') ? 1 : 0);
    if (p.flying) p.vy = vertical * 8;
    else if (onLadder && vertical) p.vy = vertical * 4;
    else p.vy = Math.min(20, p.vy + dt * 30);
    if (keys.has(' ') && !jumped && (p.grounded || onLadder)) { p.vy = -10.5; p.grounded = false; jumped = true; }
    if (!keys.has(' ')) jumped = false;
    const steps = Math.max(1, Math.ceil(Math.max(Math.abs(p.vx * dt), Math.abs(p.vy * dt)) / .12));
    const canStep = p.grounded && !p.flying && p.vy >= 0;
    p.grounded = false;
    for (let i = 0; i < steps; i++) {
      const nx = p.x + p.vx * dt / steps;
      if (!collide(nx, p.y)) p.x = nx;
      else if (canStep && !collide(nx, p.y - 1)) { p.y -= 1; p.x = nx; }
      const ny = Math.max(0, p.y + p.vy * dt / steps);
      if (!collide(p.x, ny)) p.y = ny;
      else { if (p.vy > 0) p.grounded = true; p.vy = 0; }
    }
    if (collide(p.x, p.y)) {
      let rescued = false;
      for (let i = 1; i < 12; i++) if (!collide(p.x, p.y - i * .25)) { p.y -= i * .25; p.vy = 0; rescued = true; break; }
      if (!rescued) { p.x = 28.5; p.y = 27; }
    }
    if (horizontal || vertical) follow = true;
    if (follow) {
      state.camera.x += (p.x - canvas.width / size / 2 - state.camera.x) * Math.min(1, dt * 4);
      state.camera.y += (p.y - canvas.height / size * .68 - state.camera.y) * Math.min(1, dt * 3);
      clampCamera();
    }
  }
  function interact() {
    const p = state.player, candidates = [];
    for (let y = Math.max(0, Math.floor(p.y) - 2); y < Math.min(state.g.length, Math.floor(p.y) + 4); y++) for (let x = Math.max(0, Math.floor(p.x) - 3); x <= Math.min(state.g[0].length - 1, Math.floor(p.x) + 3); x++) {
      const base = FW.doorBase(state.g, x, y); if (!base || base[1] !== y) continue;
      candidates.push({ x, y, distance: Math.hypot(x + .5 - p.x, y - p.y - .8) });
    }
    candidates.sort((a, b) => a.distance - b.distance);
    const door = candidates[0]; if (!door) { toast('Walk closer to a door, then press E.'); return; }
    if (state.g[door.y][door.x] === FW.OPEN_DOOR && hitsPlayer(door.x, door.y, FW.DOOR)) { toast('Step out of the doorway before closing it.'); return; }
    snapshot(); action({ type: 'interact', ...door }); toast(state.g[door.y][door.x] === FW.OPEN_DOOR ? 'Door opened.' : 'Door closed.');
  }

  function preview() {
    const c = $('project-preview').getContext('2d'), img = $('project-preview');
    c.clearRect(0, 0, img.width, img.height); c.imageSmoothingEnabled = false;
    const cell = blueprint.name === 'Workshop' ? 9 : 10, ox = (img.width - blueprint.width * cell) / 2;
    for (const a of targets.sort((a, b) => (a.layer === 'back' ? 0 : 1) - (b.layer === 'back' ? 0 : 1))) {
      c.drawImage(FWArt.texture(a.material), ox + (a.x - blueprint.x0) * cell, 9 + (a.y - blueprint.top) * cell, cell, cell);
    }
    c.fillStyle = '#7b985c'; c.fillRect(18, 111, 164, 3);
  }
  function updateUI() {
    const result = FWWorld.progress(state.g, state.back, targets);
    $('completion').textContent = `${result.percent}%`; $('progress-bar').style.width = `${result.percent}%`;
    $('checklist').replaceChildren();
    for (const [name, counts] of Object.entries(result.groups)) {
      const div = document.createElement('div'); const done = counts.done === counts.total;
      div.className = 'check' + (done ? ' finished' : '');
      const mark = document.createElement('i'); mark.textContent = done ? '✓' : '';
      const label = document.createElement('span'); label.textContent = name;
      const count = document.createElement('span'); count.textContent = `${counts.done}/${counts.total}`; div.append(mark, label, count); $('checklist').append(div);
    }
    $('status').textContent = `${state.player.flying ? 'Flight mode' : 'Creative mode'} · ${state.builds} edits · ${gravity ? 'gravity on' : 'gravity paused'}`;
    $('coordinates').textContent = `X ${Math.floor(hover?.x ?? state.player.x)} / Y ${Math.floor(hover?.y ?? state.player.y)}`;
    $('world-day').textContent = `DAY ${String(state.day).padStart(2, '0')}`;
    $('undo').disabled = !history.length; $('redo').disabled = !redoStack.length;
    $('zoom-label').textContent = `${Math.round(size / FWArt.S * 100)}%`;
    if (result.percent === 100 && !state.completed) { state.completed = true; toast('Home, sweet home! Blueprint complete. Now make it your own.'); }
    if (result.percent < 100) state.completed = false;
  }
  $('project-select').onchange = () => {
    demo = null; blueprint = FWWorld.plan($('project-select').value); targets = FWWorld.targetCells(blueprint); preview(); updateUI();
  };
  $('blueprint-toggle').onclick = () => { guide = !guide; $('blueprint-toggle').textContent = `Guide ${guide ? 'ON' : 'OFF'}`; $('blueprint-toggle').classList.toggle('active', guide); };
  function beginDemo() {
    snapshot();
    for (let y = blueprint.top; y <= blueprint.floor; y++) for (let x = blueprint.x0; x < blueprint.x0 + blueprint.width; x++) { state.g[y][x] = 0; state.back[y][x] = 0; }
    state.player.x = blueprint.x0 - 2; state.player.y = blueprint.floor - 1.8;
    record({ type: 'demo_reset', bounds: { x: blueprint.x0, top: blueprint.top, width: blueprint.width, floor: blueprint.floor } });
    demo = { index: 0, last: 0 }; follow = false; state.camera.x = blueprint.x0 - 14; state.camera.y = blueprint.top - 5; clampCamera();
    $('demo').textContent = 'Stop building'; guide = true; $('blueprint-toggle').textContent = 'Guide ON'; $('blueprint-toggle').classList.add('active'); toast('Building foundation first, then frame, roof and details.');
  }
  function toggleDemo() {
    if (demo) { demo = null; $('demo').textContent = 'Watch it being built (P)'; toast('Build paused. Continue it yourself.'); return; }
    confirmAction('Build the example?', 'This clears only the outlined build site. Your starter house and terrain stay intact. Undo restores the previous build.', beginDemo);
  }
  $('demo').onclick = toggleDemo;
  function download(name, data, mime) {
    const a = document.createElement('a'), url = URL.createObjectURL(new Blob([data], { type: mime })); a.href = url; a.download = name; a.click(); setTimeout(() => URL.revokeObjectURL(url), 1500);
  }
  $('export').onclick = () => {
    download('block-and-bloom.json', JSON.stringify({ schema: 'formworld.sandbox.v3', world: pack(), blueprint, transitions: records }), 'application/json');
    toast(`Exported world + ${records.length} recorded transitions.`);
  };
  $('screenshot').onclick = () => { canvas.toBlob(blob => { if (blob) download('willow-meadow.png', blob, 'image/png'); }); };
  $('record-toggle').onclick = () => { recording = !recording; recordLast = recording ? layers() : null; $('record-toggle').textContent = `Record ${recording ? 'ON' : 'OFF'}`; toast(recording ? 'Recording exact states, layers and physics ticks.' : `Recording stopped. ${records.length} transitions ready to export.`); };
  $('save').onclick = () => save(true); $('load').onclick = load;
  $('new-world').onclick = () => confirmAction('A fresh meadow?', 'Current changes will be replaced. Undo can restore them. Use Save first if you want to keep a permanent copy.', () => {
    snapshot(); state = FWWorld.create(); random = FW.rng(state.seed); moves = []; demo = null; resetRecording(); clampCamera(); updateUI(); toast('Fresh grass. Fresh ideas.');
  });
  $('undo').onclick = undo; $('redo').onclick = redo;
  $('gravity-toggle').onclick = () => { gravity = !gravity; moves = []; $('gravity-toggle').textContent = `Gravity ${gravity ? 'ON' : 'OFF'}`; $('gravity-toggle').classList.toggle('active', gravity); toast(gravity ? 'Testing your structure with gravity.' : 'Free design: structural gravity paused.'); };
  $('grid-toggle').onclick = () => { gridLines = !gridLines; $('grid-toggle').classList.toggle('active', gridLines); };
  $('day-toggle').onclick = () => { state.clock = state.clock > .05 && state.clock < .5 ? .73 : .25; $('day-toggle').textContent = state.clock < .5 ? 'Daylight' : 'Moonlight'; };
  $('zoom-in').onclick = () => zoom(1); $('zoom-out').onclick = () => zoom(-1); $('center-camera').onclick = center;
  $('help').onclick = () => $('help-dialog').showModal(); $('close-help').onclick = $('start-playing').onclick = () => $('help-dialog').close();
  window.addEventListener('keydown', e => {
    if ($('help-dialog').open || $('confirm-dialog').open || ['SELECT', 'INPUT', 'TEXTAREA'].includes(e.target.tagName)) return;
    const k = e.key.toLowerCase();
    if ([' ', 'arrowup', 'arrowdown', 'arrowleft', 'arrowright'].includes(k)) e.preventDefault();
    keys.add(k); if (e.repeat) return;
    if (/^[1-9]$/.test(k)) choose(hotItems[+k - 1]);
    else if (k === 'b') { tool = 'build'; updateTools(); }
    else if (k === 'x') { tool = 'erase'; updateTools(); }
    else if (k === 'q') { tool = 'pick'; updateTools(); }
    else if (k === 'e') interact();
    else if (k === 'f') { state.player.flying = !state.player.flying; state.player.vy = 0; toast(state.player.flying ? 'Flight enabled. WASD to fly.' : 'Back on your feet.'); }
    else if (k === 'g') $('grid-toggle').click();
    else if (k === 'z') undo(); else if (k === 'y') redo();
    else if (k === 'p') toggleDemo(); else if (k === 'home') center(); else if (k === '?') $('help-dialog').showModal();
    else if (k === 'n') { const res = FW.tick(state.g, random); FW.prune(state.g, state.back); moves = res.moves; lastPhysics = performance.now(); record({ type: 'physics_tick' }); }
  });
  window.addEventListener('keyup', e => keys.delete(e.key.toLowerCase()));
  window.addEventListener('blur', () => { keys.clear(); pointer = null; pan = null; });
  function loop(t) {
    const dt = Math.min(.035, (t - (lastFrame || t)) / 1000); lastFrame = t;
    const modal = $('help-dialog').open || $('confirm-dialog').open;
    if (!modal) {
      updatePlayer(dt);
      if (keys.has('arrowleft')) { state.camera.x -= dt * 12; follow = false; }
      if (keys.has('arrowright')) { state.camera.x += dt * 12; follow = false; }
      if (keys.has('arrowup')) { state.camera.y -= dt * 12; follow = false; }
      if (keys.has('arrowdown')) { state.camera.y += dt * 12; follow = false; } clampCamera();
      if (gravity && t - lastPhysics >= TICK_MS) { const res = FW.tick(state.g, random); lastPhysics = t; moves = res.moves; if (res.changed) { FW.prune(state.g, state.back); record({ type: 'physics_tick' }); } }
      if (demo && t - demo.last > 85) {
        demo.last = t;
        if (demo.index < blueprint.actions.length) action(blueprint.actions[demo.index++], true);
        else { demo = null; $('demo').textContent = 'Watch it being built (P)'; toast('Your house is built. Walk inside and open the door with E.'); }
      }
      state.clock += dt / 480; if (state.clock >= 1) { state.clock -= 1; state.day++; }
    }
    for (const p of particles) { p.x += p.vx * dt; p.y += p.vy * dt; p.vy += dt * 9; }
    particles = particles.filter(p => t - p.start < p.life);
    FWArt.draw(ctx, state.g, { back: state.back, camera: state.camera, size, time: t, clock: state.clock,
      player: state.player, ghost: guide ? targets : null, particles, moves, progress: Math.min(1, (t - lastPhysics) / TICK_MS),
      gridLines, hover, brush: selected, tool });
    if (t - lastUI > 200) { updateUI(); FWArt.minimap(mapctx, state.g, state.camera, size, state.player); lastUI = t; }
    if (t > toastUntil) $('toast').classList.remove('visible');
    if (!modal && !demo && t - lastSave > 10000) { save(); lastSave = t; }
    requestAnimationFrame(loop);
  }
  choose(selected); preview(); clampCamera(); updateUI(); toast('A / D to explore. Drag to build your first house.');
  window.FormGame = {
    get state() { return state; }, get targets() { return targets; }, get recording() { return recording; }, get records() { return records; },
    action, choose, progress: () => FWWorld.progress(state.g, state.back, targets), pack, restore, validate,
  };
  requestAnimationFrame(loop);
})();
