const FW = (() => {
  const W = 28, H = 22;
  const [EMPTY, GROUND, STONE, WOOD, LEAF, SAND, WATER, CLOUD,
    DIRT, GRASS, BRICK, GLASS, ROOF, DOOR, WALL, TORCH, LADDER, FLOWER,
    OPEN_DOOR, DOOR_TOP, OPEN_DOOR_TOP] = Array.from({ length: 21 }, (_, i) => i);
  const LOAD = new Set([GROUND, STONE, WOOD, DIRT, GRASS, BRICK, GLASS, ROOF]);
  const STRUCT = new Set([...LOAD].filter(v => v !== GROUND));
  for (const door of [DOOR, DOOR_TOP, OPEN_DOOR, OPEN_DOOR_TOP]) LOAD.add(door);
  const LOOSE = new Set([LEAF, SAND, WATER]);
  const MOVABLE = new Set(Array.from({ length: 21 }, (_, i) => i).filter(v => ![0, 1, 7].includes(v)));
  const SPAN = { [STONE]: 2, [WOOD]: 4, [DIRT]: 0, [GRASS]: 0, [BRICK]: 6, [GLASS]: 2, [ROOF]: 6 };
  const DOORS = new Set([DOOR, DOOR_TOP, OPEN_DOOR, OPEN_DOOR_TOP]);
  const PALETTE = [
    { id: WOOD, name: 'Oak planks', group: 'Build', hint: 'Floors & beams · span 4', color: '#b78143' },
    { id: STONE, name: 'Cobblestone', group: 'Build', hint: 'Foundations · span 2', color: '#838a90' },
    { id: BRICK, name: 'Brick', group: 'Build', hint: 'Strong walls · span 6', color: '#bf6750' },
    { id: ROOF, name: 'Roof tiles', group: 'Build', hint: 'Terracotta roofing · span 6', color: '#934b38' },
    { id: WALL, name: 'Timber wall', group: 'Build', hint: 'Background layer · walk through', color: '#806244', layer: 'back' },
    { id: GLASS, name: 'Window', group: 'Build', hint: 'Background window · replaces timber wall', color: '#8bd6df', layer: 'back' },
    { id: DOOR, name: 'Oak door', group: 'Build', hint: 'Two blocks high · E opens nearby doors', color: '#a4723e' },
    { id: TORCH, name: 'Torch', group: 'Decor', hint: 'Warm flickering light', color: '#f2b04d', layer: 'back' },
    { id: LADDER, name: 'Ladder', group: 'Decor', hint: 'Climb with W / S', color: '#c89c58', layer: 'back' },
    { id: DIRT, name: 'Dirt', group: 'Nature', hint: 'Shape the landscape', color: '#896044' },
    { id: GRASS, name: 'Grass', group: 'Nature', hint: 'Living green ground', color: '#68a849' },
    { id: LEAF, name: 'Leaves', group: 'Nature', hint: 'Connected canopy · never bears weight', color: '#418b48' },
    { id: FLOWER, name: 'Wildflower', group: 'Decor', hint: 'Plant on grass or dirt', color: '#edbe65', layer: 'back' },
    { id: SAND, name: 'Sand', group: 'Nature', hint: 'Loose grains fall and slide', color: '#d9c18a' },
    { id: WATER, name: 'Water', group: 'Nature', hint: 'Flows downhill · not load-bearing', color: '#529bc0' },
  ];
  const NAME = Object.fromEntries(PALETTE.map(m => [m.id, m.name]));
  const DESC = Object.fromEntries(PALETTE.map(m => [m.id, m.hint]));
  const colors = ['#d9ebf1', '#454851', '#838a90', '#b78143', '#418b48', '#d9c18a', '#529bc0', '#f4fafb',
    '#896044', '#68a849', '#bf6750', '#8bd6df', '#934b38', '#a4723e', '#806244', '#f2b04d', '#c89c58', '#edbe65', '#69422b', '#a16a35', '#543724'];
  const blank = (w = W, h = H) => Array.from({ length: h }, () => Array(w).fill(EMPTY));
  const zeros = (w = W, h = H) => { const g = blank(w, h); g[h - 1].fill(GROUND); return g; };
  const clone = g => g.map(row => row.slice());
  const inside = (g, x, y) => Number.isInteger(x) && Number.isInteger(y) && y >= 0 && y < g.length && x >= 0 && x < g[0].length;
  function rng(seed = 1) {
    let s = seed >>> 0;
    const next = () => { s = (Math.imul(1664525, s) + 1013904223) >>> 0; return s / 4294967296; };
    next.state = () => s; next.restore = v => { s = v >>> 0; }; return next;
  }
  function supported(g) {
    const h = g.length, w = g[0].length, cost = blank(w, h).map(r => r.map(() => -1)), q = [];
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) if (g[y][x] === GROUND) {
      cost[y][x] = 0; q.push([x, y]);
    }
    for (let i = 0; i < q.length; i++) {
      const [x, y] = q[i];
      for (const [dx, dy, inc] of [[0, -1, 0], [-1, 0, 1], [1, 0, 1]]) {
        const nx = x + dx, ny = y + dy;
        if (!inside(g, nx, ny) || !LOAD.has(g[ny][nx])) continue;
        const nc = cost[y][x] + inc;
        if (nc > (SPAN[g[ny][nx]] ?? 0) || (cost[ny][nx] >= 0 && cost[ny][nx] <= nc)) continue;
        cost[ny][nx] = nc; q.push([nx, ny]);
      }
    }
    return new Set(q.map(([x, y]) => y * w + x));
  }
  function attachedLeaves(g) {
    const w = g[0].length, q = [], seen = new Set();
    for (let y = 0; y < g.length; y++) for (let x = 0; x < w; x++) if (g[y][x] === WOOD) q.push([x, y, 0]);
    for (let i = 0; i < q.length; i++) {
      const [x, y, distance] = q[i]; if (distance >= 4) continue;
      for (const [dx, dy] of [[1, 0], [-1, 0], [0, 1], [0, -1]]) {
        const nx = x + dx, ny = y + dy, key = ny * w + nx;
        if (inside(g, nx, ny) && g[ny][nx] === LEAF && !seen.has(key)) {
          seen.add(key); q.push([nx, ny, distance + 1]);
        }
      }
    }
    return seen;
  }
  function waterTarget(g, x, y) {
    for (const dx of [0, -1, 1]) if (inside(g, x + dx, y + 1) && g[y + 1][x + dx] === EMPTY) return [x + dx, y + 1];
    const candidates = [];
    for (const direction of [-1, 1]) {
      for (let distance = 1; distance < g[0].length; distance++) {
        const nx = x + direction * distance;
        if (!inside(g, nx, y) || ![EMPTY, WATER].includes(g[y][nx])) break;
        if (inside(g, nx, y + 1) && g[y + 1][nx] === EMPTY) { candidates.push([distance, direction]); break; }
      }
    }
    candidates.sort((a, b) => a[0] - b[0] || a[1] - b[1]);
    if (candidates.length && g[y][x + candidates[0][1]] === EMPTY) return [x + candidates[0][1], y];
    return null;
  }
  function tick(g, rand = rng(1)) {
    const h = g.length, w = g[0].length, ok = supported(g), leaves = attachedLeaves(g);
    const xs = Array.from({ length: w }, (_, i) => i);
    for (let i = w - 1; i > 0; i--) { const j = Math.floor(rand() * (i + 1)); [xs[i], xs[j]] = [xs[j], xs[i]]; }
    const touched = new Set(), moves = [];
    const move = (x, y, nx, ny) => {
      const v = g[y][x], swapped = g[ny][nx];
      g[ny][nx] = v; g[y][x] = swapped; touched.add(y * w + x); touched.add(ny * w + nx);
      moves.push({ from: [x, y], to: [nx, ny], v, swapped });
    };
    for (let y = h - 2; y >= 0; y--) for (const x of xs) {
      if (STRUCT.has(g[y][x]) && !ok.has(y * w + x) && !touched.has(y * w + x)
          && [EMPTY, ...LOOSE].includes(g[y + 1][x])) move(x, y, x, y + 1);
    }
    for (let y = h - 2; y > 0; y--) for (const x of xs) {
      if (![DOOR, OPEN_DOOR].includes(g[y][x]) || touched.has(y * w + x)
          || ![EMPTY, ...LOOSE].includes(g[y + 1][x])) continue;
      const top = g[y - 1][x], bottom = g[y][x], displaced = g[y + 1][x];
      if (![DOOR_TOP, OPEN_DOOR_TOP].includes(top)) continue;
      g[y + 1][x] = bottom; g[y][x] = top; g[y - 1][x] = displaced;
      for (const row of [y - 1, y, y + 1]) touched.add(row * w + x);
      moves.push({ from: [x, y], to: [x, y + 1], v: bottom, swapped: 0 },
        { from: [x, y - 1], to: [x, y], v: top, swapped: 0 });
      if (displaced) moves.push({ from: [x, y + 1], to: [x, y - 1], v: displaced, swapped: 0 });
    }
    for (let y = h - 2; y >= 0; y--) for (const x of xs) {
      if (g[y][x] === LEAF && !leaves.has(y * w + x) && !touched.has(y * w + x)
          && [EMPTY, WATER].includes(g[y + 1][x])) move(x, y, x, y + 1);
    }
    for (let y = h - 2; y >= 0; y--) for (const x of xs) {
      if (g[y][x] !== SAND || touched.has(y * w + x)) continue;
      const directions = rand() < .5 ? [0, -1, 1] : [0, 1, -1];
      for (const dx of directions) if (inside(g, x + dx, y + 1) && [EMPTY, WATER].includes(g[y + 1][x + dx])) {
        move(x, y, x + dx, y + 1); break;
      }
    }
    for (let y = h - 2; y >= 0; y--) for (const x of xs) if (g[y][x] === WATER && !touched.has(y * w + x)) {
      const target = waterTarget(g, x, y); if (target) move(x, y, ...target);
    }
    return { changed: !!moves.length, moves };
  }
  function doorBase(g, x, y) {
    if (!inside(g, x, y) || !DOORS.has(g[y][x])) return null;
    return [x, [DOOR_TOP, OPEN_DOOR_TOP].includes(g[y][x]) ? y + 1 : y];
  }
  const BACK = new Set([WALL, GLASS, TORCH, LADDER, FLOWER]);
  const PANEL = new Set([WALL, GLASS]);
  const NEIGHBOURS = [[0, -1], [-1, 0], [1, 0], [0, 1]];
  const anchoredByBlock = (g, x, y, m) => {
    if (m === FLOWER) return inside(g, x, y + 1) && [GRASS, DIRT].includes(g[y + 1][x]);
    return [[0, 0], ...NEIGHBOURS].some(([dx, dy]) => inside(g, x + dx, y + dy) && LOAD.has(g[y + dy][x + dx]));
  };
  const attaches = (from, to, vertical) => (PANEL.has(from) && (PANEL.has(to) || to === TORCH || to === LADDER))
    || (from === LADDER && to === LADDER && vertical);
  function backLive(g, back) {
    const w = g[0].length, live = new Set(), q = [];
    for (let y = 0; y < back.length; y++) for (let x = 0; x < w; x++) {
      if (BACK.has(back[y][x]) && anchoredByBlock(g, x, y, back[y][x])) { live.add(y * w + x); q.push([x, y]); }
    }
    for (let i = 0; i < q.length; i++) {
      const [x, y] = q[i];
      for (const [dx, dy] of NEIGHBOURS) {
        const nx = x + dx, ny = y + dy;
        if (!inside(back, nx, ny) || live.has(ny * w + nx)) continue;
        if (attaches(back[y][x], back[ny][nx], dx === 0)) { live.add(ny * w + nx); q.push([nx, ny]); }
      }
    }
    return live;
  }
  function prune(g, back) {
    if (!back) return 0;
    const w = g[0].length, live = backLive(g, back);
    let removed = 0;
    for (let y = 0; y < back.length; y++) for (let x = 0; x < w; x++) {
      if (back[y][x] && !live.has(y * w + x)) { back[y][x] = EMPTY; removed++; }
    }
    return removed;
  }
  function canAttach(g, back, x, y, m) {
    if (anchoredByBlock(g, x, y, m)) return true;
    if (m === FLOWER) return false;
    return NEIGHBOURS.some(([dx, dy]) => inside(back, x + dx, y + dy) && attaches(back[y + dy][x + dx], m, dx === 0));
  }
  function applyAction(g, a, back = null) {
    const { type, x = 0, y = 0, material = STONE, layer = 'front' } = a;
    if (type === 'noop') return true;
    if (!inside(g, x, y) || y === g.length - 1) return false;
    if (layer === 'back') {
      if (!back || !inside(back, x, y)) return false;
      if (type === 'remove') { if (!back[y][x]) return false; back[y][x] = EMPTY; prune(g, back); return true; }
      if (type === 'place' && BACK.has(material)) {
        if (back[y][x] === material) return false;
        const old = back[y][x]; back[y][x] = EMPTY;
        if (!canAttach(g, back, x, y, material)) { back[y][x] = old; return false; }
        back[y][x] = material; prune(g, back); return true;
      }
      return false;
    }
    if (type === 'interact') {
      const base = doorBase(g, x, y); if (!base) return false;
      const [dx, dy] = base, open = g[dy][dx] === DOOR;
      g[dy][dx] = open ? OPEN_DOOR : DOOR; g[dy - 1][dx] = open ? OPEN_DOOR_TOP : DOOR_TOP;
      return true;
    }
    if (type === 'remove') {
      if (!MOVABLE.has(g[y][x])) return false;
      const base = doorBase(g, x, y);
      if (base) { g[base[1]][x] = EMPTY; g[base[1] - 1][x] = EMPTY; } else g[y][x] = EMPTY;
      prune(g, back); return true;
    }
    if (type === 'place') {
      if (!MOVABLE.has(material) || g[y][x] !== EMPTY) return false;
      if (material === DOOR) {
        if (y < 1 || g[y - 1][x] !== EMPTY || !LOAD.has(g[y + 1][x])) return false;
        g[y][x] = DOOR; g[y - 1][x] = DOOR_TOP;
      } else if (![OPEN_DOOR, DOOR_TOP, OPEN_DOOR_TOP, WALL, TORCH, LADDER, FLOWER].includes(material)) g[y][x] = material;
      else return false;
      return true;
    }
    if (type === 'drop' && (STRUCT.has(material) || LOOSE.has(material))) {
      for (let yy = 0; yy < g.length - 1; yy++) if (g[yy][x] === EMPTY) { g[yy][x] = material; return true; }
    }
    return false;
  }
  function step(g, a, rand = rng(1), back = null) {
    if (a) applyAction(g, a, back);
    for (let i = 0; i < g.length * 8; i++) if (!tick(g, rand).changed) break;
    prune(g, back);
  }
  const collidable = v => ![OPEN_DOOR, OPEN_DOOR_TOP].includes(v) && (LOAD.has(v) || v === SAND);
  function demoActions() {
    const acts = [], top = H - 6;
    for (const x of [7, 12]) for (let y = H - 2; y > top; y--) acts.push({ type: 'place', x, y, material: STONE });
    for (let x = 7; x < 13; x++) acts.push({ type: 'place', x, y: top, material: WOOD });
    return acts;
  }
  function makeTarget() { const g = zeros(), r = rng(7); for (const a of demoActions()) step(g, a, r); return g; }
  return { W, H, EMPTY, GROUND, STONE, WOOD, LEAF, SAND, WATER, CLOUD, DIRT, GRASS, BRICK, GLASS, ROOF,
    DOOR, WALL, TORCH, LADDER, FLOWER, OPEN_DOOR, DOOR_TOP, OPEN_DOOR_TOP, LOAD, STRUCT, LOOSE, MOVABLE,
    SPAN, NAME, DESC, PALETTE, colors, blank, zeros, clone, inside, rng, supported, tick, step, applyAction,
    collidable, doorBase, demoActions, makeTarget, prune, canAttach };
})();
if (typeof module !== 'undefined') module.exports = FW;
