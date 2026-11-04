const FWArt = (() => {
  const S = 28, PW = 1152, PH = 672;
  const BASE = Object.fromEntries(FW.colors.map((color, id) => [id, color.match(/[a-f0-9]{2}/g).map(n => parseInt(n, 16))]));
  const EMPTY_RGB = BASE[FW.EMPTY];
  const textures = new Map();
  function texture(v, variant = 0) {
    const key = `${v}:${variant}`;
    if (textures.has(key)) return textures.get(key);
    const canvas = document.createElement('canvas'); canvas.width = canvas.height = 16;
    const c = canvas.getContext('2d'), r = FW.rng(v * 8923 + variant * 31);
    const rect = (x, y, w, h, color) => { c.fillStyle = color; c.fillRect(x, y, w, h); };
    const fill = color => rect(0, 0, 16, 16, color);
    const speckles = (colors, count = 24) => { for (let i = 0; i < count; i++) rect(Math.floor(r() * 15), Math.floor(r() * 15), 1 + Math.floor(r() * 3), 1 + Math.floor(r() * 2), colors[i % colors.length]); };
    const bevel = () => { rect(0, 0, 16, 1, '#ffffff22'); rect(0, 0, 1, 16, '#ffffff12'); rect(0, 15, 16, 1, '#00000024'); rect(15, 0, 1, 16, '#0000001a'); };
    switch (v) {
      case FW.GROUND: fill('#41434a'); speckles(['#35373f', '#50515a', '#656268']); break;
      case FW.DIRT:
      case FW.GRASS:
        fill('#80583f'); speckles(['#654633', '#9b714e', '#b28459', '#704c37'], 35);
        if (v === FW.GRASS) { rect(0, 0, 16, 3, '#65a544'); rect(0, 0, 16, 1, '#9acf56');
          for (let x = 0; x < 16; x++) rect(x, 3, 1, 1 + Math.floor(r() * 3), x % 3 ? '#508b39' : '#77b443'); }
        break;
      case FW.STONE: {
        fill('#505f69');
        for (let y = 0; y < 16; y += 5) for (let x = (y % 2 ? -3 : 0); x < 16; x += 7) {
          rect(x + 1, y + 1, 6, 4, ['#84918c', '#78847f', '#9aa19a'][Math.floor(r() * 3)]);
          rect(x + 1, y + 1, 5, 1, '#bdc0ae66'); rect(x + 5, y + 3, 2, 2, '#33444b33');
        } break;
      }
      case FW.WOOD:
      case FW.WALL:
        fill(v === FW.WALL ? '#705039' : '#b17a3e');
        for (let y = 0; y < 16; y += 4) {
          rect(0, y, 16, 1, v === FW.WALL ? '#4a382b' : '#734b2b');
          rect(0, y + 1, 16, 1, v === FW.WALL ? '#866143' : '#d09b58');
          rect((y * 3 + variant) % 14, y, 1, 4, v === FW.WALL ? '#49372c' : '#855831');
        }
        speckles(v === FW.WALL ? ['#79573b', '#624733'] : ['#c39356', '#ae743b'], 12); break;
      case FW.BRICK:
        fill('#dec09a');
        for (let y = 0; y < 16; y += 4) for (let x = (y % 8 ? -4 : 0); x < 16; x += 8) {
          rect(x, y, 7, 3, ['#b6674c', '#c57854', '#aa5746'][Math.floor(r() * 3)]);
          rect(x, y, 7, 1, '#e797683f'); rect(x, y + 2, 7, 1, '#793f3433');
        } break;
      case FW.ROOF:
        fill('#6a3630');
        for (let y = 0; y < 16; y += 4) for (let x = (y % 8 ? -3 : 0); x < 16; x += 6) {
          rect(x, y, 5, 3, '#ad5740'); rect(x, y, 5, 1, '#df8b57'); rect(x + 1, y + 2, 4, 1, '#7c3d33');
        } break;
      case FW.GLASS:
        fill('#4a382a'); rect(1, 1, 14, 14, '#bd9660'); rect(2, 2, 12, 12, '#629caa');
        rect(2, 2, 5, 5, '#b6e2de'); rect(9, 2, 5, 5, '#8cc8d2');
        rect(2, 9, 5, 5, '#6db1c1'); rect(9, 9, 5, 5, '#aadcd9');
        rect(7, 1, 2, 14, '#d7b57b'); rect(1, 7, 14, 2, '#d7b57b');
        rect(3, 3, 1, 3, '#edf8dd'); rect(4, 3, 2, 1, '#edf8dd'); break;
      case FW.LEAF:
        fill('#367345'); speckles(['#488c49', '#62a84d', '#7cb853', '#2e623d', '#3d8241'], 50);
        rect(0, 0, 2, 2, '#96c868'); rect(8, 2, 2, 1, '#acd575'); break;
      case FW.SAND:
        fill('#d9bc80'); speckles(['#efd499', '#c5a369', '#e6c688'], 32); break;
      case FW.WATER:
        fill('#478ebdbb'); rect(0, 0, 16, 2, '#a3d9e8'); rect(0, 6, 6, 1, '#79bcdd'); rect(8, 11, 6, 1, '#73b0d2'); break;
      case FW.DOOR:
      case FW.DOOR_TOP:
      case FW.OPEN_DOOR:
      case FW.OPEN_DOOR_TOP: {
        const open = [FW.OPEN_DOOR, FW.OPEN_DOOR_TOP].includes(v);
        const left = open ? 1 : 2, width = open ? 4 : 12;
        rect(left, 0, width, 16, '#593d29'); rect(left + 1, 0, Math.max(1, width - 2), 16, '#b17d44');
        if (!open) {
          rect(4, 2, 8, 11, '#714d2d'); rect(5, 3, 6, 9, '#a36c37');
          if (v === FW.DOOR_TOP) { rect(5, 3, 6, 5, '#8ac3c0'); rect(7, 3, 1, 5, '#c9a063'); }
          else { rect(11, 5, 2, 2, '#ffe3a0'); rect(3, 14, 10, 1, '#d6a25e'); }
        } break;
      }
      case FW.LADDER:
        rect(3, 0, 2, 16, '#ae8348'); rect(11, 0, 2, 16, '#ae8348');
        for (let y = 1; y < 16; y += 4) { rect(3, y, 10, 2, '#d9ad65'); rect(3, y + 2, 10, 1, '#6f502f'); } break;
      case FW.TORCH:
        rect(7, 7, 3, 9, '#845131'); rect(8, 7, 1, 9, '#d4a16b');
        rect(6, 3, 5, 5, '#e27b31'); rect(7, 1, 3, 6, '#ffbd51'); rect(8, 2, 1, 4, '#fff1a4'); break;
      case FW.FLOWER:
        rect(7, 7, 2, 9, '#4d873d'); rect(4, 10, 4, 2, '#69a745'); rect(9, 12, 3, 2, '#69a745');
        rect(5, 3, 6, 5, variant % 2 ? '#e99393' : '#f4d174'); rect(4, 4, 8, 3, variant % 2 ? '#f4aeac' : '#ffe19a'); rect(7, 4, 2, 2, '#fff5ca'); break;
      default: fill(FW.colors[v] ?? '#d9ebf1');
    }
    if (![FW.LEAF, FW.FLOWER, FW.TORCH, FW.LADDER, FW.OPEN_DOOR, FW.OPEN_DOOR_TOP].includes(v)) bevel();
    textures.set(key, canvas); return canvas;
  }
  function icon(canvas, v) {
    const c = canvas.getContext('2d'); c.imageSmoothingEnabled = false; c.clearRect(0, 0, canvas.width, canvas.height);
    c.drawImage(texture(v), 0, 0, canvas.width, canvas.height);
  }
  function background(c, view) {
    const { camera, time, clock, size } = view, width = c.canvas.width, height = c.canvas.height;
    const light = Math.max(.12, Math.sin(clock * Math.PI * 2));
    const sky = c.createLinearGradient(0, 0, 0, height);
    if (light > .35) { sky.addColorStop(0, '#71b7c5'); sky.addColorStop(.65, '#b5dcd5'); sky.addColorStop(1, '#e4e8be'); }
    else { sky.addColorStop(0, '#293b5f'); sky.addColorStop(.65, '#657b91'); sky.addColorStop(1, '#b69a86'); }
    c.fillStyle = sky; c.fillRect(0, 0, width, height);
    c.fillStyle = light > .35 ? '#fff1b4' : '#edeedb';
    const sx = width * .78, sy = height * .19 + Math.cos(clock * Math.PI * 2) * 20;
    c.fillRect(sx - 21, sy - 21, 42, 42); c.fillStyle = '#fff4cb55'; c.fillRect(sx - 30, sy - 30, 60, 60);
    if (light < .35) for (let i = 0; i < 35; i++) {
      c.fillStyle = `rgba(255,248,226,${.35 + .2 * Math.sin(time * .001 + i)})`;
      c.fillRect((i * 171) % width, (i * 79) % (height * .48), 2, 2);
    }
    for (let i = 0; i < 5; i++) {
      const x = ((i * 321 + time * .008 - camera.x * size * .16) % (width + 200)) - 100;
      const y = 54 + (i % 3) * 54;
      c.fillStyle = '#f4faf0c0'; c.fillRect(x, y + 12, 120, 17); c.fillRect(x + 24, y, 55, 16);
      c.fillStyle = '#e3f0e0a0'; c.fillRect(x + 8, y + 28, 100, 5);
    }
    for (let layer = 0; layer < 3; layer++) {
      c.fillStyle = ['#7eada4', '#63998b', '#467b65'][layer]; c.beginPath(); c.moveTo(0, height);
      for (let x = -8; x <= width + 8; x += 8) {
        const wx = x + camera.x * size * (.06 + layer * .08);
        const y = height * (.51 + layer * .095) + Math.sin(wx / (160 - layer * 32)) * (35 + layer * 8) + Math.sin(wx / 63) * 14;
        c.lineTo(x, Math.floor(y / 8) * 8);
      }
      c.lineTo(width, height); c.closePath(); c.fill();
    }
    c.fillStyle = '#37624966';
    for (let i = 0; i < 30; i++) {
      const x = ((i * 79 - camera.x * size * .25) % (width + 80) + width + 80) % (width + 80);
      const y = height * .77 + Math.sin(i) * 15;
      c.fillRect(x + 9, y - 28, 5, 45); c.fillRect(x, y - 49, 25, 26); c.fillRect(x - 4, y - 35, 33, 20);
    }
  }
  function character(c, p, camera, size, time) {
    const x = (p.x - camera.x) * size, y = (p.y - camera.y) * size;
    c.save(); c.translate(Math.round(x), Math.round(y)); c.scale(size / 16 * p.facing, size / 16);
    const r = (x, y, w, h, color) => { c.fillStyle = color; c.fillRect(x, y, w, h); };
    const gait = p.grounded && Math.abs(p.vx) > .3 ? Math.sin(time * .014) * 2 : 0;
    r(-4, 24 + gait, 3, 5, '#3c5269'); r(1, 24 - gait, 3, 5, '#344457');
    r(-5, 28 + gait, 5, 2, '#493b32'); r(0, 28 - gait, 5, 2, '#493b32');
    r(-5, 12, 10, 13, '#397f7f'); r(-4, 13, 3, 10, '#54a698'); r(2, 15, 3, 11, '#286764');
    r(-7, 13 + gait, 3, 9, '#e2b083'); r(4, 13 - gait, 3, 9, '#dba879');
    r(-5, 3, 10, 9, '#e8bb8b'); r(-5, 0, 10, 5, '#65442e'); r(-5, 3, 2, 7, '#785438');
    r(-6, -1, 12, 3, '#cb9c58'); r(-8, 2, 16, 2, '#e8c884');
    r(2, 6, 2, 2, '#344b4a'); r(1, 10, 3, 1, '#ad704c');
    if (p.flying) { r(-7, 13, 2, 10, '#e9cd8b'); r(-9, 16, 2, 6, '#e9cd8b'); }
    c.restore();
  }
  function draw(c, g, view) {
    const { back, camera, size, time, player, ghost, particles, moves, progress, gridLines } = view;
    c.imageSmoothingEnabled = false;
    background(c, view);
    const w = g[0].length, h = g.length, sx = Math.max(0, Math.floor(camera.x)), sy = Math.max(0, Math.floor(camera.y));
    const ex = Math.min(w, Math.ceil(camera.x + c.canvas.width / size) + 1), ey = Math.min(h, Math.ceil(camera.y + c.canvas.height / size) + 1);
    const tile = (v, x, y, alpha = 1) => {
      c.globalAlpha = alpha;
      c.drawImage(texture(v, (Math.floor(x) * 17 + Math.floor(y) * 3) % 5), Math.round((x - camera.x) * size), Math.round((y - camera.y) * size), Math.ceil(size), Math.ceil(size));
      c.globalAlpha = 1;
    };
    for (let y = sy; y < ey; y++) for (let x = sx; x < ex; x++) if (back[y][x]) tile(back[y][x], x, y);
    if (ghost) for (const a of ghost) {
      const actual = (a.layer === 'back' ? back : g)[a.y]?.[a.x];
      if (actual === a.material || (a.material === FW.DOOR && actual === FW.OPEN_DOOR) || (a.material === FW.DOOR_TOP && actual === FW.OPEN_DOOR_TOP)) continue;
      tile(a.material, a.x, a.y, .20 + .04 * Math.sin(time * .003));
      c.strokeStyle = '#fff0b055'; c.lineWidth = 1;
      c.strokeRect(Math.round((a.x - camera.x) * size) + 1, Math.round((a.y - camera.y) * size) + 1, size - 2, size - 2);
    }
    const moving = new Set();
    if (progress < 1) for (const m of moves) { moving.add(`${m.to[0]}:${m.to[1]}`); if (m.swapped) moving.add(`${m.from[0]}:${m.from[1]}`); }
    for (let y = sy; y < ey; y++) for (let x = sx; x < ex; x++) if (g[y][x] && !moving.has(`${x}:${y}`)) {
      tile(g[y][x], x, y);
      if (g[y][x] === FW.WATER && g[y - 1]?.[x] !== FW.WATER) {
        c.fillStyle = '#d5f0ef88'; c.fillRect((x - camera.x) * size + Math.sin(time * .002 + x) * 2, (y - camera.y) * size + 1, size * .6, 2);
      }
    }
    if (progress < 1) for (const m of moves) {
      const ease = progress * progress;
      tile(m.v, m.from[0] + (m.to[0] - m.from[0]) * ease, m.from[1] + (m.to[1] - m.from[1]) * ease);
      if (m.swapped) tile(m.swapped, m.to[0] + (m.from[0] - m.to[0]) * ease, m.to[1] + (m.from[1] - m.to[1]) * ease);
    }
    if (gridLines) {
      c.strokeStyle = '#ffffff1a'; c.lineWidth = 1;
      for (let x = sx; x <= ex; x++) { const px = Math.round((x - camera.x) * size); c.beginPath(); c.moveTo(px, 0); c.lineTo(px, c.canvas.height); c.stroke(); }
      for (let y = sy; y <= ey; y++) { const py = Math.round((y - camera.y) * size); c.beginPath(); c.moveTo(0, py); c.lineTo(c.canvas.width, py); c.stroke(); }
    }
    character(c, player, camera, size, time);
    for (const sheep of [{ base: 30, phase: 0 }, { base: 54, phase: 2 }]) {
      const x = (sheep.base + Math.sin(time * .00012 + sheep.phase) * 2 - camera.x) * size, y = (FWWorld.SURFACE - camera.y) * size;
      c.fillStyle = '#514d42'; c.fillRect(x + 3, y - 8, 3, 8); c.fillRect(x + 20, y - 8, 3, 8);
      c.fillStyle = '#efedd4'; c.fillRect(x, y - 22, 25, 16); c.fillStyle = '#d6d9bf'; c.fillRect(x + 21, y - 20, 9, 10);
      c.fillStyle = '#3d4841'; c.fillRect(x + 27, y - 18, 2, 2);
    }
    for (const p of particles) {
      c.globalAlpha = Math.max(0, 1 - (time - p.start) / p.life); c.fillStyle = p.color;
      c.fillRect((p.x - camera.x) * size, (p.y - camera.y) * size, p.size, p.size); c.globalAlpha = 1;
    }
    const night = Math.max(0, .45 - Math.sin(view.clock * Math.PI * 2));
    if (night > 0) { c.fillStyle = `rgba(18,32,65,${night * .5})`; c.fillRect(0, 0, c.canvas.width, c.canvas.height); }
    for (let y = sy; y < ey; y++) for (let x = sx; x < ex; x++) if (back[y][x] === FW.TORCH) {
      const px = (x + .5 - camera.x) * size, py = (y + .25 - camera.y) * size;
      const radius = size * (2.6 + Math.sin(time * .007 + x) * .13), glow = c.createRadialGradient(px, py, 0, px, py, radius);
      glow.addColorStop(0, '#ffcb6d65'); glow.addColorStop(1, '#ffcb6d00'); c.fillStyle = glow; c.fillRect(px - radius, py - radius, radius * 2, radius * 2);
      c.fillStyle = time % 170 < 85 ? '#ffec91' : '#ffbb52'; c.fillRect(px - 2, py - 4 - Math.sin(time * .013) * 2, 4, 6);
    }
    if (view.hover && FW.inside(g, view.hover.x, view.hover.y)) {
      const { x, y } = view.hover;
      const px = Math.round((x - camera.x) * size), py = Math.round((y - camera.y) * size);
      if (view.brush && view.tool === 'build' && !g[y][x]) tile(view.brush.id, x, y, .55);
      c.strokeStyle = view.tool === 'erase' ? '#f3a494' : '#fff6cb'; c.lineWidth = 2; c.strokeRect(px + 1, py + 1, size - 2, size - 2);
    }
    c.font = 'bold 10px monospace';
    for (const [x, text] of [[15, 'STARTER HOUSE'], [34, 'YOUR BUILD SITE']]) {
      const px = (x - camera.x) * size, py = (18 - camera.y) * size;
      if (px < -200 || px > c.canvas.width || py < 20) continue;
      c.fillStyle = '#21392bbd'; c.fillRect(px, py - 14, text.length * 6.3 + 14, 20);
      c.fillStyle = '#fff3c8'; c.fillText(text, px + 7, py);
    }
  }
  function minimap(c, g, camera, size, player) {
    const w = g[0].length, h = g.length, sx = c.canvas.width / w, sy = c.canvas.height / h;
    c.fillStyle = '#a7ccd0'; c.fillRect(0, 0, c.canvas.width, c.canvas.height);
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) if (g[y][x]) {
      c.fillStyle = FW.colors[g[y][x]]; c.fillRect(x * sx, y * sy, Math.ceil(sx), Math.ceil(sy));
    }
    c.strokeStyle = '#fff6ca'; c.lineWidth = 1.5; c.strokeRect(camera.x * sx, camera.y * sy, PW / size * sx, PH / size * sy);
    c.fillStyle = '#fff9ef'; c.fillRect(player.x * sx - 2, player.y * sy - 2, 4, 4);
  }
  function drawExact(c, g) {
    const size = 4; c.canvas.width = g[0].length * size; c.canvas.height = g.length * size;
    for (let y = 0; y < g.length; y++) for (let x = 0; x < g[0].length; x++) {
      c.fillStyle = FW.colors[g[y][x]]; c.fillRect(x * size, y * size, size, size);
    }
  }
  return { S, PW, PH, BASE, EMPTY_RGB, texture, icon, draw, minimap, drawExact };
})();
