const FWWorld = (() => {
  const WIDTH = 96, HEIGHT = 40, SURFACE = 30;
  function plan(name = 'Cottage', x0 = 34, floor = SURFACE - 1) {
    const width = name === 'Workshop' ? 13 : 10;
    const tall = name === 'Workshop' ? 7 : 5;
    const top = floor - tall, front = [], back = [];
    const put = (x, y, material, group, layer = 'front') => (layer === 'front' ? front : back).push({
      type: 'place', x, y, material, layer, group,
    });
    for (let x = x0; x < x0 + width; x++) put(x, floor, FW.STONE, 'Foundation');
    for (let y = floor - 1; y > top; y--) {
      if (y < floor - 2) put(x0, y, FW.BRICK, 'Frame');
      put(x0 + width - 1, y, FW.BRICK, 'Frame');
      for (let x = x0 + 1; x < x0 + width - 1; x++) put(x, y, FW.WALL, 'Walls', 'back');
    }
    for (let x = x0; x < x0 + width; x++) put(x, top, name === 'Workshop' ? FW.BRICK : FW.WOOD, 'Frame');
    for (let level = 0; level < Math.floor(width / 2); level++) {
      const left = x0 + level, right = x0 + width - 1 - level, y = top - 1 - level;
      for (let x = left; x <= right; x++) put(x, y, FW.ROOF, 'Roof');
    }
    for (const x of [x0 + 2, x0 + 3, x0 + width - 3]) {
      put(x, floor - 2, FW.GLASS, 'Windows', 'back');
      put(x, floor - 3, FW.GLASS, 'Windows', 'back');
    }
    put(x0, floor - 1, FW.DOOR, 'Door');
    put(x0 + 1, floor - 2, FW.TORCH, 'Details', 'back');
    put(x0 + width - 2, floor - 2, FW.TORCH, 'Details', 'back');
    if (name === 'Workshop') for (let y = floor - 1; y > top; y--) put(x0 + 7, y, FW.LADDER, 'Details', 'back');
    const edgeDistance = x => Math.min(x - x0, x0 + width - 1 - x);
    const foreground = front.filter(a => a.material !== FW.DOOR).sort((a, b) =>
      b.y - a.y || edgeDistance(a.x) - edgeDistance(b.x) || a.x - b.x);
    const door = front.filter(a => a.material === FW.DOOR);
    const walls = back.filter(a => a.material === FW.WALL);
    const details = back.filter(a => a.material !== FW.WALL);
    return { name, x0, floor, width, top: top - Math.floor(width / 2),
      actions: [...foreground.filter(a => a.group === 'Foundation'), ...door,
        ...foreground.filter(a => a.group !== 'Foundation'), ...walls, ...details] };
  }
  function targetCells(blueprint) {
    const cells = new Map();
    for (const a of blueprint.actions) {
      cells.set(`${a.layer}:${a.x}:${a.y}`, { ...a });
      if (a.material === FW.DOOR) cells.set(`front:${a.x}:${a.y - 1}`, { ...a, y: a.y - 1, material: FW.DOOR_TOP });
    }
    return [...cells.values()];
  }
  function tree(g, x, ground, height = 6) {
    for (let y = ground - height; y < ground; y++) g[y][x] = FW.WOOD;
    const crown = ground - height;
    for (let dy = -2; dy <= 2; dy++) for (let dx = -3; dx <= 3; dx++) {
      if (Math.abs(dx) + Math.abs(dy) <= 4 && FW.inside(g, x + dx, crown + dy) && !g[crown + dy][x + dx]) g[crown + dy][x + dx] = FW.LEAF;
    }
  }
  function create(seed = 20261103) {
    const g = FW.zeros(WIDTH, HEIGHT), back = FW.blank(WIDTH, HEIGHT), r = FW.rng(seed);
    for (let x = 0; x < WIDTH; x++) {
      const ground = x > 69 ? SURFACE - Math.round(2 * Math.sin((x - 69) / 9)) : SURFACE;
      for (let y = ground; y < HEIGHT - 1; y++) g[y][x] = y === ground ? FW.GRASS : y > HEIGHT - 5 ? FW.STONE : FW.DIRT;
    }
    for (const [x, height] of [[5, 7], [9, 5], [50, 7], [61, 6], [76, 8], [89, 6]]) {
      const y = g.findIndex(row => row[x] === FW.GRASS); tree(g, x, y, height);
    }
    for (const x of [2, 12, 26, 28, 31, 46, 54, 58, 67, 80, 84]) {
      const ground = g.findIndex(row => row[x] === FW.GRASS); if (!g[ground - 1][x]) back[ground - 1][x] = FW.FLOWER;
    }
    const starter = plan('Cottage', 15);
    for (const a of starter.actions) FW.applyAction(g, a, back);
    for (let y = 25; y < 30; y++) back[y][23] = y === 27 ? FW.TORCH : FW.LADDER;
    for (let x = 64; x < 69; x++) {
      g[30][x] = FW.EMPTY; g[31][x] = FW.SAND; g[30][x] = FW.WATER;
    }
    for (let i = 0; i < 10; i++) FW.tick(g, r);
    FW.prune(g, back);
    return { version: 3, seed, g, back, player: { x: 28.5, y: SURFACE - 1.8, vx: 0, vy: 0, facing: 1, grounded: false, flying: false },
      camera: { x: 10, y: 10 }, clock: .28, day: 1, builds: 0 };
  }
  function progress(g, back, cells) {
    const groups = {}, total = cells.length;
    let complete = 0;
    for (const cell of cells) {
      const actual = (cell.layer === 'back' ? back : g)[cell.y]?.[cell.x];
      const matches = actual === cell.material || (cell.material === FW.DOOR && actual === FW.OPEN_DOOR)
        || (cell.material === FW.DOOR_TOP && actual === FW.OPEN_DOOR_TOP);
      groups[cell.group] ??= { done: 0, total: 0 }; groups[cell.group].total++;
      if (matches) { complete++; groups[cell.group].done++; }
    }
    return { complete, total, percent: Math.round(complete / total * 100), groups };
  }
  return { WIDTH, HEIGHT, SURFACE, create, plan, targetCells, progress };
})();
if (typeof module !== 'undefined') module.exports = FWWorld;
