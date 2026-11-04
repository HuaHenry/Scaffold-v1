async page => {
  await page.goto('http://127.0.0.1:8765/');
  await page.setViewportSize({ width: 1500, height: 1000 });
  const check = async (fn, message) => { if (!await page.evaluate(fn)) throw new Error(message); };
  const point = async (x, y) => page.evaluate(({ x, y }) => {
    const c = document.getElementById('world'), r = c.getBoundingClientRect(), s = FormGame.state, tile = FormGame.pack().size;
    return { x: r.left + (x + .5 - s.camera.x) * tile / c.width * r.width,
      y: r.top + (y + .5 - s.camera.y) * tile / c.height * r.height };
  }, { x, y });
  const click = async (x, y, button = 'left', shift = false) => {
    const p = await point(x, y);
    if (shift) await page.keyboard.down('Shift');
    await page.mouse.click(p.x, p.y, { button });
    if (shift) await page.keyboard.up('Shift');
  };

  await page.locator('#gravity-toggle').click();
  await page.locator('#record-toggle').click();
  await page.getByRole('button', { name: 'Timber wall', exact: true }).click();
  await click(30, 24);
  await check(() => FormGame.state.back[24][30] === 0, 'A wall was allowed to float in mid-air');
  await page.getByRole('button', { name: 'Window', exact: true }).click();
  await click(30, 24);
  await check(() => FormGame.state.back[24][30] === 0, 'A window was allowed to float in mid-air');
  await page.getByRole('button', { name: 'Timber wall', exact: true }).click();
  await click(30, 29);
  await check(() => FormGame.state.back[29][30] === FW.WALL, 'Wall placement failed');
  await page.getByRole('button', { name: 'Window', exact: true }).click();
  await click(30, 29);
  await check(() => FormGame.state.back[29][30] === FW.GLASS && FormGame.state.g[29][30] === 0, 'Window layer failed');
  await click(30, 29, 'right', true);
  await check(() => FormGame.state.back[29][30] === 0, 'Shift erase failed');
  await page.locator('#undo').click();
  await check(() => FormGame.state.back[29][30] === FW.GLASS, 'Undo failed');
  await page.locator('#redo').click();
  await check(() => FormGame.state.back[29][30] === 0, 'Redo failed');
  await page.locator('#undo').click();
  await page.locator('#save').click();
  await click(30, 29, 'right', true);
  await page.locator('#load').click();
  await check(() => FormGame.state.back[29][30] === FW.GLASS, 'Load failed');
  await check(() => FormGame.records.length >= 3 && FormGame.records[0].before && FormGame.records[0].after, 'Recording failed');

  await page.getByRole('button', { name: 'How to play' }).click();
  if (!await page.locator('#help-dialog').isVisible()) throw new Error('Help failed');
  await page.locator('#start-playing').click();
  await page.locator('#grid-toggle').click();
  await page.locator('#zoom-in').click();
  await page.locator('#day-toggle').click();
  await page.locator('#gravity-toggle').click();
  await page.locator('#demo').click();
  await page.locator('#confirm-yes').click();
  await page.waitForFunction(() => FormGame.progress().percent === 100, null, { timeout: 60000, polling: 100 });
  await check(() => FormGame.validate(FormGame.pack()), 'Invalid world after build');

  await page.locator('#world').focus();
  await page.keyboard.press('e');
  await check(() => FormGame.state.g[28][34] === FW.OPEN_DOOR && FormGame.state.g[27][34] === FW.OPEN_DOOR_TOP, 'Door interaction failed');
  await page.keyboard.down('d');
  await page.waitForFunction(() => FormGame.state.player.x > 35, null, { timeout: 10000, polling: 50 });
  await page.keyboard.up('d');
  await check(() => FormGame.state.player.x > 34.5, 'Player cannot walk through opened doorway');
  await page.waitForFunction(() => FormGame.state.player.grounded, null, { timeout: 10000, polling: 50 });
  await page.keyboard.press('f');
  await check(() => FormGame.state.player.flying, 'Flight did not enable');
  await page.keyboard.press('f');

  await page.locator('#project-select').selectOption('Workshop');
  await page.locator('#demo').click(); await page.locator('#confirm-yes').click();
  await page.waitForFunction(() => FormGame.progress().percent === 100, null, { timeout: 60000, polling: 100 });
  const exportEvent = page.waitForEvent('download');
  await page.locator('#export').click();
  const exported = await exportEvent;
  if (exported.suggestedFilename() !== 'block-and-bloom.json') throw new Error('JSON export failed');
  await exported.saveAs('output/browser-session.json');
  const screenshotEvent = page.waitForEvent('download');
  await page.locator('#screenshot').click();
  const image = await screenshotEvent;
  if (image.suggestedFilename() !== 'willow-meadow.png') throw new Error('Screenshot export failed');
  await image.saveAs('output/game-screenshot.png');
  await page.screenshot({ path: 'output/sandbox-view.png' });
  await page.setViewportSize({ width: 750, height: 1000 });
  await check(() => document.documentElement.scrollWidth <= innerWidth, 'Mobile layout overflows');
  await page.screenshot({ path: 'output/sandbox-mobile.png' });
  await page.setViewportSize({ width: 1500, height: 1000 });
  return { success: true, templates: ['Cottage', 'Workshop'], interactions: ['walls', 'windows', 'erase', 'undo', 'redo', 'save', 'load', 'doors', 'walking', 'flight', 'record', 'export'], final: await page.evaluate(() => FormGame.progress()) };
}
