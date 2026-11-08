import json
from pathlib import Path
import shutil
import subprocess
import unittest

import numpy as np

from core import (Action, FormWorld, EMPTY, GROUND, WOOD, LEAF, SAND, WATER,
                  BRICK, GLASS, DOOR, DOOR_TOP, OPEN_DOOR, OPEN_DOOR_TOP, GRASS,
                  WALL, TORCH, LADDER, FLOWER, LOAD, STRUCT, zeros, check_violations)
from render import render, parse

ROOT = Path(__file__).parent


class SandboxPhysicsTests(unittest.TestCase):
    def test_background_does_not_support_structure(self):
        w = FormWorld()
        self.assertTrue(w.apply_action(Action('place', 8, 20, WALL, 'back')))
        w.step(Action('place', 8, 19, WOOD))
        self.assertEqual(w.grid[20, 8], WOOD)
        self.assertEqual(w.background[20, 8], WALL)

    def test_window_replaces_wall_without_front_collision(self):
        w = FormWorld()
        w.apply_action(Action('place', 8, 20, WALL, 'back'))
        self.assertTrue(w.apply_action(Action('place', 8, 20, GLASS, 'back')))
        self.assertEqual(w.background[20, 8], GLASS)
        self.assertEqual(w.grid[20, 8], EMPTY)

    def test_background_items_cannot_float(self):
        w = FormWorld()
        for m in (WALL, GLASS, TORCH, LADDER, FLOWER):
            self.assertFalse(w.apply_action(Action('place', 10, 8, m, 'back')), m)
        self.assertEqual(int(w.background.sum()), 0)
        # A flower needs grass or dirt below, not just any block.
        self.assertFalse(w.apply_action(Action('place', 10, 20, FLOWER, 'back')))
        w.grid[20, 10] = GRASS
        w.grid[19, 10] = EMPTY
        self.assertTrue(w.apply_action(Action('place', 10, 19, FLOWER, 'back')))

    def test_background_chains_attach_and_fall_away_with_their_anchor(self):
        w = FormWorld()
        w.grid[20, 5] = BRICK
        for y in range(19, 15, -1):
            self.assertTrue(w.apply_action(Action('place', 5, y, WALL, 'back')), y)
        self.assertTrue(w.apply_action(Action('place', 6, 17, TORCH, 'back')))
        self.assertTrue(w.apply_action(Action('place', 4, 15, LADDER, 'back')) is False)
        w.apply_action(Action('place', 5, 15, LADDER, 'back'))
        self.assertTrue(w.apply_action(Action('place', 5, 14, LADDER, 'back')))
        # Break the only anchor and the whole chain disappears with it.
        self.assertTrue(w.apply_action(Action('remove', 5, 20)))
        self.assertEqual(int(np.count_nonzero(w.background)), 0)

    def test_erasing_a_middle_wall_keeps_anchored_walls(self):
        w = FormWorld()
        w.grid[20, 5] = BRICK
        w.grid[20, 9] = BRICK
        for x in range(5, 10):
            w.apply_action(Action('place', x, 19, WALL, 'back'))
        w.apply_action(Action('remove', 7, 19, layer='back'))
        self.assertEqual(int(np.count_nonzero(w.background)), 4)

    def test_doors_need_two_cells_and_floor(self):
        w = FormWorld()
        self.assertFalse(w.apply_action(Action('place', 5, 8, DOOR)))
        w.grid[19, 5] = WOOD
        self.assertFalse(w.apply_action(Action('place', 5, 20, DOOR)))
        w.grid[19, 5] = EMPTY
        self.assertTrue(w.apply_action(Action('place', 5, 20, DOOR)))
        self.assertEqual((w.grid[20, 5], w.grid[19, 5]), (DOOR, DOOR_TOP))
        self.assertTrue(w.apply_action(Action('interact', 5, 19)))
        self.assertEqual((w.grid[20, 5], w.grid[19, 5]), (OPEN_DOOR, OPEN_DOOR_TOP))
        w.apply_action(Action('remove', 5, 19))
        self.assertEqual((w.grid[20, 5], w.grid[19, 5]), (EMPTY, EMPTY))

    def test_doorframe_still_supports_when_open(self):
        w = FormWorld()
        w.apply_action(Action('place', 5, 20, DOOR))
        w.step(Action('place', 5, 18, BRICK))
        self.assertTrue(w.supported()[18, 5])
        w.step(Action('interact', 5, 20))
        self.assertEqual(w.grid[18, 5], BRICK)
        self.assertTrue(w.supported()[18, 5])

    def test_entire_door_falls_when_floor_removed(self):
        w = FormWorld()
        w.grid[17, 5] = BRICK
        w.apply_action(Action('place', 5, 16, DOOR))
        w.grid[17, 5] = EMPTY
        w.step()
        self.assertEqual((w.grid[20, 5], w.grid[19, 5]), (DOOR, DOOR_TOP))
        self.assertEqual(np.count_nonzero(w.grid == DOOR_TOP), 1)

    def test_crown_connects_through_leaves_not_direct_wood_only(self):
        w = FormWorld()
        w.grid[20, 7] = WOOD
        w.grid[19, 7] = LEAF
        w.grid[19, 8] = LEAF
        w.step()
        self.assertEqual(w.grid[19, 8], LEAF)
        self.assertFalse(LEAF in LOAD)

    def test_mixed_mass_conserved_and_water_reaches_rest(self):
        w = FormWorld(72)
        r = np.random.default_rng(5)
        for _ in range(180):
            x, y = int(r.integers(0, 28)), int(r.integers(0, 20))
            if not w.grid[y, x]:
                w.grid[y, x] = int(r.choice([WOOD, LEAF, SAND, WATER, BRICK]))
        before = np.bincount(w.grid.ravel(), minlength=21)
        for _ in range(350):
            changed, _ = w.tick()
            if not changed:
                break
        self.assertFalse(changed, 'Physics should settle, not keep water oscillating')
        np.testing.assert_array_equal(before, np.bincount(w.grid.ravel(), minlength=21))
        self.assertEqual(sum(check_violations(w.grid).values()), 0)

    def test_exact_palette_roundtrip_all_materials_and_custom_sizes(self):
        grid = np.arange(40 * 96, dtype=np.uint16).reshape(40, 96) % 21
        np.testing.assert_array_equal(parse(render(grid, 4), cell=4), grid)

    @unittest.skipUnless(shutil.which('node'), 'Node needed for browser/Python parity')
    def test_javascript_python_tick_and_action_parity(self):
        actions = []
        r = np.random.default_rng(32)
        for _ in range(110):
            actions.append(Action('place', int(r.integers(0, 28)), int(r.integers(0, 21)),
                                  int(r.choice([WOOD, LEAF, SAND, WATER, BRICK]))).as_dict())
        actions.extend([Action('place', 3, 20, DOOR).as_dict(), Action('interact', 3, 20).as_dict()])
        w = FormWorld(42)
        expected = []
        for a in actions:
            w.apply_action(Action(**a))
            for _ in range(3):
                w.tick()
                expected.append(w.grid.tolist())
        source = """
const FW = require('./physics.js');
const actions = JSON.parse(require('fs').readFileSync(0, 'utf8'));
const grid = FW.zeros(), r = FW.rng(42), states = [];
for (const a of actions) {
  FW.applyAction(grid, a);
  for (let t = 0; t < 3; t++) { FW.tick(grid, r); states.push(FW.clone(grid)); }
}
process.stdout.write(JSON.stringify(states));
"""
        actual = json.loads(subprocess.check_output(['node', '-e', source], cwd=ROOT, input=json.dumps(actions).encode()))
        self.assertEqual(actual, expected)

    @unittest.skipUnless(shutil.which('node'), 'Node needed for browser/Python parity')
    def test_javascript_python_background_rules_parity(self):
        r = np.random.default_rng(9)
        actions = []
        for _ in range(500):
            kind = int(r.integers(0, 4))
            x, y = int(r.integers(0, 28)), int(r.integers(14, 21))
            if kind == 0:
                actions.append(Action('place', x, y, int(r.choice([BRICK, WOOD, GRASS]))).as_dict())
            elif kind == 1:
                actions.append(Action('remove', x, y).as_dict())
            elif kind == 2:
                actions.append(Action('place', x, y, int(r.choice([WALL, GLASS, TORCH, LADDER, FLOWER])), 'back').as_dict())
            else:
                actions.append(Action('remove', x, y, layer='back').as_dict())
        w = FormWorld(3)
        results = []
        for a in actions:
            results.append(bool(w.apply_action(Action(**a))))
            w.step()
        source = """
const FW = require('./physics.js');
const actions = JSON.parse(require('fs').readFileSync(0, 'utf8'));
const g = FW.zeros(), back = FW.blank(), r = FW.rng(3), results = [];
for (const a of actions) { results.push(FW.applyAction(g, a, back)); FW.step(g, null, r, back); }
process.stdout.write(JSON.stringify({ results, g, back }));
"""
        actual = json.loads(subprocess.check_output(['node', '-e', source], cwd=ROOT, input=json.dumps(actions).encode()))
        self.assertEqual(actual['results'], results)
        self.assertEqual(actual['back'], w.background.tolist())
        self.assertEqual(actual['g'], w.grid.tolist())
        self.assertGreater(int(np.count_nonzero(w.background)), 0)

    @unittest.skipUnless(shutil.which('node'), 'Node needed for blueprint tests')
    def test_both_building_templates_stay_complete_with_gravity(self):
        source = """
const FW = require('./physics.js'); global.FW = FW;
const World = require('./world.js');
const results = [];
for (const name of ['Cottage', 'Workshop']) {
  const state = World.create(); const plan = World.plan(name); const r = FW.rng(21);
  for (const a of plan.actions) { FW.applyAction(state.g, a, state.back); FW.tick(state.g, r); }
  for (let t = 0; t < 100; t++) FW.tick(state.g, r);
  const ok = FW.supported(state.g); const bad = state.g.flatMap((row, y) => row.map((v, x) => FW.STRUCT.has(v) && !ok.has(y * row.length + x) ? 1 : 0)).reduce((a, b) => a + b, 0);
  results.push({ name, percent: World.progress(state.g, state.back, World.targetCells(plan)).percent, bad });
}
process.stdout.write(JSON.stringify(results));
"""
        actual = json.loads(subprocess.check_output(['node', '-e', source], cwd=ROOT))
        for result in actual:
            self.assertEqual(result['percent'], 100, result)
            self.assertEqual(result['bad'], 0, result)


class DatasetTests(unittest.TestCase):
    def test_all_scene_builders_apply_cleanly_and_rest_without_violations(self):
        import random
        import datasets as ds
        for kind in ds.TRAIN_KINDS:
            for scale in ('base', 'big'):
                if scale == 'big' and kind in ('hut', 'tree', 'dune', 'garden', 'pond'):
                    continue
                for s in range(3):
                    scene = ds.sample_scene(random.Random(f'{kind}{scale}{s}'), [kind], scale)
                    w = ds.settled_build(scene, 1)
                    self.assertEqual(sum(check_violations(w.grid, w.background).values()), 0, (kind, scale))
                    if kind not in ('dune', 'pond'):   # fluids legitimately move away from where they were placed
                        ideal = ds.new_world(1)
                        for a in scene['actions']:
                            self.assertTrue(ideal.apply_action(a), (kind, scale, a))
                        np.testing.assert_array_equal(ideal.grid, w.grid, (kind, scale))
                        np.testing.assert_array_equal(ideal.background, w.background, (kind, scale))

    @unittest.skipUnless(shutil.which('node'), 'Node needed for blueprint parity')
    def test_python_house_blueprint_matches_game_blueprint_exactly(self):
        import datasets as ds
        cases = {
            'Cottage': dict(kind='cottage', width=10, tall=5, beam=WOOD, windows=[2, 3, 7], ladder=None, door_left=True),
            'Workshop': dict(kind='cottage', width=13, tall=7, beam=BRICK, windows=[2, 3, 10], ladder=7, door_left=True),
        }
        for name, spec in cases.items():
            source = f"""
const FW = require('./physics.js'); global.FW = FW;
const World = require('./world.js');
process.stdout.write(JSON.stringify(World.plan('{name}', 9, {ds.FLOOR}).actions.map(a => [a.x, a.y, a.material, a.layer, a.type])));
"""
            js = json.loads(subprocess.check_output(['node', '-e', source], cwd=ROOT))
            py = [[a.x, a.y, a.material, 'back' if a.layer == 'back' else 'front', a.type] for a in ds.house_actions(spec, 9)]
            self.assertEqual(py, js, name)

    def test_training_conditions_share_scenes_and_budgets(self):
        import datasets as ds
        res = ds.make_episode_set(0, 0)
        eps = res['episodes']
        self.assertEqual({c: eps[c]['fg'].shape for c in ds.CONDITIONS}, {c: (ds.T_DEFAULT, ds.DH, ds.DW) for c in ds.CONDITIONS})
        # process: ordered continuous formation; shuffle: same frames, permuted, no valid transitions
        self.assertTrue(eps['process']['cont'][1:ds.F_FORM].all() and not eps['process']['cont'][0])
        self.assertFalse(eps['shuffle']['cont'][:ds.F_FORM].any())
        proc = sorted(ds.state_hash(f, b) for f, b in zip(eps['process']['fg'][:ds.F_FORM], eps['process']['bg'][:ds.F_FORM]))
        shuf = sorted(ds.state_hash(f, b) for f, b in zip(eps['shuffle']['fg'][:ds.F_FORM], eps['shuffle']['bg'][:ds.F_FORM]))
        self.assertEqual(proc, shuf)
        self.assertFalse(np.array_equal(eps['process']['fg'][:ds.F_FORM], eps['shuffle']['fg'][:ds.F_FORM]))
        # same interventions on the finished structure in state and process
        self.assertEqual(eps['state']['actions'][:4 * ds.CLIP], eps['process']['actions'][ds.F_FORM:])
        for c in ('state', 'partial'):
            self.assertTrue((eps[c]['phase'] == 1).all())
        # the formation starts from bare terrain and ends at the finished structure
        np.testing.assert_array_equal(eps['process']['fg'][0], ds.terrain())
        self.assertTrue(res['meta']['formation_settled'])
        np.testing.assert_array_equal(eps['process']['fg'][ds.F_FORM - 1], res['final'][0])

    def test_generation_is_reproducible(self):
        import datasets as ds
        a, b = ds.make_episode_set(5, 3), ds.make_episode_set(5, 3)
        for c in ds.CONDITIONS:
            np.testing.assert_array_equal(a['episodes'][c]['fg'], b['episodes'][c]['fg'])
            np.testing.assert_array_equal(a['episodes'][c]['bg'], b['episodes'][c]['bg'])
        self.assertEqual(a['meta']['final_hash'], b['meta']['final_hash'])

    def test_bench_item_has_valid_ground_truth(self):
        import datasets as ds
        item = None
        for attempt in range(20):
            item = ds.sample_bench_item('scale_ood', 0, 0, attempt, set())
            if item:
                break
        self.assertIsNotNone(item)
        self.assertEqual(sum(check_violations(item['fg'], item['bg']).values()), 0)
        for run in item['pair']:
            self.assertEqual(run['violations'], 0)
            self.assertLess(run['settled_at'], ds.HORIZON)
            self.assertEqual(run['fg'].shape, (ds.HORIZON, ds.DH, ds.DW))
        self.assertGreaterEqual(item['pair'][0]['label'], item['pair'][1]['label'])

    def test_layered_render_round_trip_on_stacks(self):
        import datasets as ds
        from render import render_layers, compose
        res = ds.make_episode_set(0, 1)
        fg, bg = res['episodes']['process']['fg'], res['episodes']['process']['bg']
        frames = render_layers(fg, bg, ds.CELL)
        self.assertEqual(frames.shape, (fg.shape[0], ds.DH * ds.CELL, ds.DW * ds.CELL, 3))
        np.testing.assert_array_equal(parse(frames[-1], cell=ds.CELL), compose(fg[-1], bg[-1]))


class DatasetV3Tests(unittest.TestCase):
    def test_new_blueprints_apply_cleanly(self):
        import random
        import datasets as ds
        import datasets_v3 as v3
        for kind in ('multistory', 'colonnade', 'aqueduct', 'fortress', 'stairs', 'granary'):
            for scale in ('base', 'big') if kind != 'granary' else ('base',):
                for s in range(3):
                    spec = v3.sample_spec(kind, random.Random(s), scale)
                    w = ds.new_world(1)
                    for act, _tag in v3.build_items(spec, 10):
                        self.assertTrue(w.apply_action(act), (kind, scale, s, act))
                        w.tick()
                        w.prune_background()
                    for _ in range(300):
                        c, _m = w.tick()
                        r = w.prune_background()
                        if not c and not r:
                            break
                    self.assertEqual(sum(check_violations(w.grid, w.background).values()), 0, (kind, scale, s))

    def test_v3_episodes_share_budgets_and_reproduce(self):
        import datasets_v3 as v3
        a = v3.make_episode_set(3, 1)
        b = v3.make_episode_set(3, 1)
        for c, ep in a['episodes'].items():
            self.assertEqual(ep['fg'].shape[0], v3.T_DEFAULT)
            np.testing.assert_array_equal(ep['fg'], b['episodes'][c]['fg'])
        self.assertTrue(a['meta']['formation_settled'])

    def test_simple_training_and_hard_evaluation_profile(self):
        import datasets as base
        import datasets_v3 as v3
        easy = v3.make_episode_set(0, 0, p_mistake=0, scaffold=False, profile='simple_to_complex')
        self.assertLessEqual(easy['meta']['n_struct'], 2)
        self.assertTrue(set(easy['meta']['kinds']).issubset(base.TRAIN_KINDS))
        self.assertEqual(set(easy['meta']['event_kinds']), {'single'})
        for c in ('state', 'process'):
            self.assertEqual(easy['episodes'][c]['fg'].shape[0], 64)
        for split in v3.BENCH_SPLITS:
            item = v3._bench_worker((split, 2027, 0, set(), 'simple_to_complex'))
            cascades = sorted((r['cascade'] for r in item['pair']))
            self.assertGreaterEqual(cascades[0], 2)
            self.assertGreaterEqual(cascades[1], 6)
            self.assertTrue(item['rank_valid'])
            self.assertGreaterEqual(max(r['depth'] for r in item['pair']), 3)

    def test_v3_bench_item_is_hard_and_valid(self):
        import datasets_v3 as v3
        item = None
        for attempt in range(30):
            item = v3.sample_bench_item('multi_ood', 0, 0, attempt, set())
            if item:
                break
        self.assertIsNotNone(item)
        self.assertEqual(sum(check_violations(item['fg'], item['bg']).values()), 0)
        for run in item['pair']:
            self.assertEqual(run['violations'], 0)


if __name__ == '__main__':
    unittest.main()
