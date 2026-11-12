"""Check frozen-state failures with temporary files and mocked execution only."""

import json
import io
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
import run_controlled_study as study


class ControlledStudyTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.revision = 'synthetic-committed-revision'
        self.compiler_version = 'synthetic compiler 1.0'
        names = [
            'src/blocked_streaming_scalar.cpp', 'src/oracle_fp64.cpp',
            'src/input.cpp', 'src/metrics.cpp', 'include/attn.h',
            'tools/bench_streaming.cpp', 'tools/build_integrated.py',
            'tools/run_controlled_study.py', 'tools/run_project_grid.py',
            'tools/run_scalar_bench.py', 'tools/run_streaming_experiment.py',
            'tools/runtime_conditions.py',
        ]
        for name in names:
            path = self.root/name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('synthetic source fixture: ' + name, encoding='utf-8')
        self.binary_name = 'build/attn_bench_all' + study.SUFFIX
        self.binary = self.root/self.binary_name
        self.binary.parent.mkdir()
        self.binary.write_bytes(b'synthetic benchmark bytes; never executable')
        self.binding = dict(
            status='passed', binary_sha256=study.sha(self.binary),
            compiler=self.compiler_version,
            source_sha256={name: study.sha(self.root/name) for name in names[:5]},
        )
        self.build_record = self.root/'build/integrated_build.json'
        self.build_record.write_text(json.dumps(self.binding), encoding='utf-8')
        self.expected = study.capture_frozen_state(
            self.root, self.binding, self.revision, self.binary_name)
        self.git = self.enterContext(patch.object(study.subprocess, 'check_output', side_effect=self.git_output))

    def git_output(self, command, **kwargs):
        if command == [study.CXX, '--version']:
            return self.compiler_version + '\n'
        self.assertEqual(command[0], 'git')
        return self.revision + '\n' if command[1] == 'rev-parse' else ''

    def test_unchanged_state_passes(self):
        study.verify_frozen_state(self.root, self.expected)

    def test_output_outside_results_rejected_before_execution(self):
        for out in (self.root/'other_study', self.root/'results/../escaped_study'):
            with (
                self.subTest(out=out),
                patch.object(study, 'ROOT', self.root),
                patch.object(sys, 'argv', ['run_controlled_study.py', '--out', str(out),
                                          '--cpu-order', '0', '--physical-cores', '1']),
                patch.object(sys, 'stderr', io.StringIO()) as errors,
                patch.object(study.subprocess, 'run') as execute,
            ):
                with self.assertRaises(SystemExit) as stopped:
                    study.main()
                self.assertEqual(stopped.exception.code, 2)
                self.assertIn('repository results directory', errors.getvalue())
                self.assertFalse(out.resolve().exists())
                execute.assert_not_called()
                self.git.assert_not_called()

    def test_changed_revision_rejected(self):
        self.revision = 'different-synthetic-revision'
        with self.assertRaisesRegex(ValueError, 'Git revision changed'):
            study.verify_frozen_state(self.root, self.expected)

    def test_dirty_tree_rejected(self):
        self.git.side_effect = lambda command, **kwargs: (
            self.revision if command[1] == 'rev-parse' else ' M tools/run_project_grid.py\n')
        with self.assertRaisesRegex(ValueError, 'working tree is no longer clean'):
            study.verify_frozen_state(self.root, self.expected)

    def test_artifact_and_source_changes_rejected(self):
        cases = [
            (self.binary_name, 'native binary'),
            ('build/integrated_build.json', 'build record'),
            ('src/blocked_streaming_scalar.cpp', 'source'),
            ('tools/run_project_grid.py', 'source'),
        ]
        for name, label in cases:
            with self.subTest(name=name):
                path = self.root/name
                original = path.read_bytes()
                try:
                    path.write_bytes(original + b' changed fixture')
                    with self.assertRaisesRegex(ValueError, 'Frozen ' + label + ' changed'):
                        study.verify_frozen_state(self.root, self.expected)
                finally:
                    path.write_bytes(original)

    def run_mock_study(self, failure=None):
        out = self.root/'results/synthetic_study'
        native_commands = []
        self.tuning_commands = []
        self.compile_calls = 0
        if failure == 'compiler':
            self.compiler_version = 'different synthetic compiler 2.0'

        def fake_run(command, **kwargs):
            if '-o' in command:
                self.compile_calls += 1
                Path(command[command.index('-o') + 1]).write_bytes(b'synthetic tuning binary')
            elif len(command) > 1 and command[1] == 'tune':
                self.tuning_commands.append(command)
                if failure == 'tuning_binary_during':
                    Path(command[0]).write_bytes(b'replaced synthetic tuning binary')
            elif 'tools/run_project_grid.py' in command:
                native_commands.append(command)
                if failure == 'native_binary' and len(native_commands) == 1:
                    self.binary.write_bytes(b'replacement benchmark, also never executable')
                    replacement = dict(self.binding, binary_sha256=study.sha(self.binary))
                    self.build_record.write_text(json.dumps(replacement), encoding='utf-8')
            return SimpleNamespace(returncode=0, stdout='synthetic tuning text', stderr='')

        def fake_affinity(selection):
            if failure == 'tuning_binary_before' and selection == '0':
                (self.root/'build'/('attn_block_tune'+study.SUFFIX)).write_bytes(
                    b'replaced before synthetic tuning call')

        conditions = {'fixture': 'synthetic conditions'}
        condition_results = ([conditions, conditions, {'fixture': 'different conditions'}]
                             if failure == 'conditions' else None)
        with (
            patch.object(study, 'ROOT', self.root),
            patch.object(sys, 'argv', ['run_controlled_study.py', '--out', str(out),
                                      '--cpu-order', '0,1,2,3,4,5,6,7', '--physical-cores', '8']),
            patch.dict(os.environ, {'OMP_WAIT_POLICY': 'PASSIVE'}),
            patch.object(study.subprocess, 'run', side_effect=fake_run),
            patch.object(study, 'apply_cpu_affinity', side_effect=fake_affinity),
            patch.object(study, 'capture_conditions', return_value=conditions,
                         side_effect=condition_results),
            patch.object(study, 'summarize', return_value=[
                dict(key_block=128, median_ms=3.0),
                dict(key_block=256, median_ms=2.0),
                dict(key_block=512, median_ms=4.0),
            ]),
        ):
            if failure:
                expected_error = {
                    'native_binary': 'Frozen native binary changed',
                    'compiler': 'Tuning compiler differs',
                    'conditions': 'runtime conditions changed between batches',
                    'tuning_binary_before': 'Tuning binary changed after build',
                    'tuning_binary_during': 'Tuning binary changed after build',
                }[failure]
                with self.assertRaisesRegex(ValueError, expected_error):
                    study.main()
            else:
                study.main()
        return out, native_commands

    def test_mock_collection_preserves_predeclared_guard(self):
        out, commands = self.run_mock_study()
        protocol = json.loads((out/'protocol.json').read_text())
        frozen = json.loads((out/'frozen_settings.json').read_text())
        status = json.loads((out/'study_status.json').read_text())
        self.assertEqual(len(commands), 3)
        self.assertEqual(protocol['status'], 'planned')
        self.assertEqual(frozen['protocol_sha256'], study.sha(out/'protocol.json'))
        self.assertEqual(frozen['frozen_state'], protocol['frozen_state'])
        self.assertEqual(frozen['selected_key_block'], 256)
        tuning_build = json.loads((out/'tuning/build.json').read_text())
        self.assertEqual(tuning_build['compiler'], self.binding['compiler'])
        self.assertEqual(frozen['tuning_runtime_conditions'], {'fixture': 'synthetic conditions'})
        self.assertEqual(status['status'], 'native_collected')
        self.assertTrue(all(job['status'] == 'passed' for job in status['jobs']))

    def test_replacement_build_stops_after_first_native_batch(self):
        out, commands = self.run_mock_study(failure='native_binary')
        status = json.loads((out/'study_status.json').read_text())
        self.assertEqual(len(commands), 1)
        self.assertEqual(status['status'], 'failed')
        self.assertEqual(status['jobs'][0]['status'], 'failed')
        self.assertIn('Frozen native binary changed', status['error'])

    def assert_tuning_failed_before_freeze(self, out, commands):
        self.assertEqual(commands, [])
        self.assertFalse((out/'frozen_settings.json').exists())
        self.assertEqual(json.loads((out/'study_status.json').read_text())['status'], 'failed')

    def test_different_compiler_stops_before_build(self):
        out, commands = self.run_mock_study(failure='compiler')
        self.assert_tuning_failed_before_freeze(out, commands)
        self.assertEqual(self.compile_calls, 0)
        self.assertEqual(self.tuning_commands, [])

    def test_conditions_change_stops_before_second_tuning_batch(self):
        out, commands = self.run_mock_study(failure='conditions')
        self.assert_tuning_failed_before_freeze(out, commands)
        self.assertEqual(len(self.tuning_commands), 1)

    def test_tuning_binary_change_before_call_rejected(self):
        out, commands = self.run_mock_study(failure='tuning_binary_before')
        self.assert_tuning_failed_before_freeze(out, commands)
        self.assertEqual(self.tuning_commands, [])

    def test_tuning_binary_change_during_call_rejected(self):
        out, commands = self.run_mock_study(failure='tuning_binary_during')
        self.assert_tuning_failed_before_freeze(out, commands)
        self.assertEqual(len(self.tuning_commands), 1)


if __name__ == '__main__':
    unittest.main()
