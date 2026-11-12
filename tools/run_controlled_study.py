"""Tune once, freeze the plan, then collect three separate complete native grids."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys

from build_integrated import ROOT, BASE_FLAGS, CXX, SUFFIX
from run_streaming_experiment import summarize, write_csv
from runtime_conditions import apply_cpu_affinity, capture_conditions, parse_cpu_list


def stamp():
    return datetime.now(timezone.utc).isoformat()


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def capture_frozen_state(root, binding, revision, native_binary):
    """Bind the benchmark and the scripts that control collection before tuning."""
    controls = (
        'tools/run_controlled_study.py', 'tools/run_project_grid.py',
        'tools/run_scalar_bench.py', 'tools/run_streaming_experiment.py',
        'tools/runtime_conditions.py', 'tools/bench_streaming.cpp',
    )
    sources = dict(binding['source_sha256'])
    sources.update({name: sha(root/name) for name in controls})
    return dict(source_revision=revision, native_binary=native_binary,
                native_binary_sha256=binding['binary_sha256'],
                build_record='build/integrated_build.json',
                build_record_sha256=sha(root/'build/integrated_build.json'),
                source_sha256=sources)


def verify_frozen_state(root, expected):
    """Reject changed state even if a replacement build is internally consistent."""
    revision = subprocess.check_output(
        ['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()
    if revision != expected['source_revision']:
        raise ValueError('Frozen Git revision changed')
    if subprocess.check_output(['git', 'status', '--porcelain'], cwd=root, text=True).strip():
        raise ValueError('Frozen working tree is no longer clean')
    checks = [
        (expected['native_binary'], expected['native_binary_sha256'], 'native binary'),
        (expected['build_record'], expected['build_record_sha256'], 'build record'),
        *[(name, digest, 'source') for name, digest in expected['source_sha256'].items()],
    ]
    for name, digest, kind in checks:
        path = root/name
        if not path.is_file() or sha(path) != digest:
            raise ValueError(f'Frozen {kind} changed: {name}')


def verify_tuning_binary(binary, expected_sha256):
    if not binary.is_file() or sha(binary) != expected_sha256:
        raise ValueError('Tuning binary changed after build')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--cpu-order',required=True)
    parser.add_argument('--physical-cores',type=int,required=True)
    args=parser.parse_args()
    cpus=parse_cpu_list(args.cpu_order)
    if not 1 <= args.physical_cores <= 4096:
        parser.error('Physical core count must be positive and verified on this machine')
    if len(cpus)<min(8,args.physical_cores) or args.out.exists():
        parser.error('Need enough verified CPUs and a new output directory')
    out=args.out.resolve()
    try:
        out.relative_to(ROOT/'results')
    except ValueError:
        parser.error('Output must be inside this repository results directory')
    if subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip():
        parser.error('Commit the tested code before starting the study')
    if os.environ.get('OMP_WAIT_POLICY')!='PASSIVE':
        parser.error('This protocol requires OMP_WAIT_POLICY=PASSIVE')
    revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    binding=json.loads((ROOT/'build/integrated_build.json').read_text())
    if binding['status']!='passed' or binding['binary_sha256']!=sha(ROOT/'build'/('attn_bench_all'+SUFFIX)):
        parser.error('Rebuild the integrated executable first')
    for name,digest in binding['source_sha256'].items():
        if sha(ROOT/name)!=digest:parser.error('Source changed after build: '+name)
    frozen_state = capture_frozen_state(
        ROOT, binding, revision, 'build/attn_bench_all'+SUFFIX)
    verify_frozen_state(ROOT, frozen_state)
    out.mkdir(parents=True)
    protocol=dict(status='planned',created_utc=stamp(),source_revision=revision,
        cpu_order=cpus,physical_cores=args.physical_cores,
        native_binary_sha256=binding['binary_sha256'],
        frozen_state=frozen_state,
        tuning=dict(shape=[2048,64,64],query_block=16,candidates=[128,256,512],
            mode='blocked scalar T1',batches=3,warmup=10,reps=20,seed=20261022,
            compiler=binding['compiler'],
            phases=[0,1,2],selection='minimum median of three batch medians; exact tie: smaller Bk'),
        main=dict(batches=3,warmup=10,reps=20,seed=20261020,
            job_orders=['forward','reverse','rotate'],configurations_per_batch=60 if args.physical_cores>=8 else None,
            variant_order='existing cyclic rotation in each sample; residual predecessor-order effects are not removed'),
        scheduling='serial first CPU; OpenMP first T CPU IDs; eligibility, not worker pinning',
        waiting='PASSIVE; thread wake-up overhead remains inside native call',
        thermal_policy='no deliberate batch pause; temperature/frequency not controlled; report all three batches',
        jobs=[])
    save(out/'protocol.json',protocol)
    tune_dir=out/'tuning'
    tune_dir.mkdir()
    sources=['src/blocked_streaming_scalar.cpp','src/oracle_fp64.cpp','src/input.cpp','src/metrics.cpp','tools/bench_streaming.cpp']
    binary=ROOT/'build'/('attn_block_tune.exe' if os.name=='nt' else 'attn_block_tune')
    try:
        version_lines = subprocess.check_output(
            [CXX, '--version'], cwd=ROOT, text=True, encoding='utf-8', errors='replace').splitlines()
        tune_compiler = version_lines[0] if version_lines else ''
        if tune_compiler != binding['compiler']:
            raise ValueError('Tuning compiler differs from the native build compiler')
        tune_sources = sorted(set(sources + [p.relative_to(ROOT).as_posix() for p in ROOT.glob('include/*.h')]
                                 + ['tools/run_controlled_study.py', 'tools/run_streaming_experiment.py',
                                    'tools/runtime_conditions.py', 'tools/build_integrated.py']))
        before_build = {s:sha(ROOT/s) for s in tune_sources}
        subprocess.run([CXX,*BASE_FLAGS,*sources,'-o',str(binary)],cwd=ROOT,check=True)
        if before_build != {s:sha(ROOT/s) for s in tune_sources}:
            raise ValueError('Tuning source changed during build')
        tune_binding=dict(binary_sha256=sha(binary),source_sha256=before_build,
                          compiler=tune_compiler,flags=BASE_FLAGS)
        save(tune_dir/'build.json',tune_binding)
        all_summary=[]
        tuning_conditions = None
        for batch in range(3):
            verify_frozen_state(ROOT, frozen_state)
            print('Tuning batch',batch+1,flush=True)
            apply_cpu_affinity(str(cpus[0]))
            before=capture_conditions()
            if tuning_conditions is None:
                tuning_conditions = before
            elif before != tuning_conditions:
                raise ValueError('Tuning runtime conditions changed between batches')
            verify_tuning_binary(binary, tune_binding['binary_sha256'])
            started=stamp()
            result=subprocess.run([str(binary),'tune',str(batch)],cwd=ROOT,capture_output=True,text=True,check=True)
            finished=stamp()
            after=capture_conditions()
            verify_tuning_binary(binary, tune_binding['binary_sha256'])
            verify_frozen_state(ROOT, frozen_state)
            if before!=after:raise ValueError('Tuning runtime conditions changed')
            rows=summarize(result.stdout,[(2048,64,64,bk) for bk in (128,256,512)],phase=batch)
            (tune_dir/f'batch{batch+1}_raw.csv').write_text(result.stdout,encoding='utf-8',newline='')
            write_csv(tune_dir/f'batch{batch+1}_summary.csv',rows)
            save(tune_dir/f'batch{batch+1}_environment.json',dict(status='passed',started_utc=started,
                finished_utc=finished,phase=batch,runtime_conditions=before,source_revision=revision,**tune_binding))
            all_summary.append(rows)
        candidates=[]
        for bk in (128,256,512):
            medians=[next(r['median_ms'] for r in rows if r['key_block']==bk) for rows in all_summary]
            candidates.append(dict(key_block=bk,batch_medians_ms=medians,selection_value_ms=statistics.median(medians)))
        selected=min(candidates,key=lambda r:(r['selection_value_ms'],r['key_block']))['key_block']
        verify_frozen_state(ROOT, frozen_state)
        freeze=dict(status='frozen',frozen_utc=stamp(),selected_key_block=selected,candidates=candidates,
            protocol_sha256=sha(out/'protocol.json'),source_revision=revision,
            native_binary_sha256=binding['binary_sha256'],cpu_order=cpus,
            frozen_state=frozen_state,
            tuning_runtime_conditions=tuning_conditions,
            scope='one scalar-selected common Bk; not per-mode optimal')
        save(out/'frozen_settings.json',freeze)
        print('Frozen Bk',selected,flush=True)
        # Child runners select their own prefix; restore the eligible pool before launch.
        apply_cpu_affinity(','.join(map(str,cpus)))
        for batch,order in enumerate(('forward','reverse','rotate'),1):
            verify_frozen_state(ROOT, frozen_state)
            job=dict(batch=batch,started_utc=stamp(),status='running')
            protocol['jobs'].append(job)
            protocol['status']='running'
            save(out/'study_status.json',protocol)
            destination=out/f'native_{batch}'
            command=[sys.executable,'tools/run_project_grid.py','--out',destination.relative_to(ROOT).as_posix(),
                '--physical-cores',str(args.physical_cores),'--streaming','--key-block',str(selected),
                '--query-block','16','--warmup','10','--reps','20','--seed','20261020',
                '--cpu-order',args.cpu_order,'--job-order',order]
            job['command']=command[1:]
            print('Native batch',batch,order,flush=True)
            with (out/f'native_{batch}.log').open('w',encoding='utf-8') as log:
                subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
            verify_frozen_state(ROOT, frozen_state)
            job.update(status='passed',finished_utc=stamp())
            save(out/'study_status.json',protocol)
        protocol['status']='native_collected'
    except Exception as error:
        protocol.update(status='failed',error_type=type(error).__name__,error=str(error))
        if protocol['jobs'] and protocol['jobs'][-1]['status'] == 'running':
            protocol['jobs'][-1].update(status='failed', finished_utc=stamp())
        raise
    finally:
        protocol['finished_utc']=stamp()
        save(out/'study_status.json',protocol)
    print('Collected all three native batches; analyze the saved records next.',flush=True)


if __name__=='__main__':main()
