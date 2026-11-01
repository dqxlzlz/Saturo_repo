import io
import json
import subprocess
import sys
from types import SimpleNamespace

import pytest

from saturo.cli import endtoend_experiment as e2e
from saturo.cli import global_scheduler as scheduler_cli


def config_file(tmp_path, **updates):
    payload = dict(model='org/model', resource_config_filter='tp1_pp1',
                   rt_workload_csv='rt.csv', be_workload_csv='be.csv',
                   num_req=24, target_rps=4, output_dir=str(tmp_path/'run'))
    payload.update(updates)
    path=tmp_path/'experiment.json'
    path.write_text(json.dumps(payload))
    return path


def test_config_cli_precedence_and_all_scheduler_options_round_trip(tmp_path):
    path=config_file(tmp_path, adaptive_role_switching=False,worker_warmup=False,
        decode_saturation_lookahead_s=7,resource_memory_watermark=.8,
        role_switch_interval_s=3,worker_warmup_timeout_s=9,
        rt_prefill_deadline_s=20,worker_warmup_output_len=5)
    args=e2e.parse_args(['--config',str(path),'--num-req','12','--worker-warmup'])
    assert args.num_req==12 and args.window_s==3
    assert args.worker_warmup and not args.adaptive_role_switching
    command=e2e.build_command(args)
    restored=scheduler_cli.parse_args(command[2:])
    for name in vars(restored):
        if hasattr(args,name):
            assert getattr(restored,name)==getattr(args,name),name
    assert '--no-adaptive-role-switching' in command
    assert restored.decode_saturation_lookahead_s==7
    assert restored.resource_memory_watermark==.8


@pytest.mark.parametrize('updates', [
    {'num_request':24}, {'worker_warmup':'false'}, {'num_req':1.5},
    {'env':{'CUDA_VISIBLE_DEVICES':4}},
])
def test_bad_config_cannot_be_silently_ignored(tmp_path, updates):
    path=config_file(tmp_path,**updates)
    with pytest.raises(SystemExit):
        e2e.parse_args(['--config',str(path)])


def test_print_command_starts_nothing_and_writes_no_run_directory(
        tmp_path, monkeypatch, capsys):
    path=config_file(tmp_path)
    monkeypatch.setattr(sys,'argv',['e2e','--config',str(path),'--print-command'])
    def forbidden(*args,**kwargs):
        raise AssertionError('Preview must not start a subprocess')
    monkeypatch.setattr(e2e.subprocess,'Popen',forbidden)
    e2e.main()
    manifest=json.loads(capsys.readouterr().out)
    assert manifest['parameters']['num_req']==24
    assert len(manifest['stages'])==7
    assert not (tmp_path/'run').exists()


def test_config_paths_are_repo_relative_from_any_working_directory(tmp_path, monkeypatch):
    path=config_file(tmp_path)
    monkeypatch.chdir(tmp_path)
    args=e2e.parse_args(['--config',str(path)])
    assert args.rt_workload_csv==str(e2e.repo_root()/'rt.csv')
    assert args.model=='org/model'
    assert '--worker' not in e2e.build_command(args)


@pytest.mark.parametrize('plan_only',[False,True])
def test_config_environment_and_plan_only_result(tmp_path, monkeypatch, plan_only):
    path=config_file(tmp_path,env={'SATURO_REDIS_URL':None,'EXPERIMENT_TAG':'review'})
    argv=['--config',str(path)] + (['--dry-run-worker-plan'] if plan_only else [])
    args=e2e.parse_args(argv)
    monkeypatch.setenv('SATURO_REDIS_URL','inherited-url')
    def launch(command,**kwargs):
        assert 'SATURO_REDIS_URL' not in kwargs['env']
        assert kwargs['env']['EXPERIMENT_TAG']=='review'
        assert kwargs['cwd']==str(e2e.repo_root())
        target=tmp_path/'run'
        if plan_only:
            (target/'resource_worker_plan.json').write_text(json.dumps({'placements':[]}))
        else:
            (target/'summary.json').write_text(json.dumps({'status':'completed'}))
        return SimpleNamespace(stdout=io.StringIO(''),returncode=0,
                               wait=lambda:0,poll=lambda:0)
    monkeypatch.setattr(e2e.subprocess,'Popen',launch)
    result=e2e.run(args)
    assert result['status']==('planned' if plan_only else 'completed')
    manifest=json.loads((tmp_path/'run/experiment_config.json').read_text())
    assert len(manifest['stages'])==(2 if plan_only else 7)
    assert manifest['environment']['EXPERIMENT_TAG']=='review'


def test_cli_import_does_not_load_engine_or_torch():
    subprocess.run([sys.executable,'-c',
        "import saturo.cli.endtoend_experiment; import sys; "
        "assert 'torch' not in sys.modules; "
        "assert 'saturo.runtime.scheduler' not in sys.modules"],
        cwd=str(e2e.repo_root()),check=True)
