"""Run a single research process group with a strict wall-clock duty cycle.

No worker pool. STOP file and parent interruption terminate only this child
process group. One BLAS/Arrow thread, low scheduling priority, 50% duty cycle.
"""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

DUTY_CYCLE = .5
PERIOD_SECONDS = .2

class LowMemoryHeadroom(MemoryError):
    def __init__(self,free,minimum):
        self.free=free
        super().__init__(f'System memory availability {free}% below {minimum}%; stopped safely')

def mac_physical_footprint(pids):
    """Darwin RUSAGE_INFO_V0: charge resident plus compressed worker memory."""
    import ctypes
    import errno
    fields=['user_time','system_time','pkg_idle_wkups','interrupt_wkups','pageins',
            'wired_size','resident_size','phys_footprint','proc_start_abstime','proc_exit_abstime']
    class Usage(ctypes.Structure):
        _fields_=[('uuid',ctypes.c_uint8*16)]+[(name,ctypes.c_uint64) for name in fields]
    lib=ctypes.CDLL('/usr/lib/libproc.dylib',use_errno=True)
    lib.proc_pid_rusage.argtypes=[ctypes.c_int,ctypes.c_int,ctypes.c_void_p]
    lib.proc_pid_rusage.restype=ctypes.c_int
    total=0
    for pid in pids:
        usage=Usage()
        if lib.proc_pid_rusage(pid,0,ctypes.byref(usage)):
            if ctypes.get_errno()==errno.ESRCH:continue
            raise RuntimeError('Cannot measure worker compressed memory; stopped safely')
        total+=usage.phys_footprint
    return total/1024

def memory_snapshot(child_pid, rss_limit_mib, min_free_percent, warning_rss_limit_mib=None, pressure_state=None, warning_min_free_percent=35):
    """Bound total worker-tree RSS and check available system memory on macOS."""
    snapshot=subprocess.run(['ps','-axo','pid=,ppid=,rss='],capture_output=True,text=True)
    if snapshot.returncode:raise RuntimeError('Cannot monitor worker memory; stopped safely')
    rows=[tuple(map(int,row.split())) for row in snapshot.stdout.splitlines() if row.strip()]
    descendants={child_pid}
    for _ in range(len(rows)):
        additions={pid for pid,parent,_ in rows if parent in descendants}
        if additions<=descendants:break
        descendants|=additions
    rss=sum(value for pid,_,value in rows if pid in descendants)
    if rss>rss_limit_mib*1024:raise MemoryError(f'Worker tree exceeded {rss_limit_mib} MiB RSS; stopped safely')
    if sys.platform=='darwin':
        if warning_min_free_percent < 35 and child_pid > 0:
            footprint=mac_physical_footprint({pid for pid,_,_ in rows if pid in descendants})
            if footprint>rss_limit_mib*1024:
                raise MemoryError(f'Worker resident+compressed footprint exceeded {rss_limit_mib} MiB; stopped safely')
            if pressure_state is not None:pressure_state['footprint_kib']=footprint
        pressure=subprocess.run(['/usr/bin/memory_pressure','-Q'],capture_output=True,text=True)
        import re
        match=re.search(r'System-wide memory free percentage:\s*(\d+)%',pressure.stdout)
        if pressure.returncode or not match:raise RuntimeError('Cannot read system memory availability; stopped safely')
        free=int(match.group(1))
        level=subprocess.run(['sysctl','-n','kern.memorystatus_vm_pressure_level'],capture_output=True,text=True)
        if level.returncode or not level.stdout.strip():raise RuntimeError('Cannot read system pressure level; stopped safely')
        if pressure_state is not None:pressure_state['level']=level.stdout.strip()
        if level.stdout.strip()!='1':
            if level.stdout.strip()=='2' and warning_rss_limit_mib is not None:
                if rss>min(rss_limit_mib,warning_rss_limit_mib)*1024:
                    raise MemoryError(f'Warning-mode worker exceeded {warning_rss_limit_mib} MiB RSS; stopped safely')
            else:raise MemoryError(f'System memory pressure level {level.stdout.strip()}; stopped safely')
        minimum=max(warning_min_free_percent,min_free_percent) if level.stdout.strip()=='2' else min_free_percent
        if free<minimum:raise LowMemoryHeadroom(free,minimum)
    return rss


def wait_for_headroom(check,pause,swap_check,stop_check,timeout=60):
    """Pause our worker on transient headroom dips; never compute below its floor."""
    started=None
    while True:
        try:
            value=check()
            if started is not None:print('Memory headroom recovered; resuming saved in-process state',flush=True)
            return value
        except LowMemoryHeadroom as exc:
            # Below25% is a hard stop even when a worker is paused. Critical
            # pressure, excess RSS and unknown readings already raise above.
            if exc.free<25:raise
            if started is None:
                pause();started=time.monotonic()
                print('Pausing research worker for system memory headroom; no checkpoint restart',flush=True)
            swap_check();stop_check()
            if time.monotonic()-started>=timeout:raise
            time.sleep(1)


def swap_usage_mib():
    if sys.platform!='darwin':return None
    import re
    result=subprocess.run(['sysctl','vm.swapusage'],capture_output=True,text=True)
    match=re.search(r'used\s*=\s*([\d.]+)([MG])',result.stdout)
    if result.returncode or not match:raise RuntimeError('Cannot monitor swap usage; stopped safely')
    return float(match.group(1))*(1024 if match.group(2)=='G' else 1)


def effective_duty_cycle(requested,pressure_level):
    return min(requested,.2) if pressure_level=='2' else requested


def run(module,args,stop_file,timeout=1800,duty_cycle=DUTY_CYCLE,rss_limit_mib=1024,min_free_percent=25,warning_rss_limit_mib=None,max_swap_growth_mib=64,pause_low_headroom_seconds=0,adaptive_cpu=False,warning_min_free_percent=35):
    allowed={'scripts.discover_contract_signals','scripts.forecast_contract_variance','scripts.run_vega_portfolio','scripts.report_vega_portfolio','scripts.run_portfolio_rl','scripts.report_portfolio_rl','scripts.replay_frozen_atm_commission','scripts.retrain_atm_commission','scripts.audit_atm_commission_training','scripts.run_atm_cost_hurdles','scripts.audit_atm_cost_hurdles'}
    allowed.update({'scripts.run_atm_higher_cost_hurdles', 'scripts.audit_atm_higher_cost_hurdles',
                    'scripts.run_atm_nn_cost_hurdles', 'scripts.audit_atm_nn_cost_hurdles'})
    allowed.update({'scripts.continue_atm_nn_cost_hurdles', 'scripts.audit_atm_nn_cost_hurdles_complete'})
    allowed.add('scripts.report_atm_nn_full_roster')
    allowed.add('scripts.run_p25_feature_search')
    allowed.add('scripts.run_p25_earnings')
    allowed.add('scripts.run_p25_memory_ablation')
    allowed.add('scripts.run_p25_ungated_search')
    allowed.add('scripts.run_p25_variance_validation')
    allowed.add('scripts.finalize_p25_variance_candidate')
    allowed.add('scripts.diagnose_p25_variance_years')
    allowed.add('scripts.report_p25_variance_years')
    allowed.add('scripts.run_p25_capital_sensitivity')
    allowed.add('scripts.run_p25_trade_scale')
    allowed.add('scripts.run_p25_capital_policy')
    allowed.add('scripts.audit_p25_capital_policy')
    allowed.add('scripts.build_research_publication')
    allowed.add('scripts.rewrite_research_report')
    allowed.add('scripts.run_p25_validation_split')
    allowed.add('scripts.spread_paper_research')
    allowed.add('scripts.explore_put_atm_iv')
    allowed.add('scripts.run_all_pairs_earnings')
    allowed.add('scripts.run_all_pair_skew')
    allowed.add('scripts.explore_spot_iv_response')
    allowed.add('scripts.run_p25_spot_response')
    allowed.add('scripts.explore_long_spot_response')
    allowed.add('scripts.run_long_response_management')
    allowed.add('scripts.run_two_year_inventory')
    allowed.add('scripts.run_year_reversion_management')
    allowed.add('scripts.run_year_cheap_vega')
    allowed.add('scripts.export_teammate_slides')
    allowed.add('scripts.export_slide_chart_data')
    allowed.add('scripts.run_p25_reversion_q')
    allowed.add('scripts.run_p25_long_management')
    allowed.add('scripts.run_p25_short_management')
    allowed.add('scripts.run_p25_long_expiries')
    allowed.add('scripts.audit_long_expiry_parity')
    allowed.add('scripts.run_long_tenor_policy_search')
    allowed.add('scripts.run_long_tenor_full_universe')
    allowed.add('scripts.run_long_tenor_flexible_actions')
    allowed.add('scripts.report_long_tenor_search')
    allowed.add('scripts.run_long_tenor_fitted_q')
    allowed.add('scripts.run_long_tenor_reversible')
    allowed.add('scripts.run_year_option_action_critics')
    allowed.add('scripts.run_year_option_entry_rl')
    allowed.add('scripts.run_year_option_sharpe_policy')
    allowed.add('scripts.run_year_option_actual_policy')
    allowed.add('scripts.report_year_option_followup')
    allowed.add('scripts.run_native_year_entry')
    allowed.add('scripts.report_native_year_replacement')
    allowed.update({'scripts.run_atm65_ablation', 'scripts.run_atm65_ablation_audit'})
    allowed.add('scripts.diagnose_atm65_loss_years')
    allowed.update({'scripts.run_atm65_signal_mining', 'scripts.audit_atm65_signal_mining'})
    allowed.add('scripts.report_atm65_signal_concentration')
    allowed.update({'scripts.run_curve_management', 'scripts.audit_curve_management'})
    allowed.add('scripts.test_wrds_pair_signals')
    allowed.add('scripts.test_wrds_alternative_pairs')
    allowed.add('scripts.run_atm65_employee_sentiment')
    allowed.add('scripts.audit_atm65_employee_sentiment')
    allowed.add('scripts.save_employee_sentiment_checkpoint')
    allowed.add('scripts.run_atm65_transcripts')
    allowed.add('scripts.audit_atm65_transcripts')
    allowed.add('scripts.save_transcript_checkpoint')
    allowed.add('scripts.run_atm65_transcript_selector')
    allowed.add('scripts.run_atm65_rl_transcripts')
    allowed.add('scripts.export_current_rl_inputs')
    allowed.add('scripts.profile_vol_alternative_data')
    allowed.add('scripts.audit_current_financing')
    allowed.add('scripts.screen_new_vol_signals')
    allowed.add('scripts.run_atm65_volume_oi')
    allowed.add('scripts.audit_atm65_volume_oi')
    allowed.update({'scripts.run_atm65_single_leg', 'scripts.audit_atm65_single_leg'})
    allowed.add('scripts.repair_atm65_single_leg_quote')
    allowed.add('scripts.mine_atm_variance_signals')
    allowed.add('scripts.audit_atm_variance_mining')
    allowed.add('scripts.test_mined_signal_filters')
    allowed.add('scripts.run_atm65_mined_signals')
    allowed.add('scripts.audit_atm65_mined_signals')
    allowed.add('scripts.diagnose_current_trade_quality')
    allowed.add('scripts.followup_trade_quality')
    allowed.add('scripts.run_atm65_trade_quality')
    allowed.add('scripts.audit_atm65_trade_quality')
    allowed.add('scripts.explore_tail_features')
    allowed.add('scripts.run_tail_put_strategy')
    allowed.add('scripts.run_tail_put25_strategy')
    allowed.add('scripts.run_tail_put25_atm_transfer')
    allowed.add('scripts.audit_tail_put25_atm_transfer')
    allowed.add('scripts.run_tail_put25_rule_sizing')
    allowed.add('scripts.diagnose_atm_tail_gates')
    allowed.add('scripts.run_tail_put25_pooled')
    allowed.add('scripts.audit_tail_put25_pooled')
    allowed.add('scripts.repair_tail_put25_pooled')
    allowed.add('scripts.analyze_tail_universe_change')
    allowed.add('scripts.analyze_tail_legacy_calendar')
    allowed.add('scripts.compare_tail_saved_signals')
    allowed.add('scripts.run_tail_put25_legacy_rl')
    allowed.add('scripts.run_tail_put25_residual_rl')
    allowed.add('scripts.explore_opposite_wing')
    allowed.add('scripts.research_opposite_wing_signals')
    allowed.add('scripts.fetch_opposite_wing_fx')
    allowed.add('scripts.audit_opposite_wing_signals')
    allowed.add('scripts.run_smile_sizing')
    allowed.add('scripts.diagnose_smile_volatility')
    allowed.add('scripts.run_put75_cost_hurdles')
    allowed.add('scripts.run_put75_lower_hurdles')
    allowed.add('scripts.run_joint_smile')
    allowed.add('scripts.compare_smile_flat_size')
    allowed.add('scripts.run_uniform_smile')
    allowed.add('scripts.run_checkpoint_consistency')
    allowed.add('scripts.run_same_structure_smile')
    allowed.add('scripts.run_same_structure_calendar')
    allowed.add('scripts.rebuild_smile_contracts')
    allowed.add('scripts.run_recovered_smile')
    allowed.add('scripts.run_p25_signal_transfer')
    allowed.add('scripts.diagnose_p25_transfer')
    allowed.add('scripts.run_p25_maturity_transfer')
    allowed.add('scripts.run_p25_maturity_matched')
    allowed.add('scripts.diagnose_p25_recent_losses')
    allowed.add('scripts.profile_recent_loss_pairs')
    allowed.add('scripts.run_p25_risk_sizing')
    allowed.add('scripts.run_p25_action_windows')
    allowed.add('scripts.run_p25_recency')
    allowed.add('scripts.run_p25_cost_hurdles')
    allowed.add('scripts.run_p25_boundaries')
    allowed.add('scripts.run_p25_sharpe_search')
    allowed.add('scripts.submission')
    allowed.add('scripts.research_atm_rv_persistence')
    allowed.add('scripts.research_atm_rv_bandit')
    allowed.add('scripts.report_atm_p25_combination')
    allowed.add('scripts.atm_p25_allocation_q')
    allowed.add('scripts.atm_p25_residual_q')
    allowed.add('scripts.atm_p25_sharpe_policy')
    allowed.add('scripts.atm_p25_risk_policy')
    allowed.add('scripts.atm_p25_opportunity_policy')
    allowed.add('scripts.atm_p25_weekly_policy')
    allowed.add('scripts.research_atm_rv_straddle')
    allowed.add('scripts.research_atm_rv_straddle_full')
    allowed.add('scripts.prepare_straddle_quotes_buffered')
    allowed.add('scripts.run_straddle_replay_chunk')
    allowed.add('scripts.compare_entry_sizing_allocation')
    allowed.add('scripts.analyze_straddle_portfolio')
    allowed.add('scripts.straddle_borrow_stress')
    allowed.add('scripts.atm_vehicle_policy')
    allowed.add('scripts.atm_rv_straddle_bandit')
    allowed.add('scripts.run_straddle_fixed50')
    allowed.add('scripts.straddle_sizing_followup')
    allowed.add('scripts.straddle_capped_books')
    allowed.add('scripts.straddle_q_allocation')
    allowed.add('scripts.straddle_q_budget')
    allowed.add('scripts.run_capped_straddle_book')
    allowed.add('scripts.straddle_rv_q')
    allowed.add('scripts.straddle_q_diagnostics')
    allowed.add('scripts.straddle_entry_refinement')
    allowed.add('scripts.report_straddle_entry_refinement')
    allowed.add('scripts.straddle_feature_transfer')
    allowed.add('scripts.straddle_transfer_evaluation')
    allowed.add('scripts.straddle_transfer_fit')
    allowed.add('scripts.straddle_put_architecture')
    allowed.add('scripts.inspect_straddle_validation')
    allowed.add('scripts.straddle_learned_abstention')
    allowed.add('scripts.report_straddle_abstention')
    allowed.add('scripts.straddle_abstention_robustness')
    allowed.add('scripts.straddle_ordered_actor')
    allowed.add('scripts.straddle_rv_forecast')
    allowed.add('scripts.straddle_overnight_study')
    allowed.add('scripts.straddle_overnight_q')
    allowed.add('scripts.repair_straddle_checkpoints')
    allowed.add('scripts.report_straddle_overnight')
    allowed.add('scripts.straddle_context_followup')
    allowed.add('scripts.report_straddle_context')
    allowed.add('scripts.straddle_performance_followup')
    allowed.add('scripts.straddle_ensemble_search')
    allowed.add('scripts.straddle_policy_refinement')
    allowed.add('scripts.report_straddle_policy_search')
    allowed.add('scripts.report_put_straddle_bandits')
    allowed.add('scripts.report_put_straddle_overlay')
    allowed.add('scripts.put_straddle_q_frontier')
    allowed.add('scripts.put_straddle_q_frontier_regularized')
    allowed.add('scripts.report_q_profit_frontier')
    allowed.add('scripts.report_q_allocation_regimes')
    allowed.add('scripts.put_straddle_q_opportunity')
    allowed.add('scripts.report_q_opportunity')
    allowed.add('scripts.explain_retained_entry_policies')
    allowed.add('scripts.q_opportunity_tactical')
    allowed.add('scripts.report_entry_interpretability')
    allowed.add('scripts.report_q_tactical_audit')
    allowed.add('scripts.q_paper_robustness')
    allowed.add('scripts.q_custom_utility')
    allowed.add('scripts.q_custom_frontier')
    allowed.add('scripts.q_allocation_profiles')
    allowed.add('scripts.q_anchored_profiles')
    allowed.add('scripts.report_q_threshold_expansion')
    allowed.add('scripts.report_q_combined_objective')
    allowed.add('scripts.q_critic_stability')
    allowed.add('scripts.q_recent_performance')
    allowed.add('scripts.q_capital_splits')
    allowed.add('scripts.report_q_capital_splits')
    allowed.add('scripts.signal_capital_allocation')
    allowed.add('scripts.signal_allocation_discount')
    allowed.add('scripts.signal_allocation_variance')
    allowed.add('scripts.signal_learning_factorial')
    allowed.add('scripts.check_signal_allocation_variance_convergence')
    allowed.add('scripts.signal_allocation_objectives')
    allowed.add('scripts.signal_allocation_forecast_reward')
    allowed.add('scripts.signal_allocation_policy_reward')
    allowed.add('scripts.plot_signal_discount_comparison')
    allowed.add('scripts.explain_signal_allocation_099')
    allowed.add('scripts.explain_bandit_mechanisms')
    allowed.add('scripts.build_interpretable_options_paper')
    allowed.add('scripts.build_original_options_extension')
    allowed.add('scripts.build_unified_options_paper')
    allowed.add('scripts.export_options_overleaf')
    allowed.add('scripts.report_q_allocation_profiles')
    allowed.add('scripts.q_paper_diagnostics')
    allowed.add('scripts.build_allocation_extension')
    allowed.add('scripts.paper_allocation_benchmarks')
    if module not in allowed:raise ValueError('Unapproved research module')
    if warning_min_free_percent != 35:
        if not (module in {'scripts.export_options_overleaf','scripts.build_unified_options_paper','scripts.signal_learning_factorial','scripts.check_signal_allocation_variance_convergence','scripts.signal_allocation_variance','scripts.research_atm_rv_persistence','scripts.research_atm_rv_bandit','scripts.report_atm_p25_combination','scripts.atm_p25_allocation_q','scripts.atm_p25_residual_q','scripts.atm_p25_sharpe_policy','scripts.atm_p25_risk_policy','scripts.atm_p25_opportunity_policy','scripts.atm_p25_weekly_policy','scripts.research_atm_rv_straddle','scripts.research_atm_rv_straddle_full','scripts.prepare_straddle_quotes_buffered','scripts.run_straddle_replay_chunk','scripts.compare_entry_sizing_allocation','scripts.analyze_straddle_portfolio','scripts.straddle_borrow_stress','scripts.atm_vehicle_policy','scripts.atm_rv_straddle_bandit','scripts.run_straddle_fixed50','scripts.straddle_sizing_followup','scripts.straddle_capped_books','scripts.straddle_q_allocation','scripts.straddle_q_budget','scripts.run_capped_straddle_book','scripts.straddle_rv_q','scripts.straddle_q_diagnostics','scripts.straddle_entry_refinement','scripts.report_straddle_entry_refinement','scripts.straddle_feature_transfer','scripts.straddle_transfer_evaluation','scripts.straddle_transfer_fit','scripts.straddle_put_architecture','scripts.inspect_straddle_validation','scripts.straddle_learned_abstention','scripts.report_straddle_abstention','scripts.straddle_abstention_robustness','scripts.straddle_ordered_actor','scripts.straddle_rv_forecast','scripts.straddle_overnight_study','scripts.straddle_overnight_q','scripts.repair_straddle_checkpoints','scripts.report_straddle_overnight','scripts.straddle_context_followup','scripts.report_straddle_context','scripts.straddle_performance_followup','scripts.straddle_ensemble_search','scripts.straddle_policy_refinement','scripts.report_straddle_policy_search','scripts.report_put_straddle_bandits','scripts.report_put_straddle_overlay','scripts.put_straddle_q_frontier','scripts.put_straddle_q_frontier_regularized','scripts.report_q_profit_frontier','scripts.report_q_allocation_regimes','scripts.put_straddle_q_opportunity','scripts.report_q_opportunity','scripts.explain_retained_entry_policies','scripts.q_opportunity_tactical','scripts.report_entry_interpretability','scripts.report_q_tactical_audit','scripts.q_paper_robustness','scripts.q_custom_utility','scripts.q_custom_frontier','scripts.q_allocation_profiles','scripts.q_anchored_profiles','scripts.report_q_threshold_expansion','scripts.report_q_combined_objective','scripts.q_critic_stability','scripts.q_recent_performance','scripts.q_capital_splits','scripts.report_q_capital_splits','scripts.signal_capital_allocation','scripts.signal_allocation_variance','scripts.signal_allocation_discount','scripts.signal_allocation_objectives','scripts.signal_allocation_forecast_reward','scripts.signal_allocation_policy_reward','scripts.plot_signal_discount_comparison','scripts.explain_signal_allocation_099','scripts.explain_bandit_mechanisms','scripts.build_interpretable_options_paper','scripts.build_original_options_extension','scripts.report_q_allocation_profiles','scripts.q_paper_diagnostics','scripts.build_allocation_extension','scripts.paper_allocation_benchmarks'} and 30 <= warning_min_free_percent <= 35
                and rss_limit_mib <= 256 and warning_rss_limit_mib is not None and warning_rss_limit_mib <= 256
                and duty_cycle <= .2 and min_free_percent >= 30 and max_swap_growth_mib <= 256):
            raise ValueError('Reduced headroom requires the isolated ATM study at20% CPU and256 MiB or less')
    if not 0<duty_cycle<=.5 or not 128<=rss_limit_mib<=3072 or not 0<=min_free_percent<=100:
        raise ValueError('Invalid resource limits')
    if warning_rss_limit_mib is not None and (not 128<=warning_rss_limit_mib<=384 or (duty_cycle>.2 and not adaptive_cpu)):
        raise ValueError('Warning mode requires at most384 MiB and20% CPU duty')
    if adaptive_cpu and warning_rss_limit_mib is None:raise ValueError('Adaptive CPU requires warning-mode memory and swap protection')
    if not 0<=pause_low_headroom_seconds<=60:raise ValueError('Headroom pause must be at most60 seconds')
    if stop_file.exists():raise InterruptedError('STOP file present')
    # No research worker exists during the startup headroom wait. Anchor swap
    # growth only after that wait, immediately before launching the worker.
    initial_swap=None
    def swap_check():
        if initial_swap is not None and swap_usage_mib()-initial_swap>max_swap_growth_mib:
            raise MemoryError(f'System swap increased by more than {max_swap_growth_mib} MiB; stopped safely')
    def stop_check():
        if stop_file.exists():raise InterruptedError('STOP file present')
    pressure_state={} if adaptive_cpu else None
    startup=lambda:memory_snapshot(-1,rss_limit_mib,min_free_percent,warning_rss_limit_mib,pressure_state,warning_min_free_percent)
    if pause_low_headroom_seconds:
        wait_for_headroom(startup,lambda:None,swap_check,stop_check,pause_low_headroom_seconds)
    else:startup()
    initial_swap=swap_usage_mib() if warning_rss_limit_mib is not None else None
    env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',
             VECLIB_MAXIMUM_THREADS='1',NUMEXPR_NUM_THREADS='1',ARROW_NUM_THREADS='1')
    env['RESEARCH_SUPERVISOR_PID']=str(os.getpid())
    bootstrap="""import os,runpy,sys,threading,time
os.nice(15)
def watchdog():
    while True:
        if os.getppid()!=int(os.environ['RESEARCH_SUPERVISOR_PID']):os._exit(143)
        time.sleep(.2)
threading.Thread(target=watchdog,daemon=True).start()
if sys.argv[1] != 'scripts.export_options_overleaf':
    import pyarrow
    pyarrow.set_cpu_count(1);pyarrow.set_io_thread_count(1)
if sys.argv[1] in {'scripts.straddle_overnight_study','scripts.straddle_overnight_q','scripts.repair_straddle_checkpoints','scripts.report_straddle_overnight','scripts.straddle_context_followup','scripts.report_straddle_context','scripts.straddle_performance_followup','scripts.straddle_ensemble_search','scripts.straddle_policy_refinement','scripts.report_straddle_policy_search','scripts.report_put_straddle_bandits','scripts.report_put_straddle_overlay','scripts.put_straddle_q_frontier','scripts.put_straddle_q_frontier_regularized','scripts.report_q_profit_frontier','scripts.report_q_allocation_regimes','scripts.put_straddle_q_opportunity','scripts.report_q_opportunity','scripts.explain_retained_entry_policies','scripts.q_opportunity_tactical','scripts.report_entry_interpretability','scripts.report_q_tactical_audit','scripts.q_paper_robustness','scripts.q_custom_utility','scripts.q_custom_frontier','scripts.q_allocation_profiles','scripts.q_anchored_profiles','scripts.report_q_threshold_expansion','scripts.report_q_combined_objective','scripts.q_critic_stability','scripts.q_recent_performance','scripts.q_capital_splits','scripts.report_q_capital_splits','scripts.signal_capital_allocation','scripts.signal_allocation_variance','scripts.signal_allocation_discount','scripts.signal_allocation_objectives','scripts.signal_allocation_forecast_reward','scripts.signal_allocation_policy_reward','scripts.plot_signal_discount_comparison','scripts.explain_signal_allocation_099','scripts.explain_bandit_mechanisms','scripts.build_interpretable_options_paper','scripts.build_original_options_extension','scripts.report_q_allocation_profiles','scripts.q_paper_diagnostics','scripts.build_allocation_extension','scripts.paper_allocation_benchmarks'}:
    from scripts.research_atomic_io import install
    install()
m=sys.argv.pop(1);runpy.run_module(m,run_name='__main__')
"""
    child=subprocess.Popen([sys.executable,'-c',bootstrap,module,*args],env=env,start_new_session=True)
    started=time.monotonic();last_check=started;peak_rss=0
    def send(sig):
        try:os.killpg(child.pid,sig)
        except ProcessLookupError:child.wait()
        except PermissionError as exc:
            # macOS may return EPERM when the child exits between poll and
            # killpg. Suppress it only after waitpid confirms termination.
            # waitpid can lag the kernel's process-group teardown. Allow a
            # bounded reap window; a live inaccessible child is still fatal.
            try:child.wait(timeout=1)
            except subprocess.TimeoutExpired:raise exc
    try:
        while child.poll() is None:
            if stop_file.exists():raise InterruptedError('STOP file present')
            if time.monotonic()-started>timeout:raise TimeoutError('Job wall-clock limit reached; checkpoint retained')
            # Cap the whole process group, not merely the Python parent.
            effective=effective_duty_cycle(duty_cycle,pressure_state.get('level')) if adaptive_cpu else duty_cycle
            send(signal.SIGCONT);time.sleep(PERIOD_SECONDS * effective)
            if child.poll() is not None:break
            send(signal.SIGSTOP);time.sleep(PERIOD_SECONDS * (1-effective))
            if time.monotonic()-last_check>=1 and child.poll() is None:
                last_check=time.monotonic()
                check=lambda:memory_snapshot(child.pid,rss_limit_mib,min_free_percent,warning_rss_limit_mib,pressure_state,warning_min_free_percent)
                if pause_low_headroom_seconds:
                    rss=wait_for_headroom(check,lambda:send(signal.SIGSTOP),swap_check,stop_check,pause_low_headroom_seconds)
                else:rss=check()
                peak_rss=max(peak_rss,rss);swap_check()
        if child.returncode:raise subprocess.CalledProcessError(child.returncode,module)
    finally:
        if child.poll() is None:
            send(signal.SIGCONT);send(signal.SIGTERM)
            try:child.wait(timeout=5)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
        print(f'Guard observed peak worker RSS: {peak_rss/1024:.1f} MiB',flush=True)

def main():
    signal.signal(signal.SIGTERM,lambda *_:(_ for _ in ()).throw(KeyboardInterrupt()))
    p=argparse.ArgumentParser();p.add_argument('--module',required=True);p.add_argument('--timeout',type=int,default=1800)
    p.add_argument('--duty-cycle',type=float,default=DUTY_CYCLE)
    p.add_argument('--rss-limit-mib',type=int,default=1024)
    p.add_argument('--min-free-percent',type=int,default=25)
    p.add_argument('--warning-rss-limit-mib',type=int)
    p.add_argument('--max-swap-growth-mib',type=float,default=64)
    p.add_argument('--pause-low-headroom-seconds',type=int,default=0)
    p.add_argument('--adaptive-cpu',action='store_true')
    p.add_argument('--warning-min-free-percent',type=int,default=35)
    p.add_argument('--stop-file',type=Path,default=Path('artifacts/contracts/signal_discovery/STOP'))
    p.add_argument('arguments',nargs=argparse.REMAINDER);a=p.parse_args()
    run(a.module,a.arguments[1:] if a.arguments[:1]==['--'] else a.arguments,a.stop_file,a.timeout,a.duty_cycle,a.rss_limit_mib,a.min_free_percent,a.warning_rss_limit_mib,a.max_swap_growth_mib,a.pause_low_headroom_seconds,a.adaptive_cpu,a.warning_min_free_percent)

if __name__=='__main__':main()
