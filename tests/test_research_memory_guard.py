import subprocess
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.run_signal_discovery_guarded import memory_snapshot,LowMemoryHeadroom,wait_for_headroom,effective_duty_cycle


class MemoryGuardTest(unittest.TestCase):
    @patch('scripts.run_signal_discovery_guarded.mac_physical_footprint',return_value=300*1024)
    @patch('scripts.run_signal_discovery_guarded.subprocess.run')
    @patch('scripts.run_signal_discovery_guarded.sys.platform','darwin')
    def test_compressed_worker_memory_cannot_bypass_small_worker_cap(self,run,footprint):
        run.return_value=subprocess.CompletedProcess(['ps'],0,stdout='10 1 100000\n')
        with self.assertRaisesRegex(MemoryError,'resident\\+compressed'):
            memory_snapshot(10,256,30,256,warning_min_free_percent=30)
        footprint.assert_called_once_with({10})

    @patch('scripts.run_signal_discovery_guarded.subprocess.run')
    @patch('scripts.run_signal_discovery_guarded.sys.platform','darwin')
    def test_scoped_small_worker_headroom_keeps_critical_and_rss_stops(self,run):
        def command(args,**kwargs):
            result=self.command(args,**kwargs)
            if args[0]=='sysctl':result.stdout='2\n'
            if args[0]=='/usr/bin/memory_pressure':result.stdout='System-wide memory free percentage: 33%'
            return result
        run.side_effect=command
        with self.assertRaises(LowMemoryHeadroom):memory_snapshot(-1,256,30,256)
        self.assertEqual(memory_snapshot(-1,256,30,256,warning_min_free_percent=30),0)
        with self.assertRaisesRegex(MemoryError,'Worker tree'):
            memory_snapshot(10,256,30,256,warning_min_free_percent=30)
        def critical(args,**kwargs):
            result=command(args,**kwargs)
            if args[0]=='sysctl':result.stdout='4\n'
            return result
        run.side_effect=critical
        with self.assertRaisesRegex(MemoryError,'pressure level 4'):
            memory_snapshot(-1,256,30,256,warning_min_free_percent=30)

    def test_reduced_headroom_is_restricted_to_tiny_atm_worker(self):
        from scripts.run_signal_discovery_guarded import run
        for module,rss,duty in [('scripts.run_year_cheap_vega',256,.2),
                                ('scripts.research_atm_rv_persistence',384,.2),
                                ('scripts.research_atm_rv_persistence',256,.5),
                                ('scripts.report_atm_p25_combination',384,.2),
                                ('scripts.report_atm_p25_combination',256,.5),
                                ('scripts.atm_p25_allocation_q',384,.2),
                                ('scripts.atm_p25_allocation_q',256,.5),
                                ('scripts.atm_p25_residual_q',384,.2),
                                ('scripts.atm_p25_residual_q',256,.5),
                                ('scripts.atm_p25_sharpe_policy',384,.2),
                                ('scripts.atm_p25_sharpe_policy',256,.5),
                                ('scripts.atm_p25_risk_policy',384,.2),
                                ('scripts.atm_p25_risk_policy',256,.5),
                                ('scripts.atm_p25_opportunity_policy',384,.2),
                                ('scripts.atm_p25_opportunity_policy',256,.5),
                                ('scripts.atm_p25_weekly_policy',384,.2),
                                ('scripts.atm_p25_weekly_policy',256,.5),
                                ('scripts.research_atm_rv_straddle',384,.2),
                                ('scripts.research_atm_rv_straddle',256,.5),
                                ('scripts.research_atm_rv_straddle_full',384,.2),
                                ('scripts.research_atm_rv_straddle_full',256,.5),
                                ('scripts.prepare_straddle_quotes_buffered',384,.2),
                                ('scripts.prepare_straddle_quotes_buffered',256,.5),
                                ('scripts.run_straddle_replay_chunk',384,.2),
                                ('scripts.run_straddle_replay_chunk',256,.5),
                                ('scripts.compare_entry_sizing_allocation',384,.2),
                                ('scripts.compare_entry_sizing_allocation',256,.5),
                                ('scripts.analyze_straddle_portfolio',384,.2),
                                ('scripts.analyze_straddle_portfolio',256,.5),
                                ('scripts.straddle_borrow_stress',384,.2),
                                ('scripts.straddle_borrow_stress',256,.5),
                                ('scripts.atm_vehicle_policy',384,.2),
                                ('scripts.atm_vehicle_policy',256,.5),
                                ('scripts.atm_rv_straddle_bandit',384,.2),
                                ('scripts.atm_rv_straddle_bandit',256,.5),
                                ('scripts.run_straddle_fixed50',384,.2),
                                ('scripts.run_straddle_fixed50',256,.5),
                                ('scripts.straddle_sizing_followup',384,.2),
                                ('scripts.straddle_sizing_followup',256,.5),
                                ('scripts.straddle_capped_books',384,.2),
                                ('scripts.straddle_capped_books',256,.5),
                                ('scripts.straddle_q_allocation',384,.2),
                                ('scripts.straddle_q_allocation',256,.5),
                                ('scripts.straddle_q_budget',384,.2),
                                ('scripts.straddle_q_budget',256,.5),
                                ('scripts.run_capped_straddle_book',384,.2),
                                ('scripts.run_capped_straddle_book',256,.5),
                                ('scripts.straddle_rv_q',384,.2),
                                ('scripts.straddle_rv_q',256,.5),
                                ('scripts.straddle_q_diagnostics',384,.2),
                                ('scripts.straddle_q_diagnostics',256,.5),
                                ('scripts.straddle_entry_refinement',384,.2),
                                ('scripts.straddle_entry_refinement',256,.5),
                                ('scripts.report_straddle_entry_refinement',384,.2),
                                ('scripts.report_straddle_entry_refinement',256,.5),
                                ('scripts.straddle_feature_transfer',384,.2),
                                ('scripts.straddle_feature_transfer',256,.5),
                                ('scripts.straddle_transfer_evaluation',384,.2),
                                ('scripts.straddle_transfer_evaluation',256,.5),
                                ('scripts.straddle_transfer_fit',384,.2),
                                ('scripts.straddle_transfer_fit',256,.5),
                                ('scripts.straddle_put_architecture',384,.2),
                                ('scripts.straddle_put_architecture',256,.5),
                                ('scripts.inspect_straddle_validation',384,.2),
                                ('scripts.inspect_straddle_validation',256,.5),
                                ('scripts.straddle_learned_abstention',384,.2),
                                ('scripts.straddle_learned_abstention',256,.5),
                                ('scripts.report_straddle_abstention',384,.2),
                                ('scripts.report_straddle_abstention',256,.5),
                                ('scripts.straddle_abstention_robustness',384,.2),
                                ('scripts.straddle_abstention_robustness',256,.5),
                                ('scripts.straddle_ordered_actor',384,.2),
                                ('scripts.straddle_ordered_actor',256,.5),
                                ('scripts.straddle_rv_forecast',384,.2),
                                ('scripts.straddle_rv_forecast',256,.5),
                                ('scripts.straddle_overnight_study',384,.2),
                                ('scripts.straddle_overnight_study',256,.5),
                                ('scripts.straddle_overnight_q',384,.2),
                                ('scripts.straddle_overnight_q',256,.5),
                                ('scripts.repair_straddle_checkpoints',384,.2),
                                ('scripts.repair_straddle_checkpoints',256,.5),
                                ('scripts.report_straddle_overnight',384,.2),
                                ('scripts.report_straddle_overnight',256,.5)]:
            with self.assertRaisesRegex(ValueError,'Reduced headroom'):
                run(module,[],Path('/tmp/unused-atm-stop'),duty_cycle=duty,rss_limit_mib=rss,
                    min_free_percent=30,warning_rss_limit_mib=256,warning_min_free_percent=30)

    def test_swap_anchor_is_taken_after_startup_headroom_recovers(self):
        from unittest.mock import Mock
        from scripts.run_signal_discovery_guarded import run
        child=Mock(pid=987654,returncode=0)
        child.poll.return_value=0
        with tempfile.TemporaryDirectory() as temp:
            with patch('scripts.run_signal_discovery_guarded.swap_usage_mib',return_value=100.) as swap:
                def sample(*args,**kwargs):
                    self.assertEqual(swap.call_count,0)
                    return 0
                with patch('scripts.run_signal_discovery_guarded.memory_snapshot',side_effect=sample), \
                     patch('scripts.run_signal_discovery_guarded.subprocess.Popen',return_value=child):
                    run('scripts.run_long_response_management',[],Path(temp)/'STOP',duty_cycle=.2,
                        rss_limit_mib=384,min_free_percent=35,warning_rss_limit_mib=384,pause_low_headroom_seconds=60)
                swap.assert_called_once()

    def test_adaptive_cpu_preserves_warning_mode_cap(self):
        self.assertEqual(effective_duty_cycle(.5,'1'),.5)
        self.assertEqual(effective_duty_cycle(.5,'2'),.2)
        self.assertEqual(effective_duty_cycle(.1,'2'),.1)

    @patch('scripts.run_signal_discovery_guarded.subprocess.run')
    @patch('scripts.run_signal_discovery_guarded.sys.platform','darwin')
    def test_memory_sampler_reports_warning_for_cpu_throttling(self,run):
        def command(args,**kwargs):
            result=self.command(args,**kwargs)
            if args[0]=='sysctl':result.stdout='2\n'
            return result
        run.side_effect=command;state={}
        self.assertEqual(memory_snapshot(-1,384,35,384,state),0)
        self.assertEqual(state['level'],'2')
        self.assertEqual(effective_duty_cycle(.5,state['level']),.2)

    @patch('scripts.run_signal_discovery_guarded.time.sleep')
    def test_transient_headroom_dip_pauses_and_recovers(self,sleep):
        from unittest.mock import Mock
        check=Mock(side_effect=[LowMemoryHeadroom(34,35),LowMemoryHeadroom(33,35),1234])
        pause,swap,stop=Mock(),Mock(),Mock()
        self.assertEqual(wait_for_headroom(check,pause,swap,stop),1234)
        pause.assert_called_once();self.assertEqual(swap.call_count,2);self.assertEqual(stop.call_count,2)

    def test_pausing_never_masks_hard_memory_failures(self):
        from unittest.mock import Mock
        for failure in [LowMemoryHeadroom(24,35),MemoryError('critical pressure'),MemoryError('RSS')]:
            pause=Mock()
            with self.assertRaises(MemoryError):wait_for_headroom(Mock(side_effect=failure),pause,Mock(),Mock())
            pause.assert_not_called()

    @patch('scripts.run_signal_discovery_guarded.time.sleep')
    def test_paused_worker_still_checks_swap_growth_and_stop(self,sleep):
        from unittest.mock import Mock
        for index,failure in [(0,MemoryError('swap growth')),(1,InterruptedError('STOP file'))]:
            controls=[Mock(),Mock()];controls[index].side_effect=failure
            with self.assertRaises(type(failure)):
                wait_for_headroom(Mock(side_effect=LowMemoryHeadroom(34,35)),Mock(),*controls)

    def command(self,args,**kwargs):
        text='10 1 400000\n11 10 400000\n99 1 900000\n' if args[0]=='ps' else 'System-wide memory free percentage: 42%' if args[0]=='/usr/bin/memory_pressure' else '1\n'
        return subprocess.CompletedProcess(args,0,stdout=text,stderr='')

    @patch('scripts.run_signal_discovery_guarded.subprocess.run')
    @patch('scripts.run_signal_discovery_guarded.sys.platform','darwin')
    def test_worker_tree_includes_children_without_other_apps(self,run):
        run.side_effect=self.command
        self.assertEqual(memory_snapshot(10,1024,25),800000)
        with self.assertRaisesRegex(MemoryError,'Worker tree'):memory_snapshot(10,768,25)

    @patch('scripts.run_signal_discovery_guarded.subprocess.run')
    @patch('scripts.run_signal_discovery_guarded.sys.platform','darwin')
    def test_memory_warning_stops_before_loading(self,run):
        def command(args,**kwargs):
            result=self.command(args,**kwargs)
            if args[0]=='sysctl':result.stdout='2\n'
            return result
        run.side_effect=command
        with self.assertRaisesRegex(MemoryError,'pressure level 2'):memory_snapshot(-1,768,25)

    @patch('scripts.run_signal_discovery_guarded.subprocess.run')
    @patch('scripts.run_signal_discovery_guarded.sys.platform','darwin')
    def test_low_system_headroom_stops(self,run):
        def command(args,**kwargs):
            result=self.command(args,**kwargs)
            if args[0]=='/usr/bin/memory_pressure':result.stdout='System-wide memory free percentage: 19%'
            return result
        run.side_effect=command
        with self.assertRaisesRegex(MemoryError,'below 25%'):memory_snapshot(-1,768,25)

    @patch('scripts.run_signal_discovery_guarded.subprocess.run')
    @patch('scripts.run_signal_discovery_guarded.sys.platform','darwin')
    def test_warning_mode_limits_small_worker_and_rejects_critical(self,run):
        def command(args,**kwargs):
            result=self.command(args,**kwargs)
            if args[0]=='sysctl':result.stdout='2\n'
            return result
        run.side_effect=command
        self.assertEqual(memory_snapshot(-1,384,35,384),0)
        with self.assertRaisesRegex(MemoryError,'Warning-mode worker'):memory_snapshot(10,1024,35,384)
        def critical(args,**kwargs):
            result=command(args,**kwargs)
            if args[0]=='sysctl':result.stdout='4\n'
            return result
        run.side_effect=critical
        with self.assertRaisesRegex(MemoryError,'pressure level 4'):memory_snapshot(-1,384,35,384)

    @patch('scripts.run_signal_discovery_guarded.subprocess.run')
    def test_failed_process_monitor_stops(self,run):
        run.return_value=subprocess.CompletedProcess([],1,stdout='',stderr='denied')
        with self.assertRaisesRegex(RuntimeError,'Cannot monitor'):memory_snapshot(10,768,25)


if __name__=='__main__':unittest.main()
