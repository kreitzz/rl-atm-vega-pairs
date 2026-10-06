import subprocess
import unittest
from unittest.mock import patch

from scripts.run_signal_discovery_guarded import memory_snapshot,LowMemoryHeadroom,wait_for_headroom


class MemoryGuardTest(unittest.TestCase):
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
