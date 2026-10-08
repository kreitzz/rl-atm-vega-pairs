"""Observe completed portfolio sessions without changing the frozen ledger.

The original engine remains untouched.  Its ``simulate`` function is cloned
from its parsed source, with one optional argument and a detached-snapshot
callback appended to the session loop.  A structural check removes those two
additions and requires exact equality with the original function's AST.

Observers receive the *completed* signal-session close.  A policy closure may
store the next entry decision there; the engine will still fill it on the next
session and apply its original contract and portfolio risk checks.  Snapshot
objects are independent deep copies, so edits by an observer cannot alter the
financial state.  Callback errors propagate rather than losing observations.
"""
from __future__ import annotations

import ast
from copy import deepcopy
from hashlib import sha256
import inspect

from scripts import uniform_smile_engine as _engine
from scripts.uniform_smile_risk import book_risk as _cached_book_risk


_HOOK = """if session_observer is not None:
    session_observer(date, _rl_copy_snapshot(daily[-1]),
                     _rl_copy_snapshot(positions), _rl_copy_snapshot(lag_pairs),
                     _rl_copy_snapshot(lag_global))
"""


def _instrument_function(source):
    """Return the checked function AST and auditable source fingerprints."""
    parsed = ast.parse(source)
    functions = [node for node in parsed.body if isinstance(node, ast.FunctionDef)]
    if len(functions) != 1 or functions[0].name != 'simulate':
        raise ValueError('Expected exactly the frozen simulate function')
    original = functions[0]
    function = deepcopy(original)
    if any(arg.arg == 'session_observer' for arg in function.args.args):
        raise ValueError('The frozen engine already declares session_observer')
    loops = [node for node in function.body if isinstance(node, ast.For)
             and isinstance(node.iter, ast.Call)
             and isinstance(node.iter.func, ast.Name)
             and node.iter.func.id == 'enumerate'
             and len(node.iter.args) == 1
             and isinstance(node.iter.args[0], ast.Name)
             and node.iter.args[0].id == 'cal']
    if len(loops) != 1:
        raise ValueError('Expected exactly one calendar session loop')
    session_loop = loops[0]
    appends = [node for node in session_loop.body if isinstance(node, ast.Expr)
               and isinstance(node.value, ast.Call)
               and isinstance(node.value.func, ast.Attribute)
               and isinstance(node.value.func.value, ast.Name)
               and node.value.func.value.id == 'daily'
               and node.value.func.attr == 'append']
    if len(appends) != 1:
        raise ValueError('Expected exactly one completed daily ledger append')

    function.args.args.append(ast.arg(arg='session_observer'))
    function.args.defaults.append(ast.Constant(value=None))
    session_loop.body.append(ast.parse(_HOOK).body[0])

    # Fail closed if instrumentation ever changes an existing accounting node.
    restored = deepcopy(function)
    restored.args.args.pop()
    restored.args.defaults.pop()
    restored_loop = next(node for node in restored.body if isinstance(node, ast.For)
                         and ast.dump(node.iter) == ast.dump(session_loop.iter))
    restored_loop.body.pop()
    original_dump = ast.dump(original, include_attributes=False)
    if ast.dump(restored, include_attributes=False) != original_dump:
        raise AssertionError('Observer instrumentation altered the ledger AST')
    info = {
        'engine_source_sha256': sha256(source.encode('utf-8')).hexdigest(),
        'accounting_ast_sha256': sha256(original_dump.encode('utf-8')).hexdigest(),
        'accounting_ast_unchanged': True,
        'observer_timing': 'completed session after daily ledger and lag state',
        'snapshots': 'independent deep copies',
        'book_risk': 'scripts.uniform_smile_risk.book_risk',
    }
    return ast.fix_missing_locations(function), info


def _build_simulator():
    source = inspect.getsource(_engine.simulate)
    function, info = _instrument_function(source)
    namespace = dict(vars(_engine))
    namespace['book_risk'] = _cached_book_risk
    namespace['_rl_copy_snapshot'] = deepcopy
    module = ast.Module(body=[function], type_ignores=[])
    filename = inspect.getsourcefile(_engine.simulate) or '<uniform_smile_engine>'
    exec(compile(ast.fix_missing_locations(module), filename, 'exec'), namespace)
    simulator = namespace['simulate']
    simulator.__name__ = 'simulate_with_observer'
    simulator.__qualname__ = 'simulate_with_observer'
    simulator.__module__ = __name__
    return simulator, info


simulate_with_observer, INSTRUMENTATION = _build_simulator()


def instrumentation_info():
    """Return detached provenance suitable for a saved experiment manifest."""
    return deepcopy(INSTRUMENTATION)
