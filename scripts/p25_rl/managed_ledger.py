"""Completed-close observation before the frozen engine requests management.

The original engine and the entry-only observer module remain unchanged. This
module clones the original simulation AST, adds one optional argument, and
observes the current close immediately before per-cohort management requests.
An observer can therefore store a global hold/trim/close instruction that the
existing management callback translates into original next-session orders.

All observer inputs are detached snapshots. Holding records preserve original
entry and management fields and include the same causal current ``quote`` and
``spot`` that the original cohort management callback receives. Financial
calculations, elective-fill checks and order timing are preserved verbatim.
"""
from __future__ import annotations

import ast
from copy import deepcopy
from hashlib import sha256
import inspect

from scripts import uniform_smile_engine as _engine
from scripts.uniform_smile_risk import book_risk as _cached_book_risk


_HOOK = """if session_observer is not None:
    session_observer(date, _rl_copy_snapshot(_rl_original_day),
                     _rl_positions_snapshot(positions, quote, spots),
                     _rl_copy_snapshot(lag_pairs), _rl_copy_snapshot(lag_global))
"""


def _position_snapshots(positions, quote, spots):
    """Detached holdings with the original current-close management context."""
    return [deepcopy(dict(position,
                          quote=dict(quote(position['optionid'], position)),
                          spot=spots[position['secid']]))
            for position in positions]


def _instrument_function(source):
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
    if len(appends) != 1 or len(appends[0].value.args) != 1:
        raise ValueError('Expected one original daily ledger dictionary')
    assignments = [index for index, node in enumerate(session_loop.body)
                   if isinstance(node, ast.Assign)
                   and any(isinstance(target, ast.Name) and target.id == 'lag_global'
                           for target in node.targets)]
    if len(assignments) != 1:
        raise ValueError('Expected exactly one completed-close global lag state')
    insertion = assignments[0] + 1
    expected_test = ast.parse('management_policy is not None', mode='eval').body
    following = session_loop.body[insertion]
    if (not isinstance(following, ast.If)
            or ast.dump(following.test) != ast.dump(expected_test)):
        raise ValueError('Management requests no longer follow the lag state')

    # Evaluate the exact original day expression at the same close, before the
    # management callback records future orders. This adds no alternate ledger.
    hook = ast.parse(_HOOK).body[0]
    for call in [node for node in ast.walk(hook) if isinstance(node, ast.Call)]:
        for index, argument in enumerate(call.args):
            if isinstance(argument, ast.Name) and argument.id == '_rl_original_day':
                call.args[index] = deepcopy(appends[0].value.args[0])
    function.args.args.append(ast.arg(arg='session_observer'))
    function.args.defaults.append(ast.Constant(value=None))
    session_loop.body.insert(insertion, hook)

    restored = deepcopy(function)
    restored.args.args.pop()
    restored.args.defaults.pop()
    restored_loop = next(node for node in restored.body if isinstance(node, ast.For)
                         and ast.dump(node.iter) == ast.dump(session_loop.iter))
    restored_loop.body.pop(insertion)
    original_dump = ast.dump(original, include_attributes=False)
    if ast.dump(restored, include_attributes=False) != original_dump:
        raise AssertionError('Management observer changed the financial engine AST')
    info = dict(
        engine_source_sha256=sha256(source.encode('utf-8')).hexdigest(),
        accounting_ast_sha256=sha256(original_dump.encode('utf-8')).hexdigest(),
        accounting_ast_unchanged=True,
        observer_timing='completed close after lag book state, before management requests',
        day_snapshot='original daily.append dictionary expression',
        positions_snapshot='independent copies plus original current causal quote and spot',
        book_risk='scripts.uniform_smile_risk.book_risk')
    return ast.fix_missing_locations(function), info


def _build_simulator():
    function, info = _instrument_function(inspect.getsource(_engine.simulate))
    namespace = dict(vars(_engine))
    namespace['book_risk'] = _cached_book_risk
    namespace['_rl_copy_snapshot'] = deepcopy
    namespace['_rl_positions_snapshot'] = _position_snapshots
    module = ast.Module(body=[function], type_ignores=[])
    filename = inspect.getsourcefile(_engine.simulate) or '<uniform_smile_engine>'
    exec(compile(ast.fix_missing_locations(module), filename, 'exec'), namespace)
    simulator = namespace['simulate']
    simulator.__name__ = 'simulate_with_management_observer'
    simulator.__qualname__ = 'simulate_with_management_observer'
    simulator.__module__ = __name__
    return simulator, info


simulate_with_management_observer, INSTRUMENTATION = _build_simulator()


def instrumentation_info():
    """Return detached source/AST fingerprints for the experiment manifest."""
    return deepcopy(INSTRUMENTATION)
