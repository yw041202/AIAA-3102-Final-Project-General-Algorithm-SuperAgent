"""Unreadable accounting must stop before any model request or state rewrite."""
import builtins
import json
from decimal import Decimal

import pytest

from partc import BudgetManager, BudgetStateError


def valid_ledger(path):
    manager = BudgetManager(str(path), '20')
    manager.reserve('task', 'paid', 'model', 'rate', Decimal('3'))
    manager.settle('task', 'paid', 10, 10, Decimal('2'))
    manager.reserve('task', 'unknown', 'model', 'rate', Decimal('1'))
    manager.mark_uncertain('task', 'unknown', error_type='network_error')
    return json.loads(path.read_text())


@pytest.mark.parametrize('payload', [b'{truncated', b'', b'\xff', b'{}', b'[]',
    b'{"version":1,"version":1,"tasks":{}}', b'{"version":2,"tasks":{}}'])
def test_corrupt_ledger_is_preserved(tmp_path, payload):
    path = tmp_path / 'ledger.json'
    path.write_bytes(payload)
    with pytest.raises(BudgetStateError):
        BudgetManager(str(path))
    assert path.read_bytes() == payload


@pytest.mark.parametrize('damage', ['missing_entries', 'missing_budget', 'wrong_task_id',
    'unknown_status', 'missing_cost', 'missing_reservation', 'negative', 'nan',
    'duplicate_call', 'fractional_tokens', 'settled_without_tokens'])
def test_accounting_damage_does_not_reset_balance(tmp_path, damage):
    path = tmp_path / 'ledger.json'
    data = valid_ledger(path)
    task = data['tasks']['task']; entry = task['entries'][0]
    if damage == 'missing_entries': del task['entries']
    elif damage == 'missing_budget': del task['budget_rmb']
    elif damage == 'wrong_task_id': task['task_id'] = 'other'
    elif damage == 'unknown_status': entry['status'] = 'ignored'
    elif damage == 'missing_cost': entry['cost_rmb'] = None
    elif damage == 'missing_reservation': del task['entries'][1]['reserved_rmb']
    elif damage == 'negative': entry['cost_rmb'] = '-2'
    elif damage == 'nan': task['budget_rmb'] = 'NaN'
    elif damage == 'duplicate_call': task['entries'].append(entry.copy())
    elif damage == 'fractional_tokens': entry['input_tokens'] = 1.5
    elif damage == 'settled_without_tokens': entry['output_tokens'] = None
    path.write_text(json.dumps(data), encoding='utf-8')
    before = path.read_bytes()
    with pytest.raises(BudgetStateError): BudgetManager(str(path))
    assert path.read_bytes() == before


def test_permission_failure_is_not_new_ledger(tmp_path, monkeypatch):
    path = tmp_path / 'ledger.json'
    valid_ledger(path); before = path.read_bytes()
    original = builtins.open
    def denied(file, *args, **kwargs):
        if str(file) == str(path): raise PermissionError('denied')
        return original(file, *args, **kwargs)
    with monkeypatch.context() as patch:
        patch.setattr(builtins, 'open', denied)
        with pytest.raises(BudgetStateError): BudgetManager(str(path))
    assert path.read_bytes() == before


def test_directory_is_not_new_ledger(tmp_path):
    with pytest.raises(BudgetStateError): BudgetManager(str(tmp_path))
    assert tmp_path.is_dir()


def test_valid_restart_retains_paid_and_uncertain_costs(tmp_path):
    path = tmp_path / 'ledger.json'
    valid_ledger(path); before = path.read_bytes()
    summary = BudgetManager(str(path)).summary('task')
    assert summary['spent_rmb'] == '2'
    assert summary['held_reserved_rmb'] == '1'
    assert summary['available_rmb'] == '17'
    assert path.read_bytes() == before
