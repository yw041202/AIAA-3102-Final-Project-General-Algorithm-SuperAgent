"""Replay the review reconciliation from archived evidence; no network or writes.

Run from any directory with Python 3.11. Requires only the standard library.
"""
import json
from decimal import Decimal
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
ARCHIVE = ROOT / 'parta/evidence/released-20261122'


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def total(entries):
    return {
        'calls': len(entries),
        'settled': sum(e['status'] == 'settled' for e in entries),
        'uncertain': sum(e['status'] == 'uncertain' for e in entries),
        'spent_rmb': str(sum((Decimal(e['cost_rmb'] or '0') for e in entries), Decimal(0))),
        'held_reserved_rmb': str(sum((Decimal(e['reserved_rmb']) for e in entries), Decimal(0))),
    }


def verify():
    baseline = read(ARCHIVE / 'task-budget-ledger.json')['tasks']
    before = read(HERE / 'ledger-before-feedback.json')['tasks']
    after = read(HERE / 'ledger-after-feedback.json')['tasks']
    compatibility = read(ROOT / 'review-evidence/fixes-20261126/ledger-compatibility.json')['released']
    logs = {}
    for folder in (ARCHIVE, HERE / 'audit26', HERE / 'headlines-feedback27'):
        for path in folder.rglob('llm.jsonl'):
            for line in path.read_text(encoding='utf-8').splitlines():
                row = json.loads(line)
                assert row['call_id'] not in logs, 'Duplicate call ID in evidence logs'
                logs[row['call_id']] = (row, path.relative_to(ROOT).as_posix())
    report = {'checks': {}, 'tasks': {}, 'calls': []}
    all_ids = set()
    for task in ('coupon', 'headlines'):
        old, mid, new = (snapshot[task]['entries'] for snapshot in (baseline, before, after))
        # These snapshots must be strict append-only extensions, including every
        # historical uncertain reservation. Comparison includes all entry fields.
        assert mid[:len(old)] == old
        assert new[:len(mid)] == mid
        for snapshot in (before, after):
            for key in ('task_id', 'budget_rmb', 'created_at'):
                assert snapshot[task][key] == baseline[task][key]
        middle_total = total(mid)
        for key, value in middle_total.items():
            expected = compatibility[task][key]
            assert Decimal(value) == Decimal(str(expected))
        sections = {'nov22': old, 'nov26_additions': mid[len(old):],
                    'nov26_cumulative': mid, 'nov27_additions': new[len(mid):],
                    'current_cumulative': new}
        report['tasks'][task] = {name: total(entries) for name, entries in sections.items()}
        running = Decimal(0)
        for entry in new:
            call_id = entry['call_id']
            assert call_id not in all_ids
            all_ids.add(call_id)
            row, source = logs[call_id]
            assert row['task_id'] == task
            assert row['pricing_version'] == entry['pricing_version'] == 'course_off_peak_20261021'
            assert row['requested_model_id'] == entry['model'] == 'DeepSeek-V4.1-Flash'
            assert row['prompt_tokens'] == entry['input_tokens']
            assert row['completion_tokens'] == entry['output_tokens']
            if entry['status'] == 'settled':
                assert row['request_status'] == 'success'
                # Fixed course mapping in these records: 1/4 RMB per million
                # input/output tokens. This is not a claim about current pricing.
                recomputed = (Decimal(entry['input_tokens']) + 4 * Decimal(entry['output_tokens'])) / 1000000
                assert recomputed == Decimal(entry['cost_rmb']) == Decimal(row['calculated_cost'])
                running += recomputed
                assert running == Decimal(row['cumulative_cost'])
            else:
                assert entry['status'] == 'uncertain'
                assert row['request_status'] == 'error'
                assert row['error_type'] == entry['error_type'] == 'server_error'
                assert row['calculated_cost'] is None and entry['cost_rmb'] is None
                assert Decimal(entry['reserved_rmb']) > 0
            report['calls'].append({'task_id': task, 'run_id': row['run_id'],
                                    'source_log': source, **entry})
        assert {logs[e['call_id']][0]['run_id'] for e in mid[len(old):]} == {'audit26'}
        if task == 'coupon':
            assert new == mid
        else:
            assert {logs[e['call_id']][0]['run_id'] for e in new[len(mid):]} == {'feedback27'}
    assert all_ids == set(logs), 'Every ledger entry and every log must match exactly once'
    report['checks'] = {'historical_entries_unchanged': True,
                        'nov26_compatibility_totals_match': True,
                        'one_to_one_call_id_log_match': True,
                        'settled_token_costs_recomputed': True,
                        'uncertain_reservations_preserved': True}
    report['external_unverified'] = {
        'reporter': 'B / azhao258', 'task_id': 'headlines', 'reported_calls': 3,
        'reported_cost_rmb': '0.003537', 'evidence_available': False,
        'included_in_verified_ledger': False,
        'source': 'https://github.com/yw041202/AIAA-3102-Final-Project-General-Algorithm-SuperAgent/pull/3#issuecomment-7212137232',
        'note': 'User confirmed no raw log available on 2026-11-27. Do not invent call IDs, tokens or ledger entries.',
    }
    return report


if __name__ == '__main__':
    print(json.dumps(verify(), ensure_ascii=True, indent=2))
