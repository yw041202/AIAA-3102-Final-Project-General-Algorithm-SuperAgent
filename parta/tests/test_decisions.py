import copy
import json
from pathlib import Path

import pytest

from parta.agent import LLMPlanner, load_task, run
from parta.decisions import DecisionValidationError, decision_contract, parse_decision
from parta.demo import make_fixture
from partc.llm_client import MockBackend


def example():
    return decision_contract('text')['experiment_example']


@pytest.mark.parametrize('options,field,hint', [
    ({'text': {'analyzer': 'word', 'min_df': 2}}, 'text', 'remove this wrapper'),
    ({'text': {'analyzer': 'word', 'mindf': 2}}, 'text', 'preprocess_options.text.mindf: use min_df'),
    ({'mindf': 2}, 'mindf', 'use min_df'),
])
def test_b_reported_fields_get_actionable_feedback(options, field, hint):
    decision = example(); decision['preprocess_options'] = options
    before = copy.deepcopy(decision)
    with pytest.raises(DecisionValidationError) as caught:
        parse_decision(json.dumps(decision), 'text')
    feedback = caught.value.feedback
    assert feedback['path'] == 'preprocess_options'
    assert field in feedback['unknown_fields']
    assert 'min_df' in feedback['allowed_fields']
    assert hint in ' '.join(feedback['hints'])
    assert decision == before


@pytest.mark.parametrize('text', [
    '{"action":"stop","reason":None}',
    '{"action":"stop","reason":True}',
    "{'action':'stop','reason':'done'}",
    '```json\n{"action":"stop","reason":"done"}\n```',
    '{"action":"stop","reason":"done",}',
])
def test_python_literals_and_non_json_output_rejected(text):
    with pytest.raises(DecisionValidationError) as caught: parse_decision(text, 'text')
    assert caught.value.feedback['code'] == 'invalid_json'
    assert 'null/true/false' in caught.value.feedback['message']


@pytest.mark.parametrize('token', ['NaN', 'Infinity', '-Infinity', '1e999'])
def test_nonfinite_json_numbers_rejected(token):
    text = json.dumps(example()).replace('"min_df": 2', '"min_df": ' + token)
    with pytest.raises(DecisionValidationError): parse_decision(text, 'text')


def test_duplicate_keys_rejected():
    with pytest.raises(DecisionValidationError) as caught:
        parse_decision('{"action":"stop","action":"experiment","reason":"done"}', 'text')
    assert caught.value.feedback['code'] == 'duplicate_key'


@pytest.mark.parametrize('modality', ['text', 'tabular'])
def test_prompt_examples_obey_actual_contract(modality):
    contract = decision_contract(modality)
    for name in ('experiment_example', 'stop_example'):
        assert parse_decision(json.dumps(contract[name]), modality) == contract[name]


def test_json_null_accepted_and_reason_not_rewritten():
    d = example(); d['reason'] = 'None is a word in this explanation.'
    assert parse_decision(json.dumps(d), 'text') == d


def test_contract_is_sent_through_actual_c_client(tmp_path_factory, monkeypatch):
    folder = tmp_path_factory.mktemp('contract')
    config = load_task(make_fixture(folder / 'd', 'text'))
    backend = MockBackend(content=json.dumps({'action': 'stop', 'reason': 'done'}))
    monkeypatch.setattr('partc.llm_client.SDKBackend', lambda cfg: backend)
    planner = LLMPlanner(config, folder / 'run', str(folder / 'absent.env'))
    planner.propose({'data': {'modality': 'text'}, 'history': []})
    prompt = backend.calls[0]['messages'][0]['content']
    contract = json.loads(prompt.split('Decision contract (examples show syntax, not a mandatory plan):\n')[1])
    assert contract['modality'] == 'text'
    assert 'null' in prompt and 'do not nest text' in prompt
    parse_decision(json.dumps(contract['experiment_example']), 'text')


def test_invalid_text_plan_feedback_reaches_next_round_before_real_training(tmp_path_factory):
    folder = tmp_path_factory.mktemp('repair')
    config = load_task(make_fixture(folder / 'd', 'text'))
    config['max_steps'] = 3
    class RepairPlanner:
        def __init__(self): self.contexts = []
        def propose(self, context):
            self.contexts.append(copy.deepcopy(context))
            if len(self.contexts) == 1:
                d = example(); d['preprocess_options'] = {'text': {'mindf': 1}}
            elif len(self.contexts) == 2:
                feedback = context['history'][-1]['validation_error']
                assert feedback['unknown_fields'] == ['text']
                assert 'min_df' in feedback['allowed_fields']
                assert 'preprocess_options.text.mindf: use min_df' in ' '.join(feedback['hints'])
                d = example(); d['preprocess_options']['min_df'] = 1
            else:
                d = {'action': 'stop', 'reason': 'repair verified'}
            return {'request_status': 'success', 'output_text': json.dumps(d)}
        def summary(self): return {'offline': True, 'live_calls': 0}
    planner = RepairPlanner()
    result = run(config, planner=planner, offline=True)
    assert result['status'] == 'success', result
    assert result['selected']['experiment_id'] == 'experiment_002'
    directory = Path(result['run_directory'])
    assert not (directory / 'requests/experiment_001.json').exists()
    assert (directory / 'requests/experiment_002.json').is_file()
    assert (directory / 'predictions.csv').is_file()
    assert 'validation_error' in (directory / 'decisions.jsonl').read_text(encoding='utf-8')
