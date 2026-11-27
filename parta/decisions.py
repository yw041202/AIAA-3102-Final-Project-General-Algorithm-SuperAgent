"""One decision contract shared by the planner prompt and strict validation."""
import json
from difflib import get_close_matches

MODELS = {
    'logistic_regression': {'C', 'max_iter', 'class_weight', 'solver'},
    'random_forest': {'n_estimators', 'max_depth', 'min_samples_leaf', 'max_features', 'class_weight', 'n_jobs'},
}
PREPROCESS = {
    'tabular': {'encoding', 'numeric_columns'},
    'text': {'analyzer', 'ngram_range', 'max_features', 'min_df', 'lowercase', 'combine'},
}


def decision_contract(modality):
    options = ({'encoding': 'onehot', 'numeric_columns': []} if modality == 'tabular' else
               {'analyzer': 'word', 'ngram_range': [1, 2], 'max_features': 50000,
                'min_df': 2, 'lowercase': True, 'combine': 'concat'})
    return {
        'format': 'Exactly one strict JSON object. Double-quoted keys/strings; null, true, false. '
                  'Never Python None/True/False, NaN, Infinity, comments, duplicate keys or markdown fences.',
        'modality': modality,
        'model_parameters': {name: sorted(keys) for name, keys in MODELS.items()},
        'preprocess_fields': sorted(PREPROCESS[modality]),
        'structure': 'preprocess_options is a flat object. Put its parameters directly there; '
                     'do not nest text, tabular, tfidf or other wrappers. Spell min_df with an underscore.',
        'experiment_example': {'action': 'experiment', 'reason': 'Illustrative syntax only; choose from observations and feedback.',
                               'model': {'name': 'logistic_regression', 'params': {'C': 1.0, 'max_iter': 1000, 'class_weight': None}},
                               'preprocess_options': options},
        'stop_example': {'action': 'stop', 'reason': 'Explain why to stop.'},
    }


class DecisionValidationError(ValueError):
    def __init__(self, code, path, message, **details):
        self.feedback = {'code': code, 'path': path, 'message': message, **details}
        super().__init__(json.dumps(self.feedback, ensure_ascii=False, allow_nan=False))


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise DecisionValidationError('duplicate_key', '$', 'Use each JSON key once.', field=key)
        result[key] = value
    return result


def _constant(value):
    raise DecisionValidationError('invalid_json_constant', '$', 'NaN and Infinity are not valid JSON values.')


def _fields(value, allowed, required, path):
    if not isinstance(value, dict):
        raise DecisionValidationError('wrong_type', path, 'Expected a JSON object.', allowed_fields=sorted(allowed))
    unknown, missing = sorted(set(value) - allowed), sorted(required - set(value))
    if unknown or missing:
        hints = []
        for name in unknown[:10]:
            match = get_close_matches(name, allowed, n=1, cutoff=0.7)
            if match:
                hints.append(f'{path}.{name}: use {match[0]} if that was intended.')
            if path == 'preprocess_options' and isinstance(value[name], dict):
                hints.append(f'{path}.{name}: remove this wrapper; put supported parameters directly in preprocess_options.')
                for child in sorted(value[name])[:10]:
                    match = get_close_matches(child, allowed, n=1, cutoff=0.7)
                    if child not in allowed and match:
                        hints.append(f'{path}.{name}.{child}: use {match[0]} if that was intended.')
        raise DecisionValidationError('invalid_fields', path, 'Correct the field names and object structure; do not repeat the invalid object.',
                                      unknown_fields=unknown, missing_fields=missing,
                                      allowed_fields=sorted(allowed), hints=hints)


def parse_decision(text, modality):
    if modality not in PREPROCESS:
        raise DecisionValidationError('unsupported_modality', '$', 'Only tabular/text planning is supported.')
    try:
        decision = json.loads(text, parse_constant=_constant, object_pairs_hook=_object)
    except json.JSONDecodeError as exc:
        raise DecisionValidationError('invalid_json', '$',
            'Return strict JSON only: double quotes, null/true/false instead of None/True/False; no comments or markdown.',
            line=exc.lineno, column=exc.colno, parser_message=exc.msg) from None
    if not isinstance(decision, dict):
        raise DecisionValidationError('wrong_type', '$', 'Expected one JSON object.')
    action = decision.get('action')
    if action not in ('experiment', 'stop'):
        raise DecisionValidationError('invalid_action', 'action', 'Use experiment or stop.')
    fields = {'action', 'reason'} if action == 'stop' else {'action', 'reason', 'model', 'preprocess_options'}
    _fields(decision, fields, fields, '$')
    if not isinstance(decision['reason'], str) or not decision['reason'].strip():
        raise DecisionValidationError('invalid_reason', 'reason', 'Provide a nonempty explanation string.')
    if action == 'stop':
        return decision
    model = decision['model']
    _fields(model, {'name', 'params'}, {'name', 'params'}, 'model')
    if not isinstance(model['name'], str) or model['name'] not in MODELS:
        raise DecisionValidationError('unsupported_model', 'model.name', 'Choose a supported model.', allowed_models=sorted(MODELS))
    _fields(model['params'], MODELS[model['name']], set(), 'model.params')
    _fields(decision['preprocess_options'], PREPROCESS[modality], set(), 'preprocess_options')
    # Parameter values are still checked by B/D. Invalid decisions are never
    # silently repaired, flattened or executed by this parser.
    try:
        json.dumps(decision, allow_nan=False)
    except ValueError:
        raise DecisionValidationError('nonfinite_value', '$', 'All numeric values must be finite.') from None
    return decision
