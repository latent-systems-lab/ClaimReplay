"""Export tables and descriptive curves from explicitly supplied result CSVs.

This offline presentation layer does not rerun policies, infer missing scores,
recompute confidence intervals, or substitute for native-data validation.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from .reproduce import figure

CONDITIONS = ('without_prediction', 'with_prediction', 'webassess')
POLICIES = (
    ('Analytic Random', 'random:analytic-expectation'), ('Seeded Random', 'random'),
    ('Lexical', 'baseline:lexical-checkworthiness-v1'),
    ('Citation Scarcity', 'baseline:citation-scarcity'),
    ('Evidence NLI', 'baseline:evidence-nli-risk-v1'),
    ('AFaCTA', 'external:afacta-verifiability-v1'),
    ('GPT-5.4-mini', 'llm:gpt-5.4-mini-review-priority-v1'),
    ('Oracle (diagnostic)', 'oracle'),
)


def read(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError('Empty result CSV')
    return rows


def number(value):
    result = float(value)
    if not math.isfinite(result):
        raise ValueError('Non-finite result value')
    return result


def escape(value):
    replacements = {'\\': r'\textbackslash{}', '&': r'\&', '%': r'\%',
                    '$': r'\$', '#': r'\#', '_': r'\_', '{': r'\{',
                    '}': r'\}', '~': r'\textasciitilde{}', '^': r'\textasciicircum{}'}
    return ''.join(replacements.get(char, char) for char in str(value))


def table(output, name, headers, rows):
    with (output / (name + '.csv')).open('w', encoding='utf-8', newline='') as stream:
        writer = csv.writer(stream); writer.writerow(headers); writer.writerows(rows)
    lines = [r'\begin{tabular}{' + 'l' * len(headers) + '}', r'\toprule',
             ' & '.join(map(escape, headers)) + r' \\', r'\midrule']
    lines += [' & '.join(map(escape, row)) + r' \\' for row in rows]
    (output / (name + '.tex')).write_text('\n'.join(lines + [r'\bottomrule', r'\end{tabular}']) + '\n', encoding='utf-8')


def unique(rows, **keys):
    matches = [row for row in rows if all(
        number(row[key]) == value if isinstance(value, (int, float)) else row[key] == value
        for key, value in keys.items())]
    if len(matches) != 1:
        raise ValueError(f'Expected exactly one aggregate row for {keys}; got {len(matches)}')
    return matches[0]


def export(output, *, replay_results=None, trec_results=None, priority_results=None,
           controlled_results=(), judge_results=None):
    if output.exists():
        raise ValueError('Choose a new output directory; existing results are never overwritten')
    if priority_results and not trec_results:
        raise ValueError('Priority comparison requires TREC reference results')
    inputs = {k: v for k, v in dict(replay=replay_results, trec=trec_results,
                                   priority=priority_results, judge=judge_results).items() if v}
    inputs.update({f'controlled_{i}': p for i, p in enumerate(controlled_results)})
    if not inputs:
        raise ValueError('Provide at least one result CSV')
    loaded = {key: read(path) for key, path in inputs.items()}
    output.mkdir(parents=True)
    receipt = dict(schema_version='claimreplay.reporting.v1', status='RUNNING',
                   inputs={key: hashlib.sha256(p.read_bytes()).hexdigest() for key, p in inputs.items()},
                   conditions_pooled=False, api_calls=0, model_inference=False,
                   statistical_analysis_recomputed=False)
    try:
        if replay_results:
            rows = loaded['replay']
            table(output, 'replay_summary', ['Policy', 'Task', 'Budget', 'Estimand', 'Recall', 'Selected coverage'],
                  [[r['policy'], r['task'], r['budget'], r['estimand'], r['recall'], r['selected_coverage']] for r in rows])
            for i, task in enumerate(sorted({r['task'] for r in rows})):
                curves = [dict(policy=r['policy'], budget=number(r['budget']),
                               recall=number(r['recall']) if r['recall'] else None) for r in rows if r['task'] == task]
                figure(output / f'replay_curve_{i}.svg', curves, task)
        if trec_results:
            trec = loaded['trec']; priority = loaded.get('priority', [])
            labels = [p for p in POLICIES if not p[1].startswith('llm:') or priority_results]
            result = []
            for label, policy in labels:
                source = priority if policy.startswith('llm:') else trec
                values = []
                for condition in CONDITIONS:
                    rows = [r for r in source if r.get('task', 'review_need_detection') == 'review_need_detection']
                    row = unique(rows, condition=condition, policy=policy, budget=.2)
                    values.append(f"{number(row['topic_macro_recall']):.4f}")
                result.append([label, *values])
            table(output, 'trec_primary', ['Policy', *CONDITIONS], result)
            coverage = []
            for condition in CONDITIONS:
                curves = []
                for label, policy in labels:
                    source = priority if policy.startswith('llm:') else trec
                    subset = [r for r in source if r['condition'] == condition and r['policy'] == policy
                              and r.get('task', 'review_need_detection') == 'review_need_detection']
                    for budget in sorted({number(r['budget']) for r in subset}):
                        r = unique(subset, budget=budget)
                        curves.append(dict(policy=label, budget=budget, recall=number(r['topic_macro_recall'])))
                        # Analytic expectation has no realized selected set.
                        if policy == 'random:analytic-expectation':
                            continue
                        coverage.append([condition, label, budget, r['selected_total'], r['selected_assessed'],
                                         r['selected_unassessed'], r['selected_assessment_coverage'], r['realized_budget']])
                figure(output / (condition + '.svg'), curves, condition)
            table(output, 'selection_coverage', ['Condition', 'Policy', 'Budget', 'Selected', 'Assessed',
                  'Unassessed', 'Selected coverage', 'Realized fraction'], coverage)
        if controlled_results:
            rows = [r for key, values in loaded.items() if key.startswith('controlled_') for r in values]
            policies = ('analytic_random', 'baseline:lexical-checkworthiness-v1', 'transfer:afacta-target-only-v1',
                        'transfer:citation-scarcity-source-documents-v1', 'transfer:nli-first-cited-abstract-v1')
            result = []
            for dataset in ('feverous', 'scifact'):
                for target in ('native_non_support', 'native_refutation'):
                    vals = []
                    for policy in policies:
                        subset = [r for r in rows if r['dataset'] == dataset and r['target'] == target and r['policy'] == policy
                                  and number(r['queue_size']) == 20 and number(r['budget']) == .2]
                        if len(subset) > 1:
                            # Separate evidence runs may repeat the same reference rows.
                            # Coalesce only exact agreement on every shared field.
                            first = subset[0]
                            if any(any(r[k] != first[k] for k in r.keys() & first.keys()) for r in subset[1:]):
                                raise ValueError('Conflicting controlled comparison')
                            receipt['coalesced_controlled_rows'] = receipt.get('coalesced_controlled_rows', 0) + len(subset) - 1
                            subset = subset[:1]
                        # Only the known unsupported evidence-policy/domain combinations are N/A.
                        unavailable = dataset == 'feverous' and policy in policies[-2:]
                        if not subset and not unavailable:
                            raise ValueError(f'Missing controlled comparison: {dataset}/{target}/{policy}')
                        vals.append(f"{number(subset[0]['group_macro_recall']):.4f}" if subset else 'N/A')
                    result.append([dataset, target, *vals])
            table(output, 'controlled', ['Dataset', 'Native target', 'Analytic Random', 'Lexical', 'AFaCTA', 'Citation', 'NLI'], result)
        if judge_results:
            rows = [unique(loaded['judge'], condition=c) for c in CONDITIONS]
            table(output, 'judge_agreement', ['Condition', 'Assessed', 'Abstain', 'Agreement (%)', 'Macro-F1', 'Binary (%)'],
                  [[r['condition'], r['assessed_count'], r['abstentions'], f"{100*number(r['exact_agreement']):.1f}",
                    f"{number(r['macro_f1']):.3f}", f"{100*number(r['binary_agreement']):.1f}"] for r in rows])
        receipt['status'] = 'PASS'
    except Exception as exc:
        receipt.update(status='FAIL', error=str(exc))
        raise
    finally:
        receipt['outputs'] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(output.iterdir()) if p.name != 'receipt.json'}
        (output / 'receipt.json').write_text(json.dumps(receipt, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ('replay-results', 'trec-results', 'priority-results', 'judge-results'):
        parser.add_argument('--' + flag, type=Path)
    parser.add_argument('--controlled-results', type=Path, action='append', default=[])
    parser.add_argument('--output', type=Path, required=True)
    print(json.dumps(export(**vars(parser.parse_args())), indent=2))


if __name__ == '__main__':
    main()
