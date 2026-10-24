import csv
import json
from pathlib import Path
import pytest
from claimreplay.reporting import export, POLICIES, CONDITIONS
from claimreplay.reproduce import run


def write(path, rows):
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    return path


def test_reporting_on_reproduction(tmp_path):
    root = Path(__file__).resolve().parents[1]
    run(root/'configs/toy_reproduction.json', root, tmp_path/'run', 'deterministic')
    source = tmp_path/'run/synthetic/results.csv'
    first = export(tmp_path/'first', replay_results=source)
    repeat = export(tmp_path/'repeat', replay_results=source)
    assert first['status'] == 'PASS' and first['outputs'] == repeat['outputs']
    assert first['api_calls'] == 0 and not first['statistical_analysis_recomputed']
    assert str(tmp_path) not in json.dumps(first)
    with pytest.raises(ValueError, match='new output'):
        export(tmp_path/'first', replay_results=source)


def trec_rows():
    return [dict(condition=c, policy=p, budget=b, topic_macro_recall=.25,
                 selected_total=4, selected_assessed=3, selected_unassessed=1,
                 selected_assessment_coverage=.75, realized_budget=.25)
            for c in CONDITIONS for _, p in POLICIES if not p.startswith('llm:') for b in (.05, .1, .2)]


def test_separate_conditions_and_coverage(tmp_path):
    source = write(tmp_path/'trec.csv', trec_rows())
    receipt = export(tmp_path/'out', trec_results=source)
    assert receipt['status'] == 'PASS' and not receipt['conditions_pooled']
    for condition in CONDITIONS:
        assert (tmp_path/'out'/f'{condition}.svg').is_file()
    with (tmp_path/'out/trec_primary.csv').open() as stream:
        rows = list(csv.reader(stream))
    assert rows[0] == ['Policy', *CONDITIONS] and len(rows) == 8
    with (tmp_path/'out/selection_coverage.csv').open() as stream:
        coverage = list(csv.DictReader(stream))
    assert len(coverage) == 54 and all(r['Unassessed'] == '1' for r in coverage)


@pytest.mark.parametrize('mutation', ['duplicate', 'missing', 'nan'])
def test_rejects_ambiguous_or_invalid_results(tmp_path, mutation):
    rows = trec_rows()
    if mutation == 'duplicate': rows.append(rows[-1].copy())
    elif mutation == 'missing': rows.pop()
    else: rows[-1]['topic_macro_recall'] = 'nan'
    source = write(tmp_path/'trec.csv', rows)
    with pytest.raises(ValueError): export(tmp_path/'out', trec_results=source)
    assert json.loads((tmp_path/'out/receipt.json').read_text())['status'] == 'FAIL'


def test_priority_requires_reference_and_no_empty_export(tmp_path):
    with pytest.raises(ValueError): export(tmp_path/'out', priority_results=tmp_path/'absent')
    with pytest.raises(ValueError): export(tmp_path/'out')
    assert not (tmp_path/'out').exists()


def test_controlled_repeated_references_must_agree(tmp_path):
    policies = ('analytic_random', 'baseline:lexical-checkworthiness-v1', 'transfer:afacta-target-only-v1',
                'transfer:citation-scarcity-source-documents-v1', 'transfer:nli-first-cited-abstract-v1')
    rows = [dict(dataset=d, target=t, queue_size=20, budget=.2, policy=p, group_macro_recall=.2)
            for d in ('feverous', 'scifact') for t in ('native_non_support', 'native_refutation')
            for p in policies if not (d == 'feverous' and p in policies[-2:])]
    source = write(tmp_path/'controlled.csv', rows)
    receipt = export(tmp_path/'same', controlled_results=[source, source])
    assert receipt['status'] == 'PASS' and receipt['coalesced_controlled_rows'] == len(rows)
    altered = [r.copy() for r in rows]; altered[0]['group_macro_recall'] = .3
    other = write(tmp_path/'conflict.csv', altered)
    with pytest.raises(ValueError, match='Conflicting'):
        export(tmp_path/'bad', controlled_results=[source, other])
