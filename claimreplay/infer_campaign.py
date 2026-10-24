"""Explicit fresh-inference entry point. Default backend is mock, never paid.

Uses existing frozen request builders and scoring implementations. Fresh outputs
are a new replication, not a promise of historical numerical identity.
"""
from __future__ import annotations

import argparse
import json
import os
import math
from pathlib import Path
import time

from .evidence import EvidenceTextCache, materialize_response_evidence
from .llm import (ProviderConfiguration, MockStructuredProvider, OpenAIResponsesProvider,
                  StructuredResponseCache, ExecutionLimits, execute_structured_requests,
                  content_hash, estimate_request_tokens)
from .llm_evaluation import priority_request, judge_request, validate_priority_output, validate_judge_output
from .models import SupportLevel
from .provenance import verify, save
from .reproduce import ingest, summarize, csv_write, figure
from .replay import replay
from .policies import VerificationPolicy, RankedUnit, TriageCost


MODEL="gpt-5.4-mini-2026-03-17"


def authorized(backend, *, paid=False, local=False):
    if backend=="openai" and not (paid and os.environ.get("CLAIMREPLAY_API_BILLING_CONFIRMED")=="1" and os.environ.get("OPENAI_API_KEY"," ").strip()):
        raise ValueError("Paid inference blocked: explicit --authorize-paid-inference, billing confirmation and credential are all required")
    if backend=="local" and not local:
        raise ValueError("Local model inference blocked: --authorize-local-inference required")


def requests_for(responses, role, prompt):
    requests=[]
    for response in responses:
        assessments=response.assessment_map()
        visible=response.evidence_policy_view() if role=="judge" else response.policy_view()
        for unit in visible.units:
            scoped=f"{response.response_id}/{unit.unit_id}"
            metadata={"response_id":response.response_id,"unit_id":unit.unit_id}
            if role=="priority": requests.append(priority_request(scoped,unit.text,prompt,metadata=metadata))
            elif assessments[unit.unit_id].support is not SupportLevel.UNASSESSED:
                first=unit.evidence_texts[0] if unit.evidence_texts else None
                if not first: raise ValueError("Assessed unit lacks original first evidence; later substitution forbidden")
                requests.append(judge_request(scoped,unit.text,first,prompt,metadata=metadata))
    return requests


class TargetOnlyAfacta(VerificationPolicy):
    name="transfer:afacta-target-only-v1"
    def __init__(self,scorer,cache_path=None):
        self.scorer=scorer; self.cache={}; self.cost=None;self.cache_path=cache_path
        if cache_path is not None and cache_path.exists():
            payload=json.loads(cache_path.read_text(encoding="utf-8"))
            if payload.get("identity")!=scorer.identity: raise ValueError("Target-only AFaCTA cache identity mismatch")
            self.cache=payload["entries"]
    def rank(self,view,*,seed):
        pending=[u for u in view.units if content_hash(u.text) not in self.cache]
        before=self.scorer.model_calls
        if pending:
            values=self.scorer.score([{"text":u.text,"target_span":[0,len(u.text)]} for u in pending])
            if len(values)!=len(pending): raise ValueError("Incomplete AFaCTA scores")
            for u,v in zip(pending,values): self.cache[content_hash(u.text)]=v
        self.cost=TriageCost(len(view.units),len(view.units),model_calls=self.scorer.model_calls-before,
            cache_hits=len(view.units)-len(pending),cache_misses=len(pending),
            input_tokens=sum(self.cache[content_hash(u.text)]["input_tokens"] for u in view.units))
        rows=[RankedUnit(u.unit_id,self.cache[content_hash(u.text)]["score"]) for u in view.units]
        if any(not math.isfinite(r.score) or not 0<=r.score<=1 for r in rows): raise ValueError("Invalid AFaCTA score")
        return tuple(sorted(rows,key=lambda r:(-r.score,r.unit_id)))
    def triage_cost(self,view): return self.cost
    def save(self):
        if self.cache_path is not None: save(self.cache_path,dict(identity=self.scorer.identity,entries=self.cache,contains_gold=False))


class MockAfacta:
    identity={"mock":True}; model_calls=0
    def score(self,rows):
        self.model_calls+=1
        return [{"score":min(len(r["text"]),100)/100,"input_tokens":len(r["text"].split())} for r in rows]


def run(config_path, resource_root, data_root, output, role, backend="mock", *, paid=False, local=False,
        preflight=False, shard_index=0, shard_count=1, cache_dir=None):
    authorized(backend,paid=paid,local=local)
    if role in {"nli","afacta"} and backend=="openai": raise ValueError("Neural checkpoints require local backend, not API")
    if role in {"priority","judge"} and backend=="local": raise ValueError("Frozen remote snapshot requires openai backend; mock is an implementation fixture")
    if shard_count<1 or not 0<=shard_index<shard_count: raise ValueError("Invalid shard")
    if output.exists(): raise ValueError("Choose a new output directory")
    cfg=json.loads(config_path.read_text(encoding="utf-8"))
    if cfg.get("model")!=MODEL: raise ValueError("Frozen snapshot substitution forbidden")
    native=json.loads(verify(resource_root,cfg["native_config"]).read_text(encoding="utf-8"))
    if role in {"priority","judge"}: native.pop("controlled",None)
    populations=ingest(native,data_root)
    hydrated=[]
    for name,responses,tasks,cluster in populations:
        if role in {"judge","nli"} and name in native.get("trec",{}):
            cache=EvidenceTextCache.load_jsonl(verify(data_root,cfg["evidence"][name]))
            fresh=[]
            for response in responses:
                response,report=materialize_response_evidence(response,cache)
                if report.missing_evidence_ids: raise ValueError("Missing cited evidence IDs")
                fresh.append(response)
            responses=fresh
        hydrated.append((name,responses,tasks,cluster))
    output.mkdir(parents=True)
    receipt=dict(role=role,backend=backend,status="RUNNING",historical_identity_guaranteed=False,api_execution_requested=backend=="openai",
                 neural_execution_requested=backend=="local",results_role="IMPLEMENTATION_FIXTURE_ONLY" if backend=="mock" else "FRESH_REPLICATION",populations={})
    try:
        if role in {"priority","judge"}:
            prompt=verify(resource_root,cfg[role+"_prompt"]).read_text(encoding="utf-8").strip()
            protocol=json.loads(verify(resource_root,cfg[role+"_protocol"]).read_text(encoding="utf-8"))
            if protocol["model"]!=MODEL: raise ValueError("Protocol model mismatch")
            configured=protocol["execution_ceiling"]
            limit=configured.get("maximum_unique_requests",configured.get("maximum_unique_requests_per_run"))
            configuration=ProviderConfiguration(provider="mock",model="deterministic-mock-v1",endpoint="mock://local") if backend=="mock" else ProviderConfiguration(provider="openai",model=MODEL)
            for name,responses,_,_ in hydrated:
                requests=requests_for(responses,role,prompt)
                requests=[r for r in requests if int(r.input_hash,16)%shard_count==shard_index]
                receipt["populations"][name]=dict(records=len(requests),unique_requests=len({r.input_hash for r in requests}),
                    estimated_input_tokens=sum(estimate_request_tokens(r) for r in {r.input_hash:r for r in requests}.values()))
                if preflight: continue
                provider=MockStructuredProvider(lambda request: {"priority_score":50} if role=="priority" else {"support_label":"ABSTAIN"}) if backend=="mock" else OpenAIResponsesProvider(configuration)
                cache=StructuredResponseCache((cache_dir or output/"cache")/name,configuration)
                validator=validate_priority_output if role=="priority" else validate_judge_output
                records=execute_structured_requests(requests,provider,cache,ExecutionLimits(limit,configured["maximum_planned_tokens_per_run"],concurrency=4),validator)
                path=output/name/f"records-shard-{shard_index:03d}-of-{shard_count:03d}.jsonl";path.parent.mkdir(parents=True,exist_ok=True)
                path.write_text("".join(json.dumps(r.to_dict(),sort_keys=True)+"\n" for r in records),encoding="utf-8")
                if any(r.status!="ok" for r in records): raise ValueError("Incomplete inference; cache retained for explicit recovery")
        else:
            from .evidence_policies import TransformersNliScorer, DeterministicMockNliScorer, EvidenceScoreCache, NliEvidenceRiskPolicy
            from .external_policies import LocalAfactaScorer, AfactaPolicy
            if backend=="local":
                import torch
                torch.set_num_threads(4)
            if role=="nli":
                if backend=="local":
                    from huggingface_hub import snapshot_download
                    provenance=json.loads(verify(resource_root,cfg["nli_manifest"]).read_text(encoding="utf-8"))
                    from .evidence_policies import DEFAULT_MODEL_ID, DEFAULT_MODEL_REVISION
                    if provenance["repo"]!=DEFAULT_MODEL_ID or provenance["revision"]!=DEFAULT_MODEL_REVISION: raise ValueError("NLI model identity mismatch")
                    snapshot=Path(snapshot_download(repo_id=provenance["repo"],revision=provenance["revision"],local_files_only=True))
                    for name,info in provenance["files"].items(): verify(snapshot,dict(path=name,sha256=info["sha256"]))
                    scorer=TransformersNliScorer(batch_size=8,local_files_only=True,asset_sha256={n:r["sha256"] for n,r in provenance["files"].items()})
                else: scorer=DeterministicMockNliScorer()
                cache=EvidenceScoreCache(output/"nli_cache.json")
                policy=NliEvidenceRiskPolicy(scorer,cache=cache)
                policy.name="mock:nli" if backend=="mock" else "baseline:evidence-nli-risk-v1"
                target=policy
            else:
                acquisition=json.loads(verify(resource_root,cfg["afacta_manifest"]).read_text(encoding="utf-8")) if backend=="local" else {}
                scorer=MockAfacta() if backend=="mock" else LocalAfactaScorer(data_root/cfg["afacta_model_dir"],acquisition,batch_size=8,threads=4,seed=47)
                policy=AfactaPolicy(scorer,cache_path=output/"afacta_cache.json")
                target=TargetOnlyAfacta(scorer,cache_path=output/"afacta_target_cache.json")
                if backend=="mock": policy.name="mock:afacta-context"; target.name="mock:afacta-target"
            for name,responses,tasks,cluster in hydrated:
                if role=="nli" and name.startswith("feverous_"): continue
                dest=output/name;dest.mkdir(); traces=[]
                selected_policy=target if name.startswith(("feverous_","scifact_")) else policy
                if role=="nli" and backend=="local": selected_policy.name="transfer:nli-first-cited-abstract-v1" if name.startswith("scifact_") else "baseline:evidence-nli-risk-v1"
                if not preflight:
                    for response in responses:
                        for task in tasks:
                            for budget in native["budgets"]: traces.append(replay(response,selected_policy,task,budget,seed=native["seed"]).to_dict())
                    table,clusters=summarize(traces,cluster)
                    csv_write(dest/"results.csv",table);csv_write(dest/"cluster_results.csv",clusters)
                    (dest/"replay_traces.jsonl").write_text("".join(json.dumps(r,sort_keys=True)+"\n" for r in traces),encoding="utf-8")
                receipt["populations"][name]=dict(responses=len(responses),traces=len(traces))
            if not preflight:
                if role=="nli": cache.save()
                else: policy.save();target.save()
        receipt["status"]="PREFLIGHT_ONLY" if preflight else "PASS"
    except Exception as exc:
        receipt.update(status="FAIL",error=str(exc));raise
    finally: save(output/"inference_receipt.json",receipt)
    return receipt


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--config",type=Path,required=True)
    p.add_argument("--resource-root",type=Path,default=Path("."));p.add_argument("--data-dir",type=Path,required=True);p.add_argument("--output",type=Path,required=True)
    p.add_argument("--role",choices=("priority","judge","nli","afacta"),required=True);p.add_argument("--backend",choices=("mock","openai","local"),default="mock")
    p.add_argument("--authorize-paid-inference",action="store_true");p.add_argument("--authorize-local-inference",action="store_true");p.add_argument("--preflight",action="store_true")
    p.add_argument("--shard-index",type=int,default=0);p.add_argument("--shard-count",type=int,default=1)
    p.add_argument("--cache-dir",type=Path,help="Reusable private structured-response cache; content/provider/configuration bound")
    a=p.parse_args();print(json.dumps(run(a.config,a.resource_root,a.data_dir,a.output,a.role,a.backend,paid=a.authorize_paid_inference,local=a.authorize_local_inference,
        preflight=a.preflight,shard_index=a.shard_index,shard_count=a.shard_count,cache_dir=a.cache_dir),indent=2))


if __name__=="__main__": main()
