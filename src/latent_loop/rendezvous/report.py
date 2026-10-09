"""Regenerate population metrics and scientific figures from recorded decisions."""

import json
from pathlib import Path

import numpy as np

from .data import ROOT, read_rows
from .study import paired_ci, save, softmax

REPORT = Path("reports/experiment_05e")
FIGURES = REPORT / "figures"


def correctness(rows, method):
    return np.array([[r["seeds"][j]["choices"][method] in r["optimal"] for r in rows] for j in range(len(rows[0]["seeds"]))], dtype=float)


def ece(confidence, labels):
    return sum(np.mean((confidence >= a) & (confidence < b)) * abs(np.mean(confidence[(confidence >= a) & (confidence < b)])-np.mean(labels[(confidence >= a) & (confidence < b)])) for a,b in zip(np.linspace(0,1.000001,11)[:-1],np.linspace(0,1.000001,11)[1:]) if np.any((confidence >= a) & (confidence < b)))


def auc(score, label):
    from scipy.stats import rankdata
    score, label = np.asarray(score), np.asarray(label, dtype=bool)
    n = label.sum(); m = len(label)-n
    if not n or not m:
        return None
    return float((rankdata(score)[label].sum()-n*(n+1)/2)/(n*m))


def summarize(config):
    summary = {"splits": {}, "primary": {}, "inference_scope": "Conditional selection among four verified single-token labels; not free-text generation."}
    frozen = json.loads((ROOT / "protocol_freeze.json").read_text())
    tokenids = json.loads((ROOT / "backbone_train_validation.json").read_text())["label_tokens"]
    for path in sorted((ROOT / "decisions").glob("*.jsonl.gz")):
        split = path.name.split(".")[0]; rows = read_rows(path)
        methods = list(rows[0]["seeds"][0]["choices"])
        base = correctness(rows, "base")
        metrics = {}
        for method in methods:
            correct = correctness(rows, method)
            p = np.array([[softmax(r["seeds"][j]["scores"][method]) for r in rows] for j in range(3)])
            selected = np.array([[r["seeds"][j]["choices"][method] for r in rows] for j in range(3)])
            confidence = np.take_along_axis(p, selected[...,None], axis=-1)[...,0]
            y = np.array([[i in r["optimal"] for i in range(4)] for r in rows])
            metrics[method] = {"accuracy": float(correct.mean()), "seed_accuracy": correct.mean(axis=1).tolist(), "wrong_to_right": float(((base == 0)&(correct == 1)).mean()), "right_to_wrong": float(((base == 1)&(correct == 0)).mean()), "changed_fraction": float(np.mean([[r["seeds"][j]["choices"][method] != r["base_choice"] for r in rows] for j in range(3)])), "conditional_set_nll": float(-np.log(np.maximum((p*y[None]).sum(-1),1e-15)).mean()), "selection_ece": float(ece(confidence.ravel(),correct.ravel())), "selected_confidence_auroc": auc(confidence.ravel(),correct.ravel()), "family_accuracy": {family: float(correct[:,[r["family"] == family for r in rows]].mean()) for family in ("routing","state","graph")}}
        base_label_counts = {l: sum(r["labels"][r["base_choice"]] == l for r in rows)/len(rows) for l in "ABCD"}
        selected_q = frozen["selected"]["objective"]
        all_q = np.array([[r["seeds"][j]["scores"][selected_q] for r in rows] for j in range(3)])
        targets = np.array([[i in r["optimal"] for i in range(4)] for r in rows])
        q_at_base = np.array([[all_q[j,i,r["base_choice"]] for i,r in enumerate(rows)] for j in range(3)])
        summary["splits"][split] = {"count": len(rows), "metrics": metrics, "base_selected_label_frequency": base_label_counts, "unrestricted_permitted_label_fraction": sum(r["unrestricted_token"] in tokenids.values() for r in rows)/len(rows), "unrestricted_matches_constrained_label": sum(r["unrestricted_token"] == tokenids[r["labels"][r["base_choice"]]] for r in rows)/len(rows), "candidate_quality_auroc": auc(all_q.ravel(),np.tile(targets,(3,1,1)).ravel()), "q_current_base_correctness_auroc": auc(q_at_base.ravel(),base.ravel()), "candidate_sigmoid_brier": float(np.mean((1/(1+np.exp(-np.clip(all_q,-40,40)))-targets[None])**2)) if selected_q == "correctness" else None}
        if split == "structure":
            fusion, direct = correctness(rows,"fusion"), correctness(rows,"c2")
            summary["primary"] = {"fusion_minus_base": paired_ci(fusion-base), "fusion_minus_c2": paired_ci(fusion-direct), "corrections_per_seed": ((base == 0)&(fusion == 1)).sum(axis=1).tolist(), "degradations_per_seed": ((base == 1)&(fusion == 0)).sum(axis=1).tolist()}
    perm_path = ROOT / "decisions/permutations.jsonl.gz"
    if perm_path.exists():
        perm = read_rows(perm_path)
        originals = {r["id"]:r for split in ("id","structure") for r in read_rows(ROOT / "decisions" / f"{split}.jsonl.gz")}
        source = {r["id"]:r for r in read_rows(ROOT / "datasets/permutations.jsonl.gz")}
        summary["permutation"] = {}
        for kind in ("candidate_relabel", "symbol_and_candidate_relabel"):
            subset = [r for r in perm if source[r["id"]]["permutation_kind"] == kind]
            summary["permutation"][kind] = {m: {"semantic_agreement": float(np.mean([[r["seeds"][j]["choices"][m] == originals[r["parent_id"]]["seeds"][j]["choices"][m] for r in subset] for j in range(3)])), "accuracy": float(correctness(subset,m).mean()), "parent_accuracy": float(np.mean([[originals[r["parent_id"]]["seeds"][j]["choices"][m] in r["optimal"] for r in subset] for j in range(3)]))} for m in ("base","fusion","q_only","c2")}
    save(ROOT / "summary.json", summary)
    return summary


def report(config):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    summary = summarize(config)
    FIGURES.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"figure.dpi":130,"font.size":10,"axes.spines.top":False,"axes.spines.right":False})
    splits = [s for s in ("id","wording","format","structure","harder") if s in summary["splits"]]
    def plot(name, callback, figsize=(8,4.4)):
        fig, ax = plt.subplots(figsize=figsize); callback(ax);fig.tight_layout();fig.savefig(FIGURES / f"{name}.svg");plt.close(fig)
    def grouped(ax, methods, labels=None):
        xx=np.arange(len(splits)); w=.8/len(methods)
        for j,m in enumerate(methods):
            ax.bar(xx+(j-(len(methods)-1)/2)*w,[100*summary['splits'][s]['metrics'][m]['accuracy'] for s in splits],width=w,label=(labels or methods)[j])
        ax.set_xticks(xx,splits);ax.set_ylabel('Optimal-action accuracy (%)');ax.legend();ax.set_ylim(0,100)
    plot("01_frozen_vs_scored",lambda ax:grouped(ax,['base','fusion']))
    def transitions(ax):
        xx=np.arange(len(splits))
        ax.bar(xx-.18,[100*summary['splits'][s]['metrics']['fusion']['wrong_to_right'] for s in splits],.36,label='Wrong → right')
        ax.bar(xx+.18,[100*summary['splits'][s]['metrics']['fusion']['right_to_wrong'] for s in splits],.36,label='Right → wrong');ax.set_xticks(xx,splits);ax.set_ylabel('Share of decisions (%)');ax.legend()
    plot("02_corrections_degradations",transitions)
    def seeds(ax):
        for s in splits:
            ax.plot(config['training_seeds'],[100*(a-b) for a,b in zip(summary['splits'][s]['metrics']['fusion']['seed_accuracy'],summary['splits'][s]['metrics']['base']['seed_accuracy'])],marker='o',label=s)
        ax.axhline(0,color='gray');ax.set_ylabel('Fusion minus frozen (percentage points)');ax.set_xlabel('Training seed');ax.legend(ncol=2)
    plot("03_seed_effects",seeds)
    plot("04_classifier_controls",lambda ax:grouped(ax,['q_only','fusion','c1','c2','residual']),figsize=(10,4.4))
    def families(ax):
        xx=np.arange(3)
        for j,m in enumerate(['base','fusion','c2']):
            ax.bar(xx+(j-1)*.25,[100*summary['splits']['structure']['metrics'][m]['family_accuracy'][f] for f in ['routing','state','graph']],.25,label=m)
        ax.set_xticks(xx,['routing','state','graph']);ax.set_ylabel('Structural-shift accuracy (%)');ax.legend()
    plot("05_task_families",families)
    plot("06_shift_controls",lambda ax:grouped(ax,['base','fusion','logits_only','description_only','abstain_prior']),figsize=(10,4.4))
    def permutation(ax):
        for j,m in enumerate(['base','fusion','q_only','c2']):
            ax.bar(np.arange(2)+(j-1.5)*.18,[100*summary['permutation'][k][m]['semantic_agreement'] for k in summary['permutation']],.18,label=m)
        ax.set_xticks([0,1],['Candidate relabel','Symbols + candidates']);ax.set_ylabel('Semantic agreement (%)');ax.set_ylim(0,100);ax.legend()
    if 'permutation' in summary:
        plot("07_candidate_order",permutation)
    def calibration(ax):
        rows=read_rows(ROOT/'decisions/structure.jsonl.gz')
        for m in ['base','temperature','fusion','c2']:
            conf=[];truth=[]
            for r in rows:
                for seed in r['seeds']:
                    p=softmax(seed['scores'][m]);a=seed['choices'][m];conf.append(p[a]);truth.append(a in r['optimal'])
            conf,truth=np.array(conf),np.array(truth);xx=[];yy=[]
            for a,b in zip(np.linspace(0,1.000001,11)[:-1],np.linspace(0,1.000001,11)[1:]):
                ix=(conf>=a)&(conf<b)
                if ix.any():xx.append(conf[ix].mean());yy.append(truth[ix].mean())
            ax.plot(xx,yy,marker='o',label=m)
        ax.plot([0,1],[0,1],color='gray',ls='--');ax.set_xlabel('Selected softmax confidence');ax.set_ylabel('Observed correctness');ax.legend()
    plot("08_calibration",calibration)
    if (ROOT/'timing.json').exists():
        timing=json.loads((ROOT/'timing.json').read_text())
        def latency(ax):
            for m in ['base','fusion','c2']:
                ax.scatter(timing['batch1'][m]['median_ms'],100*summary['splits']['structure']['metrics'][m]['accuracy'],label=m,s=60)
            ax.set_xlabel('Full-policy batch-1 median latency (ms)');ax.set_ylabel('Structural-shift accuracy (%)');ax.legend()
        plot("09_accuracy_latency",latency)
    rows=read_rows(ROOT/'decisions/structure.jsonl.gz'); raw={r['id']:r for r in read_rows(ROOT/'datasets/structure.jsonl.gz')}
    examples=[]
    for category in ['corrected','damaged']:
        found=next((r for r in rows if (r['base_choice'] in r['optimal']) == (category=='damaged') and (r['seeds'][0]['choices']['fusion'] in r['optimal']) == (category=='corrected')),None)
        if found:examples.append({'category':category,'task':raw[found['id']],'decision':found})
    save(ROOT/'changed_examples.json',examples)
    def changes(ax):
        ax.axis('off');text=[]
        for e in examples:
            r=e['decision'];task=e['task'];labels=task['labels'];a=r['base_choice'];b=r['seeds'][0]['choices']['fusion']
            text.append(f"{e['category'].upper()} — {task['family']} — seed 511\nFrozen: {labels[a]} ({task['candidate_descriptions'][a]})\nFusion: {labels[b]} ({task['candidate_descriptions'][b]})\nAccepted labels: {', '.join(labels[j] for j in r['optimal'])}\nID: {r['id'][:16]}")
        ax.text(.01,.98,'\n\n'.join(text) or 'No changed-decision examples exist.',va='top',wrap=True)
    plot("10_changed_decisions",changes)
    path=REPORT/'the_rendezvous.md'
    if path.exists():
        render_html(path)


def render_html(path):
    import base64
    import markdown
    import re
    path=Path(path)
    content=markdown.markdown(path.read_text(),extensions=['tables','fenced_code'])
    def inline(match):
        p=path.parent / match.group(1)
        if p.is_file() and p.suffix=='.svg':
            return 'src="data:image/svg+xml;base64,'+base64.b64encode(p.read_bytes()).decode()+'"'
        return match.group(0)
    content=re.sub(r'src="([^"]+)"',inline,content)
    path.with_suffix('.html').write_text('<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>'+path.stem+'</title><style>body{max-width:1050px;margin:40px auto;padding:0 24px;font:17px/1.55 system-ui;color:#17202b}img{max-width:100%}table{border-collapse:collapse;width:100%;font-size:14px}td,th{border:1px solid #bcc7d2;padding:6px;text-align:left}pre{overflow:auto;background:#eef2f7;padding:16px}a{color:#1554a2}</style>'+content+'</html>\n')
