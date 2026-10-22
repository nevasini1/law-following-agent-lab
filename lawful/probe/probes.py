"""White-box legality probes: logistic regression on residual-stream activations at the final prompt token."""
from __future__ import annotations

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


@torch.no_grad()
def last_token_states(model, tok, chats: list[list[dict]], batch: int = 4) -> tuple[np.ndarray, np.ndarray]:
    """Hidden states (n, layers+1, d) at the last prompt token, plus P(Yes) from the next-token logits."""
    tok.padding_side = "left"
    yes, no = tok.encode("Yes")[0], tok.encode("No")[0]
    feats, p_yes = [], []
    for i in range(0, len(chats), batch):
        prompts = [tok.apply_chat_template(c, tokenize=False, add_generation_prompt=True) for c in chats[i:i + batch]]
        enc = tok(prompts, return_tensors="pt", padding=True)
        out = model(**enc, output_hidden_states=True)
        feats.append(torch.stack(out.hidden_states, 1)[:, :, -1].float().numpy())
        lg = out.logits[:, -1, [yes, no]].float()
        p_yes += torch.softmax(lg, -1)[:, 0].tolist()
    return np.concatenate(feats), np.array(p_yes)


def layer_sweep(X: np.ndarray, y: np.ndarray, groups: np.ndarray, C: float = 0.05, folds: int = 5) -> list[float]:
    """Grouped-CV AUROC of a linear probe at every layer (groups keep a scenario's items in one fold)."""
    aucs = []
    for layer in range(X.shape[1]):
        pred = np.zeros(len(y))
        for tr, te in GroupKFold(folds).split(X, y, groups):
            clf = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=3000))
            pred[te] = clf.fit(X[tr, layer], y[tr]).predict_proba(X[te, layer])[:, 1]
        aucs.append(roc_auc_score(y, pred))
    return aucs


def transfer(X, y, fam, train_fams, layer, C: float = 0.05) -> float:
    tr = np.isin(fam, train_fams)
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=3000)).fit(X[tr, layer], y[tr])
    te = ~tr
    return roc_auc_score(y[te], clf.predict_proba(X[te, layer])[:, 1]) if len(set(y[te])) > 1 else float("nan")
