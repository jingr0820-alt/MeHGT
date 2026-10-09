#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
通用主模型训练脚本

推荐用法：
1) 主模型（HGT mech）
   python step4_train_main_model.py --use-mechanism --tag hgt_mech_main

2) 附加版本（HGT full = mechanism + focal）
   python step4_train_main_model.py --use-mechanism --use-focal --tag hgt_full_main

3) 纯 HGT（无 mechanism, 无 focal）
   python step4_train_main_model.py --tag hgt_plain_main
"""

import gc
import os
import json
import random
import argparse
from pathlib import Path
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score
from torch_geometric.nn import HGTConv, Linear


def find_project_root() -> Path:
    here = Path(__file__).resolve().parent
    for candidate in [here, here.parent]:
        if (candidate / "data").exists() and (candidate / "code").exists():
            return candidate
    raise FileNotFoundError("Cannot locate the repository root containing code/ and data/.")


PROJECT_ROOT = find_project_root()


class TCM_HGT(torch.nn.Module):
    def __init__(
        self,
        hidden_channels,
        out_channels,
        num_heads,
        data_metadata,
        num_nodes_dict,
        use_mechanism: bool = True,
    ):
        super().__init__()
        self.use_mechanism = use_mechanism

        self.node_lins = torch.nn.ModuleDict()
        self.learnable_embs = torch.nn.ModuleDict()

        for node_type in data_metadata[0]:
            self.node_lins[node_type] = Linear(-1, hidden_channels)

            if node_type in ["herb", "target"]:
                num_nodes = num_nodes_dict[node_type]
                self.learnable_embs[node_type] = torch.nn.Embedding(num_nodes, hidden_channels)
                torch.nn.init.xavier_uniform_(self.learnable_embs[node_type].weight)

        self.convs = torch.nn.ModuleList()
        for _ in range(2):
            self.convs.append(HGTConv(hidden_channels, hidden_channels, data_metadata, num_heads))

        mech_dim = 3 if use_mechanism else 0
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(hidden_channels * 4 + mech_dim, hidden_channels),
            torch.nn.LayerNorm(hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(hidden_channels, out_channels),
        )

    def forward(self, x_dict, edge_index_dict):
        x_new = {}
        for node_type, x in x_dict.items():
            h_fixed = self.node_lins[node_type](x)
            if node_type in self.learnable_embs:
                indices = torch.arange(x.size(0), device=x.device)
                x_new[node_type] = h_fixed + self.learnable_embs[node_type](indices)
            else:
                x_new[node_type] = h_fixed

        for conv in self.convs:
            x_new = conv(x_new, edge_index_dict)
            x_new = {node_type: F.gelu(x) for node_type, x in x_new.items()}

        return x_new

    def decode(self, z_herb, z_disease, edge_label_index, mech_features=None):
        row, col = edge_label_index
        z_src = z_herb[row]
        z_dst = z_disease[col]

        pair_prod = z_src * z_dst
        pair_abs_diff = torch.abs(z_src - z_dst)
        pair_features = torch.cat([z_src, z_dst, pair_prod, pair_abs_diff], dim=-1)

        if self.use_mechanism:
            if mech_features is None:
                raise ValueError("use_mechanism=True but mech_features is None")
            mech_features = mech_features.to(
                device=pair_features.device,
                dtype=pair_features.dtype,
            )
            decoder_input = torch.cat([pair_features, mech_features], dim=-1)
        else:
            decoder_input = pair_features

        return self.classifier(decoder_input).view(-1)


def aggregate_features(source_x, edge_index, target_num_nodes):
    src_idx = edge_index[0].to("cpu")
    dst_idx = edge_index[1].to("cpu")
    source_x = source_x.to("cpu")

    feat_dim = source_x.size(1)
    sums = torch.zeros(target_num_nodes, feat_dim, dtype=source_x.dtype)
    counts = torch.zeros(target_num_nodes, 1, dtype=source_x.dtype)

    sums.index_add_(0, dst_idx, source_x[src_idx])
    ones = torch.ones(src_idx.size(0), 1, dtype=source_x.dtype)
    counts.index_add_(0, dst_idx, ones)
    counts[counts == 0] = 1.0

    return sums / counts


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _select_device() -> torch.device:
    if torch.cuda.is_available():
        try:
            free_bytes, total_bytes = torch.cuda.mem_get_info()
            free_ratio = free_bytes / float(total_bytes)
            if free_ratio > 0.25:
                print(f"Using CUDA (free {free_bytes / 1024**3:.1f} GiB / {total_bytes / 1024**3:.1f} GiB).")
                return torch.device("cuda")
            print(
                f"CUDA detected but free memory is low "
                f"({free_bytes / 1024**3:.1f} GiB / {total_bytes / 1024**3:.1f} GiB). "
                f"Falling back to CPU for safe training."
            )
        except Exception:
            print("CUDA detected. mem_get_info unavailable, trying CUDA.")
            return torch.device("cuda")

    print("Using CPU for training.")
    return torch.device("cpu")


def safe_roc_auc(y_true, y_score):
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    if y_true.size == 0 or len(np.unique(y_true)) < 2:
        return None
    return float(roc_auc_score(y_true, y_score))


def safe_ap(y_true, y_score):
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    if y_true.size == 0 or len(np.unique(y_true)) < 2:
        return None
    return float(average_precision_score(y_true, y_score))


def safe_log1p(x):
    return float(np.log1p(max(float(x), 0.0)))


def build_neighbor_sets(edge_index_np, left_to_right=True):
    neighbors = {}
    left_arr, right_arr = edge_index_np[0], edge_index_np[1]
    if not left_to_right:
        left_arr, right_arr = right_arr, left_arr

    for left, right in zip(left_arr, right_arr):
        neighbors.setdefault(int(left), set()).add(int(right))
    return neighbors


def compute_target_weights(data):
    ht_key = None
    for key in [
        ("herb", "interacts", "target"),
        ("herb", "interacts_with", "target"),
        ("herb", "targets", "target"),
    ]:
        if key in data.edge_types:
            ht_key = key
            break
    if ht_key is None:
        raise KeyError(f"Cannot find herb->target edge type. Available: {data.edge_types}")

    td_key = None
    for key in [
        ("target", "rev_associates", "disease"),
        ("target", "associated_with", "disease"),
    ]:
        if key in data.edge_types:
            td_key = key
            break
    if td_key is None:
        raise KeyError(f"Cannot find target->disease edge type. Available: {data.edge_types}")

    ht_edge_np = data[ht_key].edge_index.detach().cpu().numpy()
    td_edge_np = data[td_key].edge_index.detach().cpu().numpy()

    herb_to_targets = build_neighbor_sets(ht_edge_np, left_to_right=True)

    disease_to_targets = {}
    for t, d in zip(td_edge_np[0], td_edge_np[1]):
        disease_to_targets.setdefault(int(d), set()).add(int(t))

    target_x = data["target"].x.detach().cpu().numpy()
    target_idf = {t_id: float(target_x[t_id, 0]) for t_id in range(target_x.shape[0])}

    target_dis_counts = {}
    for t, d in zip(td_edge_np[0], td_edge_np[1]):
        t = int(t)
        target_dis_counts[t] = target_dis_counts.get(t, 0) + 1

    target_weights = {}
    for t_id, idf_val in target_idf.items():
        dis_cnt = max(target_dis_counts.get(t_id, 1), 1)
        disease_penalty = safe_log1p(dis_cnt)
        if disease_penalty <= 0:
            disease_penalty = 1.0
        target_weights[t_id] = float(idf_val / disease_penalty)

    return herb_to_targets, disease_to_targets, target_weights


def build_mech_features(edge_index, herb_to_targets, disease_to_targets, target_weights, device):
    edge_np = edge_index.detach().cpu().numpy()
    rows, cols = edge_np[0], edge_np[1]
    feats = np.zeros((len(rows), 3), dtype=np.float32)

    for i, (h_id, d_id) in enumerate(zip(rows, cols)):
        herb_targets = herb_to_targets.get(int(h_id), set())
        disease_targets = disease_to_targets.get(int(d_id), set())
        shared = herb_targets & disease_targets

        if shared:
            shared_weights = [target_weights.get(t, 0.0) for t in shared]
            raw_shared_weight = float(sum(shared_weights))
            mech_score = float(np.log1p(raw_shared_weight))
            shared_count = float(len(shared))
        else:
            raw_shared_weight = 0.0
            mech_score = 0.0
            shared_count = 0.0

        feats[i, 0] = shared_count
        feats[i, 1] = raw_shared_weight
        feats[i, 2] = mech_score

    feats[:, 0] = np.log1p(feats[:, 0])
    feats[:, 1] = np.log1p(feats[:, 1])
    feats[:, 2] = np.log1p(feats[:, 2])

    return torch.tensor(feats, dtype=torch.float32, device=device)


def focal_bce_with_logits(
    logits,
    targets,
    pos_weight=None,
    gamma: float = 2.0,
    alpha: float = 0.25,
):
    targets = targets.float()
    bce = torch.nn.functional.binary_cross_entropy_with_logits(
        logits,
        targets,
        reduction="none",
        pos_weight=pos_weight,
    )
    probs = torch.sigmoid(logits)
    pt = torch.where(targets > 0.5, probs, 1.0 - probs)
    focal_factor = (1.0 - pt).pow(gamma)

    alpha_t = torch.where(
        targets > 0.5,
        torch.full_like(targets, alpha),
        torch.full_like(targets, 1.0 - alpha),
    )
    loss = alpha_t * focal_factor * bce
    return loss.mean()


def evaluate_split(model, data, mp_edge_index_dict, edge_index, edge_label, mech_features, use_amp):
    model.eval()
    with torch.no_grad():
        eval_ctx = torch.amp.autocast("cuda") if use_amp else nullcontext()
        with eval_ctx:
            z = model(data.x_dict, mp_edge_index_dict)
            logits = model.decode(z["herb"], z["disease"], edge_index, mech_features)
            preds = torch.sigmoid(logits).detach().cpu().numpy()

    auc = safe_roc_auc(edge_label, preds)
    ap = safe_ap(edge_label, preds)
    logit_max = float(logits.max().item()) if logits.numel() > 0 else None
    logit_min = float(logits.min().item()) if logits.numel() > 0 else None
    return auc, ap, preds, logit_min, logit_max


def format_metric(x):
    return "nan" if x is None else f"{x:.4f}"


def main():
    parser = argparse.ArgumentParser(description="Train main HGT model with optional mechanism/focal switches.")
    parser.add_argument("--use-mechanism", action="store_true", help="Use mechanism-aware decoder features.")
    parser.add_argument("--use-focal", action="store_true", help="Use focal BCE loss instead of BCEWithLogits.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--max-epochs", type=int, default=1000, help="Maximum number of training epochs.")
    parser.add_argument("--eval-every", type=int, default=20, help="Evaluate on validation every N epochs.")
    parser.add_argument("--patience-checks", type=int, default=10, help="Early stopping patience measured in validation checks.")
    parser.add_argument(
        "--tag",
        type=str,
        default=None,
        help="Custom output tag. Example: hgt_mech_main / hgt_full_main",
    )
    parser.add_argument("--graph-path", type=str, default="out/graph/hetero_data.pt")
    parser.add_argument("--dataset-meta-path", type=str, default="out/dataset/dataset_meta.json")
    parser.add_argument("--model-dir", type=str, default="out/models")
    parser.add_argument("--eval-dir", type=str, default="out/eval")
    args = parser.parse_args()

    print("Step 4 - Train Main HGT Model")
    set_seed(args.seed)

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    device = _select_device()

    graph_path = Path(args.graph_path)
    if not graph_path.is_absolute():
        graph_path = PROJECT_ROOT / graph_path
    dataset_meta_path = Path(args.dataset_meta_path)
    if not dataset_meta_path.is_absolute():
        dataset_meta_path = PROJECT_ROOT / dataset_meta_path
    model_dir = Path(args.model_dir)
    eval_dir = Path(args.eval_dir)
    if not model_dir.is_absolute():
        model_dir = PROJECT_ROOT / model_dir
    if not eval_dir.is_absolute():
        eval_dir = PROJECT_ROOT / eval_dir

    if not graph_path.exists():
        raise FileNotFoundError(f"Graph not found: {graph_path}")

    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(eval_dir, exist_ok=True)

    dataset_meta = {}
    if dataset_meta_path.exists():
        with open(dataset_meta_path, "r", encoding="utf-8") as f:
            dataset_meta = json.load(f)
        print(
            "[Meta] dataset_mode="
            f"{dataset_meta.get('dataset_mode')} | label_source={dataset_meta.get('label_source')}"
        )
        if dataset_meta.get("dataset_mode") != "external_cpm_strict":
            print("[Warn] You are not using the strict external-label evaluation dataset.")

    if args.tag is not None:
        base_tag = args.tag
    else:
        if args.use_mechanism and args.use_focal:
            base_tag = "hgt_full_main"
        elif args.use_mechanism and not args.use_focal:
            base_tag = "hgt_mech_main"
        elif (not args.use_mechanism) and args.use_focal:
            base_tag = "hgt_plain_focal_main"
        else:
            base_tag = "hgt_plain_main"

    best_model_path = model_dir / f"{base_tag}_best.pt"
    compat_model_path = model_dir / f"{base_tag}.pth"
    metrics_path = eval_dir / f"{base_tag}_metrics.json"

    data = torch.load(graph_path, weights_only=False)
    print(f"[OK] Graph loaded: {graph_path}")

    if data["target"].x.dim() == 1:
        data["target"].x = data["target"].x.view(-1, 1)
    elif data["target"].x.size(1) > 1:
        data["target"].x = data["target"].x[:, :1]

    td_key_candidates = [
        ("target", "rev_associates", "disease"),
        ("target", "associated_with", "disease"),
    ]
    used_td_key = None
    for key in td_key_candidates:
        if key in data.edge_types:
            td_edge = data[key].edge_index
            data["disease"].x = aggregate_features(
                data["target"].x, td_edge, data["disease"].num_nodes
            )
            used_td_key = key
            break

    if used_td_key is not None:
        print(f"[OK] disease.x aggregated from edge type: {used_td_key}")
    else:
        print("[WARN] No target->disease edge found for aggregation. Keeping original disease.x.")

    data = data.to(device)

    mp_edge_index_dict = data.edge_index_dict.copy()
    leakage_keys = [
        ("herb", "treats", "disease"),
        ("disease", "rev_treats", "herb"),
    ]

    removed_count = 0
    for key in leakage_keys:
        if key in mp_edge_index_dict:
            del mp_edge_index_dict[key]
            removed_count += 1

    print(f"[Security] Removed {removed_count} leakage edge types from message passing.")

    if ("herb", "treats", "disease") not in data.edge_types:
        raise KeyError("Supervision edge type ('herb', 'treats', 'disease') not found in graph.")

    edge_obj = data["herb", "treats", "disease"]

    required_attrs = ["edge_index", "edge_label", "train_mask", "val_mask"]
    missing = [a for a in required_attrs if not hasattr(edge_obj, a)]
    if missing:
        raise AttributeError(
            f"Supervision edge is missing required attributes: {missing}. "
            f"Please verify step3 output."
        )

    train_idx = edge_obj.edge_index[:, edge_obj.train_mask]
    train_y = edge_obj.edge_label[edge_obj.train_mask].float()

    val_idx = edge_obj.edge_index[:, edge_obj.val_mask]
    val_y = edge_obj.edge_label[edge_obj.val_mask].float().cpu().numpy()

    has_test = hasattr(edge_obj, "test_mask")
    if has_test:
        test_idx = edge_obj.edge_index[:, edge_obj.test_mask]
        test_y = edge_obj.edge_label[edge_obj.test_mask].float().cpu().numpy()
    else:
        test_idx = None
        test_y = None
        print("[WARN] test_mask not found. Final test evaluation will be skipped.")

    print(f"Train edges: {train_idx.size(1)}")
    print(f"Val edges  : {val_idx.size(1)}")
    if has_test:
        print(f"Test edges : {test_idx.size(1)}")

    train_y_cpu = train_y.detach().cpu()
    pos_count = int((train_y_cpu == 1).sum().item())
    neg_count = int((train_y_cpu == 0).sum().item())
    print(f"Train positives: {pos_count}")
    print(f"Train negatives: {neg_count}")

    if pos_count == 0:
        raise ValueError("训练集中没有正样本，无法训练。")
    if neg_count == 0:
        raise ValueError("训练集中没有负样本，无法训练。")

    pos_weight_value = neg_count / max(pos_count, 1)
    pos_weight = torch.tensor([pos_weight_value], dtype=torch.float, device=device)
    print(f"Using pos_weight = {pos_weight_value:.4f}")

    num_nodes_dict = {
        "herb": data["herb"].num_nodes,
        "target": data["target"].num_nodes,
        "disease": data["disease"].num_nodes,
    }

    if args.use_mechanism:
        herb_to_targets, disease_to_targets, target_weights = compute_target_weights(data)
        print("[OK] Built mechanism priors from herb-target and target-disease graph.")
        train_mech = build_mech_features(train_idx, herb_to_targets, disease_to_targets, target_weights, device)
        val_mech = build_mech_features(val_idx, herb_to_targets, disease_to_targets, target_weights, device)
        test_mech = (
            build_mech_features(test_idx, herb_to_targets, disease_to_targets, target_weights, device)
            if has_test else None
        )
    else:
        train_mech = None
        val_mech = None
        test_mech = None
        print("[Info] Mechanism-aware decoder is disabled.")

    model = TCM_HGT(
        hidden_channels=48,
        out_channels=1,
        num_heads=4,
        data_metadata=data.metadata(),
        num_nodes_dict=num_nodes_dict,
        use_mechanism=args.use_mechanism,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-4)

    if args.use_focal:
        loss_name = "focal_bce_with_logits"
        focal_gamma = 2.0
        focal_alpha = 0.25
        print(f"Using {loss_name} (gamma={focal_gamma}, alpha={focal_alpha})")
    else:
        loss_name = "bce_with_logits"
        focal_gamma = None
        focal_alpha = None
        print(f"Using {loss_name}")

    use_amp = (device.type == "cuda")
    if use_amp:
        if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
            scaler = torch.amp.GradScaler("cuda", enabled=True)
        else:
            scaler = torch.cuda.amp.GradScaler(enabled=True)
    else:
        scaler = None

    print("Start training (max 1000 epochs)...")
    best_val_auc = float("-inf")
    best_val_ap = float("-inf")
    best_epoch = 0
    eval_every = max(int(args.eval_every), 1)
    patience_checks = max(int(args.patience_checks), 1)
    patience_counter = 0
    final_train_loss = None
    grad_clip_norm = 2.0

    print(f"Training config: max_epochs={args.max_epochs}, eval_every={eval_every}, patience_checks={patience_checks}")

    for epoch in range(1, int(args.max_epochs) + 1):
        model.train()
        optimizer.zero_grad()

        autocast_ctx = torch.amp.autocast("cuda") if use_amp else nullcontext()

        with autocast_ctx:
            z = model(data.x_dict, mp_edge_index_dict)
            pred = model.decode(z["herb"], z["disease"], train_idx, train_mech)

            if args.use_focal:
                loss = focal_bce_with_logits(
                    pred,
                    train_y,
                    pos_weight=pos_weight,
                    gamma=2.0,
                    alpha=0.25,
                )
            else:
                loss = F.binary_cross_entropy_with_logits(
                    pred,
                    train_y,
                    pos_weight=pos_weight,
                )

        if use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            optimizer.step()

        final_train_loss = float(loss.item())

        if epoch % eval_every == 0:
            val_auc, val_ap, _, min_logit, max_logit = evaluate_split(
                model, data, mp_edge_index_dict, val_idx, val_y, val_mech, use_amp
            )

            logit_text = "n/a" if min_logit is None or max_logit is None else f"[{min_logit:.2f}, {max_logit:.2f}]"
            print(
                f"Epoch {epoch:04d} | "
                f"Loss: {loss.item():.4f} | "
                f"GradNorm: {float(grad_norm):.4f} | "
                f"Val AUC: {format_metric(val_auc)} | "
                f"Val AP: {format_metric(val_ap)} | "
                f"Logit Range: {logit_text}"
            )

            improved = False
            if val_auc is not None and val_auc > best_val_auc:
                improved = True
            elif val_auc is not None and best_val_auc != float("-inf") and abs(val_auc - best_val_auc) < 1e-6:
                if val_ap is not None and val_ap > best_val_ap:
                    improved = True

            if improved:
                best_val_auc = float(val_auc) if val_auc is not None else best_val_auc
                best_val_ap = float(val_ap) if val_ap is not None else best_val_ap
                best_epoch = epoch
                torch.save(model.state_dict(), best_model_path)
                torch.save(model.state_dict(), compat_model_path)
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience_checks:
                print(f"[Early Stopping] No improvement for {patience_checks} validation checks.")
                break

    if best_model_path.exists():
        state = torch.load(best_model_path, map_location=device, weights_only=False)
        model.load_state_dict(state)
        print(f"[OK] Reloaded best checkpoint from epoch {best_epoch}.")
    else:
        print("[WARN] Best checkpoint was not saved; using current model weights.")

    final_val_auc, final_val_ap, _, _, _ = evaluate_split(
        model, data, mp_edge_index_dict, val_idx, val_y, val_mech, use_amp
    )

    if has_test:
        test_auc, test_ap, _, _, _ = evaluate_split(
            model, data, mp_edge_index_dict, test_idx, test_y, test_mech, use_amp
        )
    else:
        test_auc = None
        test_ap = None

    metrics = {
        "tag": base_tag,
        "model_seed": int(args.seed),
        "split_seed": dataset_meta.get("split_seed", dataset_meta.get("seed")),
        "use_mechanism": bool(args.use_mechanism),
        "use_focal": bool(args.use_focal),
        "best_val_auc": None if best_val_auc == float("-inf") else float(best_val_auc),
        "best_val_ap": None if best_val_ap == float("-inf") else float(best_val_ap),
        "best_epoch": int(best_epoch),
        "final_val_auc": None if final_val_auc is None else float(final_val_auc),
        "final_val_ap": None if final_val_ap is None else float(final_val_ap),
        "test_auc": None if test_auc is None else float(test_auc),
        "test_ap": None if test_ap is None else float(test_ap),
        "final_train_loss": None if final_train_loss is None else float(final_train_loss),
        "train_edges": int(train_idx.size(1)),
        "val_edges": int(val_idx.size(1)),
        "test_edges": None if not has_test else int(test_idx.size(1)),
        "train_pos_edges": int(pos_count),
        "train_neg_edges": int(neg_count),
        "pos_weight": float(pos_weight_value),
        "loss_name": loss_name,
        "focal_gamma": None if focal_gamma is None else float(focal_gamma),
        "focal_alpha": None if focal_alpha is None else float(focal_alpha),
        "eval_every": int(eval_every),
        "patience_checks": int(patience_checks),
        "grad_clip_norm": float(grad_clip_norm),
        "model_path_best": str(best_model_path),
        "model_path_compat": str(compat_model_path),
        "graph_path": str(graph_path),
    }

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print("\nTraining finished.")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
