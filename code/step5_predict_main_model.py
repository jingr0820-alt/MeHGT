#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import math
import argparse
import joblib
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.nn import HGTConv, Linear


def find_project_root() -> Path:
    here = Path(__file__).resolve().parent
    for candidate in [here, here.parent]:
        if (candidate / "data").exists() and (candidate / "code").exists():
            return candidate
    raise FileNotFoundError("Cannot locate the repository root containing code/ and data/.")


PROJECT_ROOT = find_project_root()


def find_td_edge_key(data):
    candidates = [
        ("target", "rev_associates", "disease"),
        ("target", "associated_with", "disease"),
    ]
    for key in candidates:
        if key in data.edge_types:
            return key
    raise KeyError(
        "Cannot find target->disease edge type. "
        f"Available edge types: {data.edge_types}"
    )


def find_ht_edge_key(data):
    candidates = [
        ("herb", "interacts", "target"),
        ("herb", "interacts_with", "target"),
        ("herb", "targets", "target"),
    ]
    for key in candidates:
        if key in data.edge_types:
            return key
    raise KeyError(
        "Cannot find herb->target edge type. "
        f"Available edge types: {data.edge_types}"
    )


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

        pair_features = torch.cat(
            [z_src, z_dst, pair_prod, pair_abs_diff],
            dim=-1,
        )

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


def load_all_positive_pairs_from_graph(data):
    sup_key = ("herb", "treats", "disease")
    if sup_key not in data.edge_types:
        raise KeyError(f"Supervision edge type not found: {sup_key}")

    edge_obj = data[sup_key]
    required_attrs = ["edge_index", "edge_label"]
    missing = [a for a in required_attrs if not hasattr(edge_obj, a)]
    if missing:
        raise AttributeError(
            f"Supervision edge is missing required attributes: {missing}"
        )

    edge_index = edge_obj.edge_index.cpu().numpy()
    edge_label = edge_obj.edge_label.cpu().numpy()

    all_pos = set()
    for i in range(edge_index.shape[1]):
        if int(edge_label[i]) == 1:
            h = int(edge_index[0, i])
            d = int(edge_index[1, i])
            all_pos.add((h, d))

    return all_pos


def parse_mappings(mappings_obj):
    if isinstance(mappings_obj, (tuple, list)) and len(mappings_obj) >= 2:
        herb_map = mappings_obj[0]
        disease_map = mappings_obj[1]
        target_map = mappings_obj[2] if len(mappings_obj) > 2 else {}
        return herb_map, disease_map, target_map

    if isinstance(mappings_obj, dict):
        herb_map = mappings_obj.get("herb_map") or mappings_obj.get("herb")
        disease_map = mappings_obj.get("disease_map") or mappings_obj.get("disease")
        target_map = mappings_obj.get("target_map") or mappings_obj.get("target") or {}
        if herb_map is None or disease_map is None:
            raise ValueError("Cannot parse mappings.pkl")
        return herb_map, disease_map, target_map

    raise TypeError(f"Unsupported mappings format: {type(mappings_obj)}")


def build_neighbor_sets(edge_index_np, left_to_right=True):
    mapping = {}
    if left_to_right:
        for a, b in zip(edge_index_np[0], edge_index_np[1]):
            mapping.setdefault(int(a), set()).add(int(b))
    else:
        for a, b in zip(edge_index_np[0], edge_index_np[1]):
            mapping.setdefault(int(b), set()).add(int(a))
    return mapping


def safe_log1p(x):
    return math.log1p(max(float(x), 0.0))


def normalize_series(s: pd.Series):
    if len(s) == 0:
        return s
    s = s.astype(float)
    s_min = float(s.min())
    s_max = float(s.max())

    if abs(s_max - s_min) < 1e-12:
        return pd.Series(np.zeros(len(s)), index=s.index, dtype=float)

    return (s - s_min) / (s_max - s_min)


def series_quantiles(s: pd.Series):
    if len(s) == 0:
        return {
            "q00": 0.0,
            "q25": 0.0,
            "q50": 0.0,
            "q75": 0.0,
            "q90": 0.0,
            "q95": 0.0,
            "q99": 0.0,
            "q100": 0.0,
        }
    s = s.astype(float)
    return {
        "q00": float(s.quantile(0.00)),
        "q25": float(s.quantile(0.25)),
        "q50": float(s.quantile(0.50)),
        "q75": float(s.quantile(0.75)),
        "q90": float(s.quantile(0.90)),
        "q95": float(s.quantile(0.95)),
        "q99": float(s.quantile(0.99)),
        "q100": float(s.quantile(1.00)),
    }


def build_mech_features_for_batch(herb_id, disease_ids, herb_to_targets, disease_to_targets, target_weights, device):
    herb_targets = herb_to_targets.get(int(herb_id), set())
    feats = np.zeros((len(disease_ids), 3), dtype=np.float32)

    for i, d_id in enumerate(disease_ids):
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


def main():
    parser = argparse.ArgumentParser(description="Predict herb-disease candidates using trained main HGT model.")
    parser.add_argument(
        "--tag",
        type=str,
        default="hgt_mech_main",
        help="Checkpoint tag from step4_train_main_model.py, e.g. hgt_mech_main / hgt_full_main",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default=None,
        help="Optional explicit checkpoint path. Overrides --tag if provided.",
    )
    parser.add_argument(
        "--use-mechanism",
        action="store_true",
        help="Use mechanism-aware decoder during prediction. For hgt_mech_main / hgt_full_main, enable this.",
    )
    parser.add_argument(
        "--mech-alpha",
        type=float,
        default=0.35,
        help="Weight used in final reranking: Final = Prob * (1 + mech_alpha * norm_mech * has_shared)",
    )
    parser.add_argument(
        "--top-k-per-herb",
        type=int,
        default=300,
        help="Keep top-K candidate diseases per herb before global export.",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Info] Device: {device}")

    graph_path = PROJECT_ROOT / "out/graph/hetero_data.pt"
    mappings_path = PROJECT_ROOT / "out/graph/mappings.pkl"

    if args.ckpt is not None:
        ckpt_path = Path(args.ckpt)
        if not ckpt_path.is_absolute():
            ckpt_path = PROJECT_ROOT / ckpt_path
        tag = ckpt_path.stem.replace("_best", "")
    else:
        tag = args.tag
        ckpt_path = PROJECT_ROOT / "out/models" / f"{tag}_best.pt"

    out_dir = PROJECT_ROOT / "out/predictions"
    all_out_path = out_dir / f"all_predictions_{tag}.tsv"
    safe_out_path = out_dir / f"safe_candidates_weighted_{tag}.tsv"
    safe_diverse_out_path = out_dir / f"safe_candidates_weighted_diverse_{tag}.tsv"
    prob_only_out_path = out_dir / f"baseline_prob_only_{tag}.tsv"
    final_reranked_out_path = out_dir / f"final_reranked_{tag}.tsv"
    mech_loose_out_path = out_dir / f"mechanism_supported_candidates_loose_{tag}.tsv"
    mech_strict_out_path = out_dir / f"mechanism_supported_candidates_strict_{tag}.tsv"
    stats_out_path = out_dir / f"prediction_stats_{tag}.json"

    if not graph_path.exists():
        raise FileNotFoundError(f"Graph file not found: {graph_path}")
    if not mappings_path.exists():
        raise FileNotFoundError(f"Mappings file not found: {mappings_path}")
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {ckpt_path}")

    os.makedirs(out_dir, exist_ok=True)

    print("[1] Loading graph and mappings...")
    data = torch.load(graph_path, weights_only=False)
    mappings_obj = joblib.load(mappings_path)
    herb_map, disease_map, target_map = parse_mappings(mappings_obj)

    if data["target"].x.dim() == 1:
        data["target"].x = data["target"].x.view(-1, 1)
    elif data["target"].x.size(1) > 1:
        data["target"].x = data["target"].x[:, :1]

    td_key = find_td_edge_key(data)
    ht_key = find_ht_edge_key(data)

    data["disease"].x = aggregate_features(
        data["target"].x,
        data[td_key].edge_index,
        data["disease"].num_nodes,
    )

    mp_edge_index_dict = data.edge_index_dict.copy()
    for key in [
        ("herb", "treats", "disease"),
        ("disease", "rev_treats", "herb"),
    ]:
        if key in mp_edge_index_dict:
            del mp_edge_index_dict[key]

    num_nodes_dict = {
        "herb": data["herb"].num_nodes,
        "target": data["target"].num_nodes,
        "disease": data["disease"].num_nodes,
    }

    print("[2] Loading model...")
    model = TCM_HGT(
        hidden_channels=48,
        out_channels=1,
        num_heads=4,
        data_metadata=data.metadata(),
        num_nodes_dict=num_nodes_dict,
        use_mechanism=args.use_mechanism,
    ).to(device)

    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(state)
    model.eval()

    print(f"[OK] Loaded checkpoint: {ckpt_path}")
    print(f"[OK] use_mechanism = {args.use_mechanism}")

    print("[3] Preparing candidate space...")
    all_known_pos_pairs = load_all_positive_pairs_from_graph(data)

    ht_edge_np = data[ht_key].edge_index.cpu().numpy()
    td_edge_np = data[td_key].edge_index.cpu().numpy()

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

    inv_herb_map = {v: k for k, v in herb_map.items()}
    inv_disease_map = {v: k for k, v in disease_map.items()}

    herb_ids = sorted(herb_map.values())
    disease_ids = sorted(disease_map.values())

    data = data.to(device)
    for k in mp_edge_index_dict:
        mp_edge_index_dict[k] = mp_edge_index_dict[k].to(device)

    print("[4] Encoding node embeddings...")
    with torch.no_grad():
        z = model(data.x_dict, mp_edge_index_dict)
        z_herb = z["herb"]
        z_disease = z["disease"]

    print("[5] Predicting all unknown herb-disease pairs...")
    batch_size = 4096
    all_results = []

    model.eval()
    with torch.no_grad():
        for herb_idx, h_id in enumerate(herb_ids, start=1):
            herb_targets = herb_to_targets.get(h_id, set())

            candidate_disease_ids = [
                d_id for d_id in disease_ids
                if (h_id, d_id) not in all_known_pos_pairs
            ]
            if not candidate_disease_ids:
                continue

            herb_records = []

            for start in range(0, len(candidate_disease_ids), batch_size):
                batch_d_ids = candidate_disease_ids[start:start + batch_size]

                herb_tensor = torch.full(
                    (len(batch_d_ids),),
                    h_id,
                    dtype=torch.long,
                    device=device,
                )
                disease_tensor = torch.tensor(
                    batch_d_ids,
                    dtype=torch.long,
                    device=device,
                )
                edge_label_index = torch.stack([herb_tensor, disease_tensor], dim=0)

                if args.use_mechanism:
                    mech_features = build_mech_features_for_batch(
                        h_id, batch_d_ids, herb_to_targets, disease_to_targets, target_weights, device
                    )
                else:
                    mech_features = None

                logits = model.decode(z_herb, z_disease, edge_label_index, mech_features)
                probs = torch.sigmoid(logits).detach().cpu().numpy()
                logits_np = logits.detach().cpu().numpy()

                for d_id, logit, prob in zip(batch_d_ids, logits_np, probs):
                    disease_targets = disease_to_targets.get(d_id, set())
                    shared = herb_targets & disease_targets

                    if shared:
                        shared_weights = [target_weights.get(t, 0.0) for t in shared]
                        raw_shared_weight = float(sum(shared_weights))
                        mech_score = float(np.log1p(raw_shared_weight))
                    else:
                        raw_shared_weight = 0.0
                        mech_score = 0.0

                    herb_records.append(
                        {
                            "Herb_ID": inv_herb_map.get(h_id, h_id),
                            "Disease_ID": inv_disease_map.get(d_id, d_id),
                            "Herb_Idx": int(h_id),
                            "Disease_Idx": int(d_id),
                            "Logit": float(logit),
                            "Prob": float(prob),
                            "Raw_Shared_Weight": float(raw_shared_weight),
                            "Mech_Score": float(mech_score),
                            "Shared_Target_Count": int(len(shared)),
                        }
                    )

            if herb_records:
                herb_df = pd.DataFrame(herb_records)

                herb_df["Mech_Score_Norm"] = normalize_series(herb_df["Mech_Score"])
                herb_df["Has_Shared_Target"] = (herb_df["Shared_Target_Count"] > 0).astype(int)

                mech_alpha = args.mech_alpha
                herb_df["Final"] = herb_df["Prob"] * (
                    1.0 + mech_alpha * herb_df["Mech_Score_Norm"] * herb_df["Has_Shared_Target"]
                )

                herb_df = herb_df.sort_values(
                    by=["Final", "Prob", "Mech_Score", "Shared_Target_Count"],
                    ascending=False
                ).reset_index(drop=True)

                herb_df["Herb_Rank"] = np.arange(1, len(herb_df) + 1)

                herb_df = herb_df.head(args.top_k_per_herb).copy()
                all_results.append(herb_df)

            if herb_idx % 50 == 0:
                kept = sum(len(x) for x in all_results) if all_results else 0
                print(
                    f"[Progress] processed {herb_idx}/{len(herb_ids)} herbs | "
                    f"kept rows: {kept}"
                )

    if not all_results:
        empty_df = pd.DataFrame(columns=[
            "Herb_ID", "Disease_ID", "Herb_Idx", "Disease_Idx", "Logit", "Prob",
            "Raw_Shared_Weight", "Mech_Score", "Mech_Score_Norm",
            "Shared_Target_Count", "Has_Shared_Target", "Final", "Herb_Rank"
        ])
        empty_df.to_csv(all_out_path, sep="\t", index=False)
        empty_df.to_csv(safe_out_path, sep="\t", index=False)
        empty_df.to_csv(safe_diverse_out_path, sep="\t", index=False)
        empty_df.to_csv(prob_only_out_path, sep="\t", index=False)
        empty_df.to_csv(final_reranked_out_path, sep="\t", index=False)
        empty_df.to_csv(mech_loose_out_path, sep="\t", index=False)
        empty_df.to_csv(mech_strict_out_path, sep="\t", index=False)

        stats = {
            "tag": tag,
            "graph_path": str(graph_path),
            "model_path": str(ckpt_path),
            "use_mechanism": bool(args.use_mechanism),
            "ht_edge_key": str(ht_key),
            "td_edge_key": str(td_key),
            "known_positive_pairs_excluded": int(len(all_known_pos_pairs)),
            "all_predictions_rows": 0,
            "safe_candidates_rows": 0,
            "safe_candidates_diverse_rows": 0,
            "baseline_prob_only_rows": 0,
            "final_reranked_rows": 0,
            "mechanism_loose_rows": 0,
            "mechanism_strict_rows": 0,
            "shared_target_ratio_all": 0.0,
            "prob_quantiles": series_quantiles(pd.Series(dtype=float)),
            "mech_quantiles": series_quantiles(pd.Series(dtype=float)),
            "final_quantiles": series_quantiles(pd.Series(dtype=float)),
        }
        with open(stats_out_path, "w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)

        print("[WARN] No predictions generated.")
        return

    print("[6] Saving outputs...")
    df_all = pd.concat(all_results, ignore_index=True)
    df_all = df_all.sort_values(
        by=["Final", "Prob", "Mech_Score", "Shared_Target_Count"],
        ascending=False
    ).reset_index(drop=True)

    df_safe = df_all[
        (df_all["Prob"] >= 0.90) &
        (df_all["Final"] >= 1.00) &
        (df_all["Herb_Rank"] <= 20)
    ].copy()

    df_safe = df_safe.sort_values(
        by=["Final", "Prob", "Mech_Score", "Shared_Target_Count"],
        ascending=False
    ).reset_index(drop=True)

    df_safe_diverse = (
        df_safe.sort_values(
            by=["Final", "Prob", "Mech_Score", "Shared_Target_Count"],
            ascending=False
        )
        .groupby("Herb_ID", as_index=False, group_keys=False)
        .head(5)
        .reset_index(drop=True)
    )

    df_prob_only = df_all.sort_values(
        by=["Prob", "Final", "Mech_Score", "Shared_Target_Count"],
        ascending=False
    ).reset_index(drop=True)

    df_final_reranked = df_all.sort_values(
        by=["Final", "Prob", "Mech_Score", "Shared_Target_Count"],
        ascending=False
    ).reset_index(drop=True)

    df_mech_loose = df_all[
        (df_all["Has_Shared_Target"] == 1) &
        (df_all["Prob"] >= 0.70)
    ].copy()

    df_mech_loose = df_mech_loose.sort_values(
        by=["Final", "Prob", "Mech_Score", "Shared_Target_Count"],
        ascending=False
    ).reset_index(drop=True)

    df_mech_strict = df_all[
        (df_all["Prob"] >= 0.85) &
        (df_all["Shared_Target_Count"] >= 1) &
        (df_all["Herb_Rank"] <= 10)
    ].copy()

    df_mech_strict = df_mech_strict.sort_values(
        by=["Final", "Prob", "Mech_Score", "Shared_Target_Count"],
        ascending=False
    ).reset_index(drop=True)

    df_all.to_csv(all_out_path, sep="\t", index=False)
    df_safe.to_csv(safe_out_path, sep="\t", index=False)
    df_safe_diverse.to_csv(safe_diverse_out_path, sep="\t", index=False)
    df_prob_only.to_csv(prob_only_out_path, sep="\t", index=False)
    df_final_reranked.to_csv(final_reranked_out_path, sep="\t", index=False)
    df_mech_loose.to_csv(mech_loose_out_path, sep="\t", index=False)
    df_mech_strict.to_csv(mech_strict_out_path, sep="\t", index=False)

    shared_target_ratio_all = float((df_all["Has_Shared_Target"] == 1).mean()) if len(df_all) > 0 else 0.0

    stats = {
        "tag": tag,
        "graph_path": str(graph_path),
        "model_path": str(ckpt_path),
        "use_mechanism": bool(args.use_mechanism),
        "ht_edge_key": str(ht_key),
        "td_edge_key": str(td_key),
        "known_positive_pairs_excluded": int(len(all_known_pos_pairs)),
        "all_predictions_rows": int(len(df_all)),
        "safe_candidates_rows": int(len(df_safe)),
        "safe_candidates_diverse_rows": int(len(df_safe_diverse)),
        "baseline_prob_only_rows": int(len(df_prob_only)),
        "final_reranked_rows": int(len(df_final_reranked)),
        "mechanism_loose_rows": int(len(df_mech_loose)),
        "mechanism_strict_rows": int(len(df_mech_strict)),
        "unique_herbs_in_all": int(df_all["Herb_ID"].nunique()),
        "unique_diseases_in_all": int(df_all["Disease_ID"].nunique()),
        "unique_herbs_in_safe": int(df_safe["Herb_ID"].nunique()) if len(df_safe) > 0 else 0,
        "unique_diseases_in_safe": int(df_safe["Disease_ID"].nunique()) if len(df_safe) > 0 else 0,
        "unique_herbs_in_safe_diverse": int(df_safe_diverse["Herb_ID"].nunique()) if len(df_safe_diverse) > 0 else 0,
        "unique_diseases_in_safe_diverse": int(df_safe_diverse["Disease_ID"].nunique()) if len(df_safe_diverse) > 0 else 0,
        "final_score_mean_all": float(df_all["Final"].mean()),
        "prob_mean_all": float(df_all["Prob"].mean()),
        "mech_score_mean_all": float(df_all["Mech_Score"].mean()),
        "shared_target_ratio_all": shared_target_ratio_all,
        "prob_quantiles": series_quantiles(df_all["Prob"]),
        "mech_quantiles": series_quantiles(df_all["Mech_Score"]),
        "final_quantiles": series_quantiles(df_all["Final"]),
    }

    with open(stats_out_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print(f"[OK] Graph          : {graph_path}")
    print(f"[OK] Model          : {ckpt_path}")
    print(f"[OK] use_mechanism  : {args.use_mechanism}")
    print(f"[OK] HT edge        : {ht_key}")
    print(f"[OK] TD edge        : {td_key}")
    print(f"[OK] Excluded known : {len(all_known_pos_pairs)} positive herb-disease pairs")
    print(f"[OK] Saved all predictions      : {all_out_path} ({len(df_all)} rows)")
    print(f"[OK] Saved safe candidates      : {safe_out_path} ({len(df_safe)} rows)")
    print(f"[OK] Saved diverse safe cand.   : {safe_diverse_out_path} ({len(df_safe_diverse)} rows)")
    print(f"[OK] Saved prob-only baseline   : {prob_only_out_path} ({len(df_prob_only)} rows)")
    print(f"[OK] Saved final reranked       : {final_reranked_out_path} ({len(df_final_reranked)} rows)")
    print(f"[OK] Saved mech loose candidates: {mech_loose_out_path} ({len(df_mech_loose)} rows)")
    print(f"[OK] Saved mech strict cand.    : {mech_strict_out_path} ({len(df_mech_strict)} rows)")
    print(f"[OK] Saved stats                : {stats_out_path}")

    print("\nTop 10 diverse predictions (max 1 per herb):")
    df_top_diverse = (
        df_all.sort_values(
            by=["Final", "Prob", "Mech_Score", "Shared_Target_Count"],
            ascending=False
        )
        .groupby("Herb_ID", as_index=False, group_keys=False)
        .head(1)
        .head(10)
    )
    print(
        df_top_diverse[
            [
                "Herb_ID", "Disease_ID", "Prob", "Mech_Score",
                "Shared_Target_Count", "Final", "Herb_Rank"
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
