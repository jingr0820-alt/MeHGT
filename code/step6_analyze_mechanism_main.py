#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import json
import joblib
from pathlib import Path

import pandas as pd
import torch


def find_project_root() -> Path:
    here = Path(__file__).resolve().parent
    for candidate in [here, here.parent]:
        if (candidate / "data").exists() and (candidate / "code").exists():
            return candidate
    raise FileNotFoundError("Cannot locate the repository root containing code/ and data/.")


PROJECT_ROOT = find_project_root()


def read_table_auto(path: Path) -> pd.DataFrame:
    try:
        df = pd.read_csv(path, sep="\t")
        if len(df.columns) <= 1:
            df = pd.read_csv(path)
    except Exception:
        df = pd.read_csv(path)
    return df


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


def find_ht_edge_key(data):
    candidates = [
        ("herb", "interacts", "target"),
        ("herb", "interacts_with", "target"),
        ("herb", "targets", "target"),
    ]
    for key in candidates:
        if key in data.edge_types:
            return key
    raise KeyError(f"Cannot find herb->target edge type. Available: {data.edge_types}")


def find_td_edge_key(data):
    candidates = [
        ("target", "rev_associates", "disease"),
        ("target", "associated_with", "disease"),
    ]
    for key in candidates:
        if key in data.edge_types:
            return key
    raise KeyError(f"Cannot find target->disease edge type. Available: {data.edge_types}")


def load_idf_dict(idf_path: Path) -> dict:
    if not idf_path.exists():
        print(f"[WARN] IDF file not found: {idf_path}. Herb_Target_IDF will be filled with 0.0")
        return {}

    idf_df = pd.read_csv(idf_path)

    target_col = None
    for c in ["target", "Target", "EntrezID", "entrez_id", "entrez"]:
        if c in idf_df.columns:
            target_col = c
            break
    if target_col is None:
        raise KeyError(f"Cannot find target column in IDF file: {idf_path}")

    idf_col = None
    for c in ["idf", "IDF"]:
        if c in idf_df.columns:
            idf_col = c
            break
    if idf_col is None:
        raise KeyError(f"Cannot find idf column in IDF file: {idf_path}")

    return dict(zip(idf_df[target_col].astype(str), idf_df[idf_col].astype(float)))


def choose_prediction_file(pred_dir: Path, rank_mode: str, tag: str, pred_file: str = None) -> Path:
    if pred_file is not None:
        path = Path(pred_file)
        if not path.is_absolute():
            path = pred_dir / pred_file
        if not path.exists():
            raise FileNotFoundError(f"Prediction file not found: {path}")
        return path

    preferred = []
    if rank_mode == "final":
        preferred = [
            pred_dir / f"safe_candidates_weighted_diverse_{tag}.tsv",
            pred_dir / f"safe_candidates_weighted_{tag}.tsv",
            pred_dir / f"final_reranked_{tag}.tsv",
            pred_dir / f"all_predictions_{tag}.tsv",
        ]
    else:
        preferred = [
            pred_dir / f"baseline_prob_only_{tag}.tsv",
            pred_dir / f"all_predictions_{tag}.tsv",
        ]

    for p in preferred:
        if p.exists():
            return p

    # fallback: any latest file matching tag
    candidates = []
    if rank_mode == "prob":
        candidates.extend(pred_dir.glob(f"*prob_only*{tag}*.tsv"))
        candidates.extend(pred_dir.glob(f"*prob_only*{tag}*.csv"))
        candidates.extend(pred_dir.glob(f"*all_predictions*{tag}*.tsv"))
        candidates.extend(pred_dir.glob(f"*all_predictions*{tag}*.csv"))
    else:
        candidates.extend(pred_dir.glob(f"*diverse*{tag}*.tsv"))
        candidates.extend(pred_dir.glob(f"*diverse*{tag}*.csv"))
        candidates.extend(pred_dir.glob(f"*weighted*{tag}*.tsv"))
        candidates.extend(pred_dir.glob(f"*weighted*{tag}*.csv"))
        candidates.extend(pred_dir.glob(f"*safe_candidates*{tag}*.tsv"))
        candidates.extend(pred_dir.glob(f"*safe_candidates*{tag}*.csv"))
        candidates.extend(pred_dir.glob(f"*final_reranked*{tag}*.tsv"))
        candidates.extend(pred_dir.glob(f"*final_reranked*{tag}*.csv"))

    candidates = [p for p in candidates if p.exists()]
    if not candidates:
        raise FileNotFoundError(f"No prediction files found in {pred_dir} for tag={tag}, rank_mode={rank_mode}")

    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def find_col(df: pd.DataFrame, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


def summarize_top_targets(df_mech: pd.DataFrame, top_n=3):
    if df_mech is None or len(df_mech) == 0:
        return ""
    return ";".join(df_mech["Target_ID"].astype(str).head(top_n).tolist())


def main():
    parser = argparse.ArgumentParser(description="Mechanism analysis for herb-disease predictions")
    parser.add_argument("--tag", type=str, default="hgt_mech_main")
    parser.add_argument("--rank-mode", choices=["final", "prob"], default="final")
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--pred-file", type=str, default=None)
    parser.add_argument("--herb-id", type=str, default=None)
    parser.add_argument("--disease-id", type=str, default=None)
    args = parser.parse_args()

    graph_path = PROJECT_ROOT / "out/graph/hetero_data.pt"
    mappings_path = PROJECT_ROOT / "out/graph/mappings.pkl"
    idf_path = PROJECT_ROOT / "out/filters/target_idf.csv"
    pred_dir = PROJECT_ROOT / "out/predictions"
    out_dir = PROJECT_ROOT / "out/mechanism"

    if not graph_path.exists():
        raise FileNotFoundError(f"Graph file not found: {graph_path}")
    if not mappings_path.exists():
        raise FileNotFoundError(f"Mappings file not found: {mappings_path}")

    os.makedirs(out_dir, exist_ok=True)

    print("[1] Loading graph / mappings / IDF...")
    data = torch.load(graph_path, weights_only=False)
    mappings_obj = joblib.load(mappings_path)
    herb_map, disease_map, target_map = parse_mappings(mappings_obj)
    idf_dict = load_idf_dict(idf_path)

    id2target = {v: k for k, v in target_map.items()}

    ht_key = find_ht_edge_key(data)
    td_key = find_td_edge_key(data)

    ht_edges = data[ht_key].edge_index.cpu().numpy()
    herb_targets = {}
    for h, t in zip(ht_edges[0], ht_edges[1]):
        herb_targets.setdefault(int(h), set()).add(int(t))

    td_edges = data[td_key].edge_index.cpu().numpy()
    disease_targets = {}
    for t, d in zip(td_edges[0], td_edges[1]):
        disease_targets.setdefault(int(d), set()).add(int(t))

    if args.herb_id and args.disease_id:
        cases_df = pd.DataFrame([{"Herb_ID": str(args.herb_id), "Disease_ID": str(args.disease_id)}])
        source_desc = "manual herb-disease input"
    else:
        pred_file = choose_prediction_file(pred_dir, args.rank_mode, args.tag, args.pred_file)
        pred_df = read_table_auto(pred_file)

        herb_col = find_col(pred_df, ["Herb_ID", "herb_id", "herb"])
        disease_col = find_col(pred_df, ["Disease_ID", "disease_id", "disease"])
        prob_col = find_col(pred_df, ["Prob", "prob"])
        final_col = find_col(pred_df, ["Final", "final"])
        mech_col = find_col(pred_df, ["Mech_Score", "mech_score"])
        shared_cnt_col = find_col(pred_df, ["Shared_Target_Count", "shared_target_count"])
        herb_rank_col = find_col(pred_df, ["Herb_Rank", "herb_rank"])

        if herb_col is None or disease_col is None:
            raise ValueError(
                f"Prediction file missing herb/disease columns: {pred_file}\n"
                f"Columns = {list(pred_df.columns)}"
            )

        sort_cols = []
        ascending = []

        if args.rank_mode == "final" and final_col is not None:
            sort_cols.append(final_col)
            ascending.append(False)
        elif prob_col is not None:
            sort_cols.append(prob_col)
            ascending.append(False)

        if prob_col is not None and prob_col not in sort_cols:
            sort_cols.append(prob_col)
            ascending.append(False)
        if mech_col is not None:
            sort_cols.append(mech_col)
            ascending.append(False)
        if shared_cnt_col is not None:
            sort_cols.append(shared_cnt_col)
            ascending.append(False)
        if herb_rank_col is not None:
            sort_cols.append(herb_rank_col)
            ascending.append(True)

        if sort_cols:
            pred_df = pred_df.sort_values(sort_cols, ascending=ascending).reset_index(drop=True)

        keep_cols = [herb_col, disease_col]
        for c in [prob_col, final_col, mech_col, shared_cnt_col, herb_rank_col]:
            if c is not None and c not in keep_cols:
                keep_cols.append(c)

        cases_df = pred_df[keep_cols].head(args.top_k).copy()
        cases_df = cases_df.rename(columns={herb_col: "Herb_ID", disease_col: "Disease_ID"})
        source_desc = f"prediction file: {pred_file}"

    print(f"[2] Using cases from: {source_desc}")
    print(f"HT edge: {ht_key}")
    print(f"TD edge: {td_key}")

    summary_rows = []
    shared_counts = []
    shared_idf_sums = []

    tag_out_dir = out_dir / args.tag
    tag_out_dir.mkdir(parents=True, exist_ok=True)

    for _, row in cases_df.iterrows():
        h_id = str(row["Herb_ID"])
        d_id = str(row["Disease_ID"])

        h_idx = herb_map.get(h_id)
        d_idx = disease_map.get(d_id)

        if h_idx is None or d_idx is None:
            print(f"[Skip] {h_id} - {d_id}: not found in graph mappings.")
            continue

        shared_t_indices = herb_targets.get(h_idx, set()) & disease_targets.get(d_idx, set())

        shared_list = []
        for t_idx in sorted(shared_t_indices):
            target_id = str(id2target.get(t_idx, t_idx))
            herb_target_idf = float(idf_dict.get(target_id, 0.0))
            shared_list.append(
                {
                    "Herb_ID": h_id,
                    "Disease_ID": d_id,
                    "Target_Idx": int(t_idx),
                    "Target_ID": target_id,
                    "Herb_Target_IDF": herb_target_idf,
                }
            )

        summary_item = {
            "Herb_ID": h_id,
            "Disease_ID": d_id,
            "Shared_Target_Count_Recomputed": int(len(shared_list)),
        }

        for c in ["Prob", "Final", "Mech_Score", "Shared_Target_Count", "Herb_Rank"]:
            if c in row.index:
                summary_item[c] = row[c]

        if shared_list:
            df_mech = pd.DataFrame(shared_list).sort_values(
                ["Herb_Target_IDF", "Target_ID"], ascending=[False, True]
            ).reset_index(drop=True)

            idf_sum = float(df_mech["Herb_Target_IDF"].sum())
            idf_mean = float(df_mech["Herb_Target_IDF"].mean())
            top_target_id = df_mech.iloc[0]["Target_ID"]
            top_target_idf = float(df_mech.iloc[0]["Herb_Target_IDF"])
            top3_targets = summarize_top_targets(df_mech, top_n=3)

            summary_item["Top_Target_ID"] = top_target_id
            summary_item["Top_Target_IDF"] = top_target_idf
            summary_item["Shared_Target_IDF_Sum"] = idf_sum
            summary_item["Shared_Target_IDF_Mean"] = idf_mean
            summary_item["Top3_Target_IDs"] = top3_targets

            out_path = tag_out_dir / f"{h_id}__{d_id}_shared_targets.tsv"
            df_mech.to_csv(out_path, sep="\t", index=False)

            shared_counts.append(len(df_mech))
            shared_idf_sums.append(idf_sum)

            print(f"\nHerb: {h_id} | Disease: {d_id}")
            if "Prob" in summary_item:
                print(f"Prob={summary_item['Prob']:.6f}", end=" | ")
            if "Final" in summary_item:
                print(f"Final={summary_item['Final']:.6f}", end=" | ")
            if "Mech_Score" in summary_item:
                print(f"Mech_Score={summary_item['Mech_Score']:.6f}", end=" | ")
            print(
                f"Shared={len(df_mech)} | "
                f"IDF_Sum={idf_sum:.4f} | "
                f"Top3={top3_targets}"
            )
            print(df_mech.head(20).to_string(index=False))
            print(f"Saved: {out_path}")
        else:
            summary_item["Top_Target_ID"] = None
            summary_item["Top_Target_IDF"] = None
            summary_item["Shared_Target_IDF_Sum"] = 0.0
            summary_item["Shared_Target_IDF_Mean"] = 0.0
            summary_item["Top3_Target_IDs"] = ""

            shared_counts.append(0)
            shared_idf_sums.append(0.0)

            print(f"\nHerb: {h_id} | Disease: {d_id}")
            if "Prob" in summary_item:
                print(f"Prob={summary_item['Prob']:.6f}", end=" | ")
            if "Final" in summary_item:
                print(f"Final={summary_item['Final']:.6f}", end=" | ")
            if "Mech_Score" in summary_item:
                print(f"Mech_Score={summary_item['Mech_Score']:.6f}", end=" | ")
            print("No shared targets found.")

        summary_rows.append(summary_item)

    summary_df = pd.DataFrame(summary_rows)

    if len(summary_df) > 0:
        sort_cols = []
        ascending = []

        if "Final" in summary_df.columns:
            sort_cols.append("Final")
            ascending.append(False)
        elif "Prob" in summary_df.columns:
            sort_cols.append("Prob")
            ascending.append(False)

        if "Shared_Target_IDF_Sum" in summary_df.columns:
            sort_cols.append("Shared_Target_IDF_Sum")
            ascending.append(False)

        if "Shared_Target_Count_Recomputed" in summary_df.columns:
            sort_cols.append("Shared_Target_Count_Recomputed")
            ascending.append(False)

        if sort_cols:
            summary_df = summary_df.sort_values(sort_cols, ascending=ascending).reset_index(drop=True)

    def mech_strength_label(row):
        cnt = int(row.get("Shared_Target_Count_Recomputed", 0))
        idf_sum = float(row.get("Shared_Target_IDF_Sum", 0.0))

        if cnt >= 40 and idf_sum >= 180:
            return "very_strong"
        if cnt >= 25 and idf_sum >= 100:
            return "strong"
        if cnt >= 10 and idf_sum >= 40:
            return "moderate"
        if cnt >= 1:
            return "weak"
        return "none"

    summary_df["Mechanism_Strength"] = summary_df.apply(mech_strength_label, axis=1)

    summary_path = tag_out_dir / f"mechanism_summary_{args.tag}.tsv"
    summary_df.to_csv(summary_path, sep="\t", index=False)

    compact_cols = [
        "Herb_ID",
        "Disease_ID",
        "Prob",
        "Final",
        "Mech_Score",
        "Shared_Target_Count_Recomputed",
        "Shared_Target_IDF_Sum",
        "Mechanism_Strength",
        "Top3_Target_IDs",
    ]
    compact_cols = [c for c in compact_cols if c in summary_df.columns]

    compact_df = summary_df[compact_cols].copy()
    compact_path = tag_out_dir / f"mechanism_summary_compact_{args.tag}.tsv"
    compact_df.to_csv(compact_path, sep="\t", index=False)

    disease_diverse_df = (
        summary_df.sort_values(
            by=["Shared_Target_IDF_Sum", "Final", "Prob", "Shared_Target_Count_Recomputed"],
            ascending=False
        )
        .groupby("Disease_ID", as_index=False, group_keys=False)
        .head(1)
        .reset_index(drop=True)
    )
    disease_diverse_path = tag_out_dir / f"mechanism_summary_disease_diverse_{args.tag}.tsv"
    disease_diverse_df.to_csv(disease_diverse_path, sep="\t", index=False)

    shortlist_df = summary_df.copy()
    if "Mechanism_Strength" in shortlist_df.columns:
        shortlist_df = shortlist_df[
            shortlist_df["Mechanism_Strength"].isin(["very_strong", "strong", "moderate", "weak"])
        ].copy()

    shortlist_df = shortlist_df.sort_values(
        by=["Shared_Target_IDF_Sum", "Final", "Prob", "Shared_Target_Count_Recomputed"],
        ascending=False
    ).reset_index(drop=True)

    shortlist_df = (
        shortlist_df.groupby("Disease_ID", as_index=False, group_keys=False)
        .head(1)
        .reset_index(drop=True)
    )

    shortlist_df = (
        shortlist_df.sort_values(
            by=["Shared_Target_IDF_Sum", "Final", "Prob", "Shared_Target_Count_Recomputed"],
            ascending=False
        )
        .groupby("Herb_ID", as_index=False, group_keys=False)
        .head(3)
        .reset_index(drop=True)
    )

    shortlist_path = tag_out_dir / f"mechanism_summary_shortlist_{args.tag}.tsv"
    shortlist_df.to_csv(shortlist_path, sep="\t", index=False)

    meta = {
        "tag": args.tag,
        "graph_path": str(graph_path),
        "mappings_path": str(mappings_path),
        "idf_path": str(idf_path),
        "source": source_desc,
        "rank_mode": args.rank_mode,
        "top_k": int(args.top_k),
        "cases_analyzed": int(len(summary_df)),
        "ht_edge": str(ht_key),
        "td_edge": str(td_key),
        "summary_path": str(summary_path),
    }

    meta_path = tag_out_dir / f"mechanism_summary_meta_{args.tag}.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    case_stats = {
        "tag": args.tag,
        "cases_analyzed": int(len(summary_df)),
        "cases_with_shared_targets": int(sum(1 for x in shared_counts if x > 0)),
        "shared_target_count_mean": float(sum(shared_counts) / len(shared_counts)) if shared_counts else 0.0,
        "shared_target_count_max": int(max(shared_counts)) if shared_counts else 0,
        "shared_target_idf_sum_mean": float(sum(shared_idf_sums) / len(shared_idf_sums)) if shared_idf_sums else 0.0,
        "shared_target_idf_sum_max": float(max(shared_idf_sums)) if shared_idf_sums else 0.0,
        "unique_herbs": int(summary_df["Herb_ID"].nunique()) if len(summary_df) > 0 else 0,
        "unique_diseases": int(summary_df["Disease_ID"].nunique()) if len(summary_df) > 0 else 0,
    }

    stats_path = tag_out_dir / f"mechanism_case_stats_{args.tag}.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(case_stats, f, ensure_ascii=False, indent=2)

    print("\n[OK] Saved:")
    print(f" - {summary_path}")
    print(f" - {meta_path}")
    print(f" - {stats_path}")
    print(f" - {compact_path}")
    print(f" - {disease_diverse_path}")
    print(f" - {shortlist_path}")


if __name__ == "__main__":
    main()
