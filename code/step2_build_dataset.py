import argparse
import gc
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_mechanism_inputs():
    d9 = pd.read_csv(
        PROJECT_ROOT / "data" / "D9_CHP_InChIKey.tsv",
        sep="\t",
        usecols=["CHP_ID", "InChIKey"],
        dtype=str,
    ).dropna()
    d13 = pd.read_csv(
        PROJECT_ROOT / "data" / "D13_InChIKey_EntrezID.tsv",
        sep="\t",
        usecols=["InChIKey", "EntrezID"],
        dtype=str,
    ).dropna()
    d23 = pd.read_csv(
        PROJECT_ROOT / "data" / "D23_MeSH_targets.tsv",
        sep="\t",
        usecols=["MeSH", "EntrezID"],
        dtype=str,
    ).dropna()
    idf = pd.read_csv(PROJECT_ROOT / "out" / "filters" / "target_idf.csv", dtype={"target": str})

    d9 = d9.drop_duplicates()
    d13 = d13.drop_duplicates()
    d23 = d23.drop_duplicates()
    idf = idf.dropna(subset=["target", "idf"]).drop_duplicates(subset=["target"]).copy()
    idf["idf"] = idf["idf"].astype(np.float32)

    return d9, d13, d23, idf


def build_mechanism_matrices(selected_meshes=None):
    print("1/5: Loading herb-target and disease-target mechanism inputs...")
    d9, d13, d23, idf = load_mechanism_inputs()

    idf_map = dict(zip(idf["target"], idf["idf"]))
    ht = (
        d9.merge(d13, on="InChIKey", how="inner")[["CHP_ID", "EntrezID"]]
        .dropna()
        .drop_duplicates()
    )
    ht = ht[ht["EntrezID"].isin(idf_map)].copy()
    ht["idf"] = ht["EntrezID"].map(idf_map).astype(np.float32)

    herb_degree = ht.groupby("CHP_ID")["EntrezID"].nunique().to_dict()
    herbs = sorted(ht["CHP_ID"].unique())
    targets = sorted(ht["EntrezID"].unique())
    herb_map = {h: i for i, h in enumerate(herbs)}
    target_map = {t: i for i, t in enumerate(targets)}
    inv_herb = np.array(herbs, dtype=object)
    herb_degree_arr = np.array([herb_degree.get(h, 0) for h in herbs], dtype=np.float32)

    h_rows = ht["CHP_ID"].map(herb_map).to_numpy(dtype=np.int32)
    h_cols = ht["EntrezID"].map(target_map).to_numpy(dtype=np.int32)
    h_data = ht["idf"].to_numpy(dtype=np.float32)
    herb_target = coo_matrix(
        (h_data, (h_rows, h_cols)),
        shape=(len(herbs), len(targets)),
        dtype=np.float32,
    ).tocsr()

    d23 = d23[d23["EntrezID"].isin(target_map)].copy()
    if selected_meshes is not None:
        selected_meshes = set(str(x) for x in selected_meshes)
        d23 = d23[d23["MeSH"].isin(selected_meshes)].copy()

    diseases = sorted(d23["MeSH"].unique())
    disease_map = {d: i for i, d in enumerate(diseases)}
    d_rows = d23["MeSH"].map(disease_map).to_numpy(dtype=np.int32)
    d_cols = d23["EntrezID"].map(target_map).to_numpy(dtype=np.int32)
    d_data = np.ones(len(d_rows), dtype=np.float32)
    disease_target = coo_matrix(
        (d_data, (d_rows, d_cols)),
        shape=(len(diseases), len(targets)),
        dtype=np.float32,
    ).tocsr()
    disease_target.data[:] = 1.0

    print(f"   - Herbs: {len(herbs)}")
    print(f"   - Targets: {len(targets)}")
    print(f"   - Diseases with target knowledge: {len(diseases)}")

    del d9, d13, d23, idf, h_rows, h_cols, h_data, d_rows, d_cols, d_data
    gc.collect()

    return {
        "herb_target": herb_target,
        "disease_target": disease_target,
        "diseases": diseases,
        "herb_degree_arr": herb_degree_arr,
        "inv_herb": inv_herb,
        "herb_map": herb_map,
        "target_map": target_map,
    }


def build_external_positive_pairs():
    print("2/5: Building external positives from CPM indications...")
    d4 = pd.read_csv(
        PROJECT_ROOT / "data" / "D4_CPM_CHP.tsv",
        sep="\t",
        usecols=["CPM_ID", "CHP_ID"],
        dtype=str,
    ).dropna()
    d5 = pd.read_csv(
        PROJECT_ROOT / "data" / "D5_CPM_ICD11.tsv",
        sep="\t",
        usecols=["CPM_ID", "ICD11_code"],
        dtype=str,
    ).dropna()
    d20 = pd.read_csv(
        PROJECT_ROOT / "data" / "D20_ICD11_MeSH.tsv",
        sep="\t",
        usecols=["ICD11_code", "MeSH"],
        dtype=str,
    ).dropna()

    pos_raw = (
        d4.drop_duplicates()
        .merge(d5.drop_duplicates(), on="CPM_ID", how="inner")
        .merge(d20.drop_duplicates(), on="ICD11_code", how="inner")
    )

    pos_pair = (
        pos_raw.groupby(["MeSH", "CHP_ID"], as_index=False)
        .agg(
            support_cpm_count=("CPM_ID", "nunique"),
            support_icd11_count=("ICD11_code", "nunique"),
        )
        .sort_values(["MeSH", "support_cpm_count", "support_icd11_count", "CHP_ID"], ascending=[True, False, False, True])
        .reset_index(drop=True)
    )
    pos_pair["label"] = 1
    pos_pair["label_source"] = "external_cpm_indication"

    print(f"   - External positive herb-disease pairs: {len(pos_pair)}")
    print(f"   - Unique diseases: {pos_pair['MeSH'].nunique()}")
    print(f"   - Unique herbs: {pos_pair['CHP_ID'].nunique()}")
    return pos_pair


def build_strict_external_dataset(args):
    pos_df = build_external_positive_pairs()
    mech = build_mechanism_matrices(selected_meshes=pos_df["MeSH"].unique())

    valid_disease_set = set(mech["diseases"])
    valid_herb_set = set(mech["inv_herb"].tolist())
    pos_df = pos_df[pos_df["MeSH"].isin(valid_disease_set) & pos_df["CHP_ID"].isin(valid_herb_set)].copy()

    disease_positive_counts = pos_df.groupby("MeSH")["CHP_ID"].nunique()
    keep_meshes = disease_positive_counts[disease_positive_counts >= args.min_pos_per_disease].index
    pos_df = pos_df[pos_df["MeSH"].isin(keep_meshes)].copy()
    pos_df["disease_positive_count"] = pos_df["MeSH"].map(disease_positive_counts).astype(int)
    pos_df["mechanism_score"] = np.nan
    pos_df["total_degree"] = pos_df["CHP_ID"].map(
        {herb_id: float(mech["herb_degree_arr"][idx]) for idx, herb_id in enumerate(mech["inv_herb"])}
    )

    print("3/5: Mining mechanism-aware hard negatives without using mechanism as labels...")
    pos_by_disease = pos_df.groupby("MeSH")["CHP_ID"].apply(set).to_dict()
    disease_to_idx = {d: i for i, d in enumerate(mech["diseases"])}
    herb_target_t = mech["herb_target"].T.tocsr()

    neg_records = []
    disease_summary = []
    block_size = 256
    disease_list = sorted(pos_by_disease)

    for block_start in range(0, len(disease_list), block_size):
        block_diseases = disease_list[block_start:block_start + block_size]
        block_indices = [disease_to_idx[d] for d in block_diseases]
        block_scores = (mech["disease_target"][block_indices] @ herb_target_t).tocsr()

        for local_i, disease_id in enumerate(block_diseases):
            row = block_scores.getrow(local_i)
            pos_herb_ids = pos_by_disease[disease_id]
            pos_count = len(pos_herb_ids)
            desired_neg = pos_count * args.neg_per_positive

            if row.nnz == 0 or desired_neg == 0:
                disease_summary.append(
                    {
                        "MeSH": disease_id,
                        "positive_count": pos_count,
                        "negative_count": 0,
                        "candidate_pool_size": 0,
                    }
                )
                continue

            herb_idx = row.indices
            scores = row.data.astype(np.float32, copy=False)
            positive_mask = scores > 0
            herb_idx = herb_idx[positive_mask]
            scores = scores[positive_mask]

            if len(herb_idx) == 0:
                disease_summary.append(
                    {
                        "MeSH": disease_id,
                        "positive_count": pos_count,
                        "negative_count": 0,
                        "candidate_pool_size": 0,
                    }
                )
                continue

            order = np.lexsort((herb_idx, -scores))
            ordered_herbs = herb_idx[order]
            ordered_scores = scores[order]

            ranked_candidates = []
            for rank, (h_idx, score) in enumerate(zip(ordered_herbs, ordered_scores), start=1):
                herb_id = str(mech["inv_herb"][h_idx])
                if herb_id in pos_herb_ids:
                    continue
                ranked_candidates.append((rank, h_idx, score, herb_id))

            candidate_slice = [
                item for item in ranked_candidates
                if args.neg_rank_start <= item[0] <= args.neg_rank_end
            ]
            if len(candidate_slice) < desired_neg:
                candidate_slice = ranked_candidates[: max(desired_neg * 3, desired_neg)]

            if len(candidate_slice) == 0:
                disease_summary.append(
                    {
                        "MeSH": disease_id,
                        "positive_count": pos_count,
                        "negative_count": 0,
                        "candidate_pool_size": 0,
                    }
                )
                continue

            pos_degrees = pos_df.loc[pos_df["MeSH"] == disease_id, "total_degree"].astype(float).to_numpy()
            avg_pos_degree = float(pos_degrees.mean()) if len(pos_degrees) > 0 else 0.0
            candidate_herb_idx = np.array([x[1] for x in candidate_slice], dtype=np.int32)
            candidate_scores = np.array([x[2] for x in candidate_slice], dtype=np.float32)
            candidate_ranks = np.array([x[0] for x in candidate_slice], dtype=np.int32)
            candidate_degree_delta = np.abs(mech["herb_degree_arr"][candidate_herb_idx] - avg_pos_degree)
            neg_order = np.lexsort((candidate_herb_idx, candidate_ranks, -candidate_scores, candidate_degree_delta))
            chosen = [candidate_slice[i] for i in neg_order[:desired_neg]]

            for rank, h_idx, score, herb_id in chosen:
                neg_records.append(
                    {
                        "MeSH": disease_id,
                        "CHP_ID": herb_id,
                        "label": 0,
                        "label_source": "external_cpm_indication",
                        "support_cpm_count": 0,
                        "support_icd11_count": 0,
                        "disease_positive_count": pos_count,
                        "mechanism_score": float(score),
                        "total_degree": float(mech["herb_degree_arr"][h_idx]),
                        "neg_rank_reference": int(rank),
                    }
                )

            disease_summary.append(
                {
                    "MeSH": disease_id,
                    "positive_count": pos_count,
                    "negative_count": len(chosen),
                    "candidate_pool_size": len(candidate_slice),
                }
            )

        processed = min(block_start + block_size, len(disease_list))
        print(f"   - Processed {processed}/{len(disease_list)} diseases")

    neg_df = pd.DataFrame(neg_records)
    if pos_df.empty or neg_df.empty:
        raise ValueError("Strict external dataset is empty; please loosen filters or inspect source data.")

    print("4/5: Splitting by disease for zero-shot evaluation...")
    df = pd.concat([pos_df, neg_df], ignore_index=True)
    df = df.drop_duplicates(subset=["MeSH", "CHP_ID", "label"]).copy()

    meshes = np.array(sorted(df["MeSH"].unique()), dtype=object)
    np.random.seed(args.seed)
    np.random.shuffle(meshes)
    n = len(meshes)
    train_m = set(meshes[: int(n * 0.7)])
    val_m = set(meshes[int(n * 0.7): int(n * 0.8)])
    df["split"] = df["MeSH"].apply(
        lambda x: "train" if x in train_m else ("val" if x in val_m else "test")
    )

    disease_summary_df = pd.DataFrame(disease_summary).sort_values(["positive_count", "negative_count", "MeSH"], ascending=[False, False, True])
    meta = {
        "dataset_mode": "external_cpm_strict",
        "seed": int(args.seed),
        "label_source": "D4_CPM_CHP + D5_CPM_ICD11 + D20_ICD11_MeSH",
        "negative_strategy": {
            "type": "mechanism_aware_hard_negative",
            "neg_per_positive": int(args.neg_per_positive),
            "rank_window": [int(args.neg_rank_start), int(args.neg_rank_end)],
            "degree_balanced": True,
        },
        "split_strategy": "zero_shot_by_disease_mesh",
        "min_pos_per_disease": int(args.min_pos_per_disease),
        "num_samples": int(len(df)),
        "num_positive": int((df["label"] == 1).sum()),
        "num_negative": int((df["label"] == 0).sum()),
        "num_diseases": int(df["MeSH"].nunique()),
        "num_herbs": int(df["CHP_ID"].nunique()),
    }

    return df, meta, disease_summary_df


def build_legacy_overlap_dataset(args):
    print("2/5: Building legacy pseudo-label dataset from shared-target overlap...")
    mech = build_mechanism_matrices()
    herb_target_t = mech["herb_target"].T.tocsr()
    pos_records = []
    neg_records = []
    block_size = 256

    for block_start in range(0, len(mech["diseases"]), block_size):
        block_end = min(block_start + block_size, len(mech["diseases"]))
        block_scores = (mech["disease_target"][block_start:block_end] @ herb_target_t).tocsr()

        for local_i in range(block_scores.shape[0]):
            disease_idx = block_start + local_i
            disease_id = mech["diseases"][disease_idx]
            row = block_scores.getrow(local_i)
            if row.nnz == 0:
                continue

            herb_idx = row.indices
            scores = row.data.astype(np.float32, copy=False)
            positive_mask = scores > 0
            herb_idx = herb_idx[positive_mask]
            scores = scores[positive_mask]
            if len(herb_idx) == 0:
                continue

            order = np.lexsort((herb_idx, -scores))
            ordered_herbs = herb_idx[order]
            ordered_scores = scores[order]

            pos_herbs = ordered_herbs[:20]
            pos_scores = ordered_scores[:20]
            for h_idx, score in zip(pos_herbs, pos_scores):
                pos_records.append(
                    {
                        "MeSH": disease_id,
                        "CHP_ID": mech["inv_herb"][h_idx],
                        "mechanism_score": float(score),
                        "total_degree": float(mech["herb_degree_arr"][h_idx]),
                        "label": 1,
                        "label_source": "legacy_shared_target_overlap",
                    }
                )

            neg_pool_herbs = ordered_herbs[100:500]
            neg_pool_scores = ordered_scores[100:500]
            if len(neg_pool_herbs) == 0:
                continue

            avg_pos_degree = float(mech["herb_degree_arr"][pos_herbs].mean()) if len(pos_herbs) > 0 else 0.0
            neg_degree_delta = np.abs(mech["herb_degree_arr"][neg_pool_herbs] - avg_pos_degree)
            neg_order = np.lexsort((neg_pool_herbs, neg_pool_scores, neg_degree_delta))
            neg_herbs = neg_pool_herbs[neg_order[:20]]
            neg_scores = neg_pool_scores[neg_order[:20]]
            for h_idx, score in zip(neg_herbs, neg_scores):
                neg_records.append(
                    {
                        "MeSH": disease_id,
                        "CHP_ID": mech["inv_herb"][h_idx],
                        "mechanism_score": float(score),
                        "total_degree": float(mech["herb_degree_arr"][h_idx]),
                        "label": 0,
                        "label_source": "legacy_shared_target_overlap",
                    }
                )

        if block_end % 1024 == 0 or block_end == len(mech["diseases"]):
            print(f"   - Processed {block_end}/{len(mech['diseases'])} diseases")

    pos_df = pd.DataFrame(pos_records)
    neg_df = pd.DataFrame(neg_records)
    if pos_df.empty or neg_df.empty:
        raise ValueError("Legacy pseudo-label dataset is empty.")

    df = pd.concat([pos_df, neg_df], ignore_index=True)
    valid_meshes = pos_df.groupby("MeSH").size().index[:2000]
    df = df[df["MeSH"].isin(valid_meshes)].copy()
    df = df.drop_duplicates(subset=["MeSH", "CHP_ID", "label"]).copy()

    meshes = df["MeSH"].drop_duplicates().to_numpy()
    np.random.seed(args.seed)
    np.random.shuffle(meshes)
    n = len(meshes)
    train_m = set(meshes[: int(n * 0.7)])
    val_m = set(meshes[int(n * 0.7): int(n * 0.8)])
    df["split"] = df["MeSH"].apply(
        lambda x: "train" if x in train_m else ("val" if x in val_m else "test")
    )

    meta = {
        "dataset_mode": "pseudo_overlap_legacy",
        "seed": int(args.seed),
        "label_source": "shared-target overlap pseudo labels",
        "negative_strategy": {
            "type": "legacy_rank_window",
            "rank_window": [100, 500],
            "degree_balanced": True,
        },
        "split_strategy": "zero_shot_by_disease_mesh",
        "num_samples": int(len(df)),
        "num_positive": int((df["label"] == 1).sum()),
        "num_negative": int((df["label"] == 0).sum()),
        "num_diseases": int(df["MeSH"].nunique()),
        "num_herbs": int(df["CHP_ID"].nunique()),
    }
    disease_summary_df = (
        df.groupby(["MeSH", "label"]).size().unstack(fill_value=0).reset_index().rename(columns={0: "negative_count", 1: "positive_count"})
    )
    return df, meta, disease_summary_df


def main():
    parser = argparse.ArgumentParser(
        description="Build herb-disease datasets for mechanism-aware heterogeneous graph learning."
    )
    parser.add_argument(
        "--dataset-mode",
        choices=["external_cpm_strict", "pseudo_overlap_legacy"],
        default="external_cpm_strict",
        help="Strict external labels are recommended for no-leakage evaluation.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-pos-per-disease", type=int, default=5)
    parser.add_argument("--neg-per-positive", type=int, default=1)
    parser.add_argument("--neg-rank-start", type=int, default=100)
    parser.add_argument("--neg-rank-end", type=int, default=800)
    args = parser.parse_args()

    os.chdir(PROJECT_ROOT)
    out_dir = PROJECT_ROOT / "out" / "dataset"
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.dataset_mode == "external_cpm_strict":
        df, meta, disease_summary_df = build_strict_external_dataset(args)
    else:
        df, meta, disease_summary_df = build_legacy_overlap_dataset(args)

    out_path = out_dir / "hard_dataset.csv"
    meta_path = out_dir / "dataset_meta.json"
    summary_path = out_dir / "disease_sampling_summary.tsv"

    print("5/5: Saving dataset and metadata...")
    df.to_csv(out_path, index=False)
    disease_summary_df.to_csv(summary_path, sep="\t", index=False)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print("=== Dataset Summary ===")
    print(json.dumps(meta, ensure_ascii=False, indent=2))
    print(f"Saved dataset: {out_path}")
    print(f"Saved meta   : {meta_path}")
    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()
