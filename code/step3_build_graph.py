### 3. `step3_build_graph.py`
import json
import os
import random
import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MultiLabelBinarizer
from torch_geometric.data import HeteroData


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def require_columns(df: pd.DataFrame, df_name: str, required_cols):
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"{df_name} 缺少必要列: {missing}")


def build_edge_index_from_df(df: pd.DataFrame, src_col: str, dst_col: str, src_map: dict, dst_map: dict):
    tmp = df[[src_col, dst_col]].copy()
    tmp = tmp.dropna()
    tmp = tmp[tmp[src_col].isin(src_map) & tmp[dst_col].isin(dst_map)]
    tmp = tmp.drop_duplicates()

    src = tmp[src_col].map(src_map).astype(int).to_numpy()
    dst = tmp[dst_col].map(dst_map).astype(int).to_numpy()

    if len(tmp) == 0:
        return torch.empty((2, 0), dtype=torch.long), tmp

    edge_index = torch.tensor(np.vstack([src, dst]), dtype=torch.long)
    return edge_index, tmp


def main():
    parser = argparse.ArgumentParser(description="Build MeHGT heterogeneous graph.")
    parser.add_argument("--use-idf", type=int, choices=[0, 1], default=1)
    parser.add_argument("--use-filtering", type=int, choices=[0, 1], default=1)
    parser.add_argument("--dataset-path", type=str, default="out/dataset/hard_dataset.csv")
    parser.add_argument("--dataset-meta-path", type=str, default="out/dataset/dataset_meta.json")
    parser.add_argument("--output-dir", type=str, default="out/graph")
    args = parser.parse_args()
    use_idf = bool(args.use_idf)
    use_filtering = bool(args.use_filtering)
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    dataset_path = Path(args.dataset_path)
    if not dataset_path.is_absolute():
        dataset_path = PROJECT_ROOT / dataset_path
    dataset_meta_path = Path(args.dataset_meta_path)
    if not dataset_meta_path.is_absolute():
        dataset_meta_path = PROJECT_ROOT / dataset_meta_path
    output_dir.mkdir(parents=True, exist_ok=True)
    print("Step 3: Building Graph (Revised)...")
    set_seed(42)
    os.chdir(PROJECT_ROOT)

    # ========== 1. 加载数据 ==========
    print("1. Loading data...")
    d7 = pd.read_csv("data/D7_CHP_Medicinal_properties.tsv", sep="\t")
    d9 = pd.read_csv("data/D9_CHP_InChIKey.tsv", sep="\t")
    d13 = pd.read_csv("data/D13_InChIKey_EntrezID.tsv", sep="\t")
    d23 = pd.read_csv("data/D23_MeSH_targets.tsv", sep="\t")
    idf = pd.read_csv("out/filters/target_idf.csv")
    dataset = pd.read_csv(dataset_path)
    dataset_meta = None
    if dataset_meta_path.exists():
        with open(dataset_meta_path, "r", encoding="utf-8") as f:
            dataset_meta = json.load(f)
        print(
            "[Meta] dataset_mode="
            f"{dataset_meta.get('dataset_mode')} | label_source={dataset_meta.get('label_source')}"
        )

    require_columns(d7, "D7", ["CHP_ID", "Medicinal_properties"])
    require_columns(d9, "D9", ["CHP_ID", "InChIKey"])
    require_columns(d13, "D13", ["InChIKey", "EntrezID"])
    require_columns(d23, "D23", ["MeSH", "EntrezID"])
    require_columns(idf, "target_idf", ["target", "idf"])
    require_columns(dataset, "hard_dataset", ["MeSH", "CHP_ID", "label", "split"])

    # 统一主键类型
    d7["CHP_ID"] = d7["CHP_ID"].astype(str)
    d9["CHP_ID"] = d9["CHP_ID"].astype(str)
    d9["InChIKey"] = d9["InChIKey"].astype(str)
    d13["InChIKey"] = d13["InChIKey"].astype(str)
    d13["EntrezID"] = d13["EntrezID"].astype(str)
    d23["MeSH"] = d23["MeSH"].astype(str)
    d23["EntrezID"] = d23["EntrezID"].astype(str)
    idf["target"] = idf["target"].astype(str)
    dataset["MeSH"] = dataset["MeSH"].astype(str)
    dataset["CHP_ID"] = dataset["CHP_ID"].astype(str)

    # 清理空值和重复
    d9 = d9.dropna(subset=["CHP_ID", "InChIKey"]).drop_duplicates(subset=["CHP_ID", "InChIKey"])
    d13 = d13.dropna(subset=["InChIKey", "EntrezID"]).drop_duplicates(subset=["InChIKey", "EntrezID"])
    d23 = d23.dropna(subset=["MeSH", "EntrezID"]).drop_duplicates(subset=["MeSH", "EntrezID"])
    idf = idf.dropna(subset=["target", "idf"]).drop_duplicates(subset=["target"])
    dataset = dataset.dropna(subset=["MeSH", "CHP_ID", "label", "split"]).drop_duplicates()

    valid_mesh_set = set(dataset["MeSH"].unique())

    # ========== 2. Herb 特征工程 ==========
    print("2. Engineering herb features...")

    all_herbs = sorted(d9["CHP_ID"].unique())
    herb_map = {h: i for i, h in enumerate(all_herbs)}
    num_herbs = len(herb_map)

    # 2.1 TCM 属性特征
    d7 = d7.copy()
    d7["Medicinal_properties"] = d7["Medicinal_properties"].fillna("").astype(str)
    d7["props"] = d7["Medicinal_properties"].apply(
        lambda x: [p.strip() for p in x.split(",") if p.strip()]
    )

    d7_grouped = d7.groupby("CHP_ID")["props"].sum().reset_index()
    d7_grouped["props"] = d7_grouped["props"].apply(lambda xs: sorted(list(set(xs))))
    d7_grouped = d7_grouped.set_index("CHP_ID")

    herb_props_list = []
    for herb_id in all_herbs:
        if herb_id in d7_grouped.index:
            herb_props_list.append(d7_grouped.loc[herb_id, "props"])
        else:
            herb_props_list.append([])

    mlb = MultiLabelBinarizer()
    tcm_features = mlb.fit_transform(herb_props_list).astype(np.float32)

    if tcm_features.shape[1] == 0:
        tcm_features = np.zeros((num_herbs, 1), dtype=np.float32)

    print(f"   - TCM feature dim: {tcm_features.shape[1]}")

    # 2.2 Herb 度数特征（替换掉原先随机 identity）
    herb_degree_df = (
        d9.merge(d13, on="InChIKey", how="inner")[["CHP_ID", "EntrezID"]]
        .drop_duplicates()
        .groupby("CHP_ID")
        .size()
        .reset_index(name="target_degree")
    )
    herb_degree_dict = dict(zip(herb_degree_df["CHP_ID"], herb_degree_df["target_degree"]))

    herb_degree_feature = np.array(
        [[np.log1p(herb_degree_dict.get(h, 0))] for h in all_herbs],
        dtype=np.float32
    )

    # 拼接 herb 特征
    final_herb_features = np.hstack([tcm_features, herb_degree_feature]).astype(np.float32)
    x_herb = torch.tensor(final_herb_features, dtype=torch.float)

    print(f"   - Herb final dim: {x_herb.shape[1]}")

    # ========== 3. 构建 Herb-Target 图 ==========
    print("3. Constructing herb-target graph...")

    ht_raw = d9.merge(d13, on="InChIKey", how="inner")[["CHP_ID", "EntrezID"]].drop_duplicates()
    if use_idf:
        ht_raw = ht_raw.merge(idf, left_on="EntrezID", right_on="target", how="inner")
    else:
        # No-IDF ablation: retain all herb-target edges and use a neutral weight.
        ht_raw["target"] = ht_raw["EntrezID"]
        ht_raw["idf"] = 1.0

    if ht_raw.empty:
        raise ValueError("ht_raw 为空，无法构图。请检查 D9 / D13 / target_idf 是否正确。")

    initial_edges = len(ht_raw)
    print(f"   - Raw herb-target edges: {initial_edges}")

    if use_filtering:
        global_idf_threshold = ht_raw["idf"].quantile(0.25)
        ht_clean = ht_raw[ht_raw["idf"] >= global_idf_threshold].copy()
        print(f"   - Global IDF threshold (25% quantile): {global_idf_threshold:.4f}")
        print(f"   - After global filter: {len(ht_clean)}")
        K = 50
        ht_clean = ht_clean.sort_values(["CHP_ID", "idf", "EntrezID"], ascending=[True, False, True])
        ht_clean = ht_clean.groupby("CHP_ID", group_keys=False).head(K).copy()
    else:
        ht_clean = ht_raw.copy()
        print("   - Global IDF/top-K filtering: disabled")

    print(f"   - After Top-{K} per herb: {len(ht_clean)}")
    print(f"   - Reduction rate: {(1 - len(ht_clean) / initial_edges) * 100:.1f}%")

    # ========== 4. Target / Disease 节点集合 ==========
    print("4. Building node maps...")

    d23_filtered = d23[d23["MeSH"].isin(valid_mesh_set)].copy()

    if d23_filtered.empty:
        raise ValueError("d23_filtered 为空，说明 dataset 里的疾病无法在 D23 中找到靶点关系。")

    relevant_targets = set(ht_clean["EntrezID"]) | set(d23_filtered["EntrezID"])
    all_targets = sorted(relevant_targets)
    target_map = {t: i for i, t in enumerate(all_targets)}

    all_diseases = sorted(valid_mesh_set)
    disease_map = {m: i for i, m in enumerate(all_diseases)}

    print(f"   - #Herbs   : {len(herb_map)}")
    print(f"   - #Targets : {len(target_map)}")
    print(f"   - #Diseases: {len(disease_map)}")

    # ========== 5. 组装 HeteroData ==========
    print("5. Assembling HeteroData...")
    data = HeteroData()

    # 5.1 Herb nodes
    data["herb"].num_nodes = len(herb_map)
    data["herb"].x = x_herb

    # 5.2 Target nodes: 仅用 IDF 作为特征
    idf_dict = dict(zip(idf["target"], idf["idf"]))
    target_x = np.array(
        [[idf_dict.get(t, 1.0) if use_idf else 1.0] for t in all_targets],
        dtype=np.float32
    )
    data["target"].x = torch.tensor(target_x, dtype=torch.float)

    # 5.3 Disease nodes: 先放 1 维零向量，step4 再聚合 target 特征
    data["disease"].x = torch.zeros((len(disease_map), 1), dtype=torch.float)

    # ========== 6. 构建图边 ==========
    print("6. Building graph edges...")

    # Herb -> Target
    edge_index_ht, ht_clean_used = build_edge_index_from_df(
        ht_clean, "CHP_ID", "EntrezID", herb_map, target_map
    )
    data["herb", "interacts", "target"].edge_index = edge_index_ht
    data["target", "rev_interacts", "herb"].edge_index = edge_index_ht.flip(0)

    # Target -> Disease
    edge_index_td, d23_used = build_edge_index_from_df(
        d23_filtered, "EntrezID", "MeSH", target_map, disease_map
    )
    data["target", "rev_associates", "disease"].edge_index = edge_index_td
    data["disease", "associates", "target"].edge_index = edge_index_td.flip(0)

    print(f"   - Herb-Target edges : {edge_index_ht.shape[1]}")
    print(f"   - Target-Disease edges: {edge_index_td.shape[1]}")

    # ========== 7. 监督标签边 ==========
    print("7. Building supervision edges...")

    ds_valid = dataset[
        dataset["CHP_ID"].isin(herb_map) &
        dataset["MeSH"].isin(disease_map)
    ].copy()

    if ds_valid.empty:
        raise ValueError("监督数据 ds_valid 为空，请检查 step2 输出是否与当前图节点一致。")

    ds_valid["herb_idx"] = ds_valid["CHP_ID"].map(herb_map)
    ds_valid["disease_idx"] = ds_valid["MeSH"].map(disease_map)
    ds_valid = ds_valid.dropna(subset=["herb_idx", "disease_idx"]).copy()

    ds_valid["herb_idx"] = ds_valid["herb_idx"].astype(int)
    ds_valid["disease_idx"] = ds_valid["disease_idx"].astype(int)
    ds_valid["label"] = ds_valid["label"].astype(float)
    ds_valid["split"] = ds_valid["split"].astype(str)

    edge_label_index = torch.tensor(
        ds_valid[["herb_idx", "disease_idx"]].to_numpy().T,
        dtype=torch.long
    )
    edge_label = torch.tensor(ds_valid["label"].to_numpy(), dtype=torch.float)

    data["herb", "treats", "disease"].edge_index = edge_label_index
    data["herb", "treats", "disease"].edge_label = edge_label

    for split in ["train", "val", "test"]:
        mask = torch.tensor((ds_valid["split"].values == split), dtype=torch.bool)
        setattr(data["herb", "treats", "disease"], f"{split}_mask", mask)

    print(f"   - Supervision edges: {edge_label_index.shape[1]}")
    for split in ["train", "val", "test"]:
        mask = getattr(data["herb", "treats", "disease"], f"{split}_mask")
        print(f"     {split}: {int(mask.sum())}")

    # ========== 8. 保存 ==========
    torch.save(data, output_dir / "hetero_data.pt")
    joblib.dump(
        {
            "herb": herb_map,
            "target": target_map,
            "disease": disease_map,
        },
        output_dir / "mappings.pkl"
    )

    # 额外保存清洗后的边，便于 step5 / step6 审计
    ht_clean_used.to_csv(output_dir / "herb_target_cleaned.csv", index=False)
    d23_used.to_csv(output_dir / "target_disease_used.csv", index=False)
    if dataset_meta is not None:
        graph_meta = {
            "dataset_mode": dataset_meta.get("dataset_mode"),
            "label_source": dataset_meta.get("label_source"),
            "split_strategy": dataset_meta.get("split_strategy"),
            "split_seed": dataset_meta.get("split_seed", dataset_meta.get("seed")),
            "dataset_path": str(dataset_path),
            "num_supervision_edges": int(edge_label_index.shape[1]),
            "num_herb_target_edges": int(edge_index_ht.shape[1]),
            "num_target_disease_edges": int(edge_index_td.shape[1]),
            "use_idf": use_idf,
            "use_filtering": use_filtering,
            "raw_herb_target_edges": int(initial_edges),
            "edge_retention": float(edge_index_ht.shape[1] / max(initial_edges, 1)),
        }
        with open(output_dir / "graph_meta.json", "w", encoding="utf-8") as f:
            json.dump(graph_meta, f, ensure_ascii=False, indent=2)

    print("\n=== Graph Statistics ===")
    print(data)
    print("\n[OK] Saved to:")
    print(f"   - {output_dir / 'hetero_data.pt'}")
    print(f"   - {output_dir / 'mappings.pkl'}")
    print(f"   - {output_dir / 'herb_target_cleaned.csv'}")
    print(f"   - {output_dir / 'target_disease_used.csv'}")


if __name__ == "__main__":
    main()
