### 1. `step1_calc_idf.py`
# 计算靶点 IDF 权重
# 创新点A核心：区分泛靶点与特异性靶点

import os
import numpy as np
import pandas as pd


D13 = "data/D13_InChIKey_EntrezID.tsv"
D9_D13 = "data/D9_CHP_InChIKey.tsv"
OUT_PATH = "out/filters/target_idf.csv"


def validate_columns(df: pd.DataFrame, required_cols: list[str], name: str) -> None:
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"{name} 缺少必要列: {missing}. 实际列为: {list(df.columns)}")


def main() -> None:
    print("Computing Target IDF...")

    if not os.path.exists(D9_D13):
        raise FileNotFoundError(f"找不到文件: {D9_D13}")
    if not os.path.exists(D13):
        raise FileNotFoundError(f"找不到文件: {D13}")

    d9 = pd.read_csv(D9_D13, sep='\t')
    d13 = pd.read_csv(D13, sep='\t')

    validate_columns(d9, ["CHP_ID", "InChIKey"], "D9_D13")
    validate_columns(d13, ["InChIKey", "EntrezID"], "D13")

    # 去掉关键字段缺失，避免 merge 后出现伪记录
    d9 = d9[["CHP_ID", "InChIKey"]].dropna().drop_duplicates()
    d13 = d13[["InChIKey", "EntrezID"]].dropna().drop_duplicates()

    # 构建 herb-target 对应表
    merged = d9.merge(d13, on="InChIKey", how="inner")
    ht = merged[["CHP_ID", "EntrezID"]].drop_duplicates()

    if ht.empty:
        raise ValueError("合并后 herb-target 为空，请检查 InChIKey 是否对齐。")

    n_herbs = ht["CHP_ID"].nunique()
    n_targets = ht["EntrezID"].nunique()

    # 计算每个 target 出现在多少种 herb 中 (document frequency)
    df = ht.groupby("EntrezID")["CHP_ID"].nunique().reset_index(name="df")
    df = df.rename(columns={"EntrezID": "target"})

    # 使用平滑 IDF，避免出现负值：
    # - df 越大，idf 越小
    # - df 越小，idf 越大
    # - 最小值 >= 1，便于后续阈值解释
    df["idf"] = np.log((n_herbs + 1) / (df["df"] + 1)) + 1

    # 便于后续审计和 sanity check
    df["target_coverage_ratio"] = df["df"] / n_herbs

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    df = df.sort_values(["idf", "df", "target"], ascending=[False, True, True]).reset_index(drop=True)
    df.to_csv(OUT_PATH, index=False)

    print(f"Herbs: {n_herbs}")
    print(f"Targets: {n_targets}")
    print(f"Herb-target pairs: {len(ht)}")
    print(f"Saved to: {OUT_PATH}")
    print(
        "IDF stats -> "
        f"max: {df['idf'].max():.4f}, "
        f"median: {df['idf'].median():.4f}, "
        f"min: {df['idf'].min():.4f}"
    )


if __name__ == "__main__":
    main()