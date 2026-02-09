import pandas as pd
import numpy as np

# データ読み込み
input_path = "data/expression/adams2017measuring_4420-fluorescein_exp_er.csv"
xdf = pd.read_csv(input_path)
allow_only_once = False
diff_filter = True
diff_threshold = 0.1
# seqカラムの作成
if "light" in xdf.columns:
    xdf["seq"] = xdf["heavy"] + "|" + xdf["light"]
else:
    xdf["seq"] = xdf["heavy"]

print(f"元のデータ:")
print(f"  総行数: {len(xdf)}")
print(f"  ユニークな配列数: {xdf['seq'].nunique()}")

# 各配列の測定回数と最大値・最小値の差を計算（情報表示用）
sizedf = xdf.groupby("seq").size()
diff_by_seq = xdf.groupby("seq")["fitness"].max() - xdf.groupby("seq")["fitness"].min()

print(f"\n測定回数の統計:")
print(f"  平均測定回数: {sizedf.mean():.2f}")
print(f"  最大測定回数: {sizedf.max()}")
print(f"  1回のみ測定: {(sizedf == 1).sum()} 配列")
print(f"  2回以上測定: {(sizedf >= 2).sum()} 配列")

print(f"\n最大値・最小値の差の統計（測定2回以上）:")
diff_multi = diff_by_seq[diff_by_seq > 0]
if len(diff_multi) > 0:
    print(f"  平均差: {diff_multi.mean():.4f}")
    print(f"  中央値差: {diff_multi.median():.4f}")
    print(f"  最大差: {diff_multi.max():.4f}")

# フィルタリング条件の設定
if allow_only_once:
    # 1回のみの測定も許可する場合
    if diff_filter:
        # 最大値・最小値の差フィルタも適用（2回以上の測定のみ）
        valid_seqs = sizedf[(sizedf == 1) | ((sizedf >= 2) & (diff_by_seq <= diff_threshold))].index
        print(f"\nフィルタリング条件:")
        print(f"  測定回数: 1回以上")
        print(f"  最大値・最小値の差: {diff_threshold}以下（2回以上測定の場合）")
    else:
        # 差フィルタなし
        valid_seqs = sizedf.index
        print(f"\nフィルタリング条件:")
        print(f"  測定回数: 1回以上")
        print(f"  差フィルタ: なし")
else:
    # 2回以上の測定のみ許可
    if diff_filter:
        valid_seqs = sizedf[(sizedf >= 2) & (diff_by_seq <= diff_threshold)].index
        print(f"\nフィルタリング条件:")
        print(f"  測定回数: 2回以上")
        print(f"  最大値・最小値の差: {diff_threshold}以下")
    else:
        valid_seqs = sizedf[sizedf >= 2].index
        print(f"\nフィルタリング条件:")
        print(f"  測定回数: 2回以上")
        print(f"  差フィルタ: なし")

print(f"  条件を満たす配列数: {len(valid_seqs)}")

# 条件を満たす配列のデータのみを抽出
filtered_xdf = xdf[xdf['seq'].isin(valid_seqs)].copy()

print(f"  フィルタリング後のデータ行数: {len(filtered_xdf)}")

# uniqueな配列ごとにfitnessの平均を計算
# グループ化するカラムを決定
groupby_cols = ['heavy']
if 'light' in xdf.columns:
    groupby_cols.append('light')
if 'format' in xdf.columns:
    groupby_cols.append('format')

# 各グループのfitness平均を計算
result_df = filtered_xdf.groupby(groupby_cols, as_index=False).agg({
    'fitness': 'mean'
})

# 元のCSVの最初のカラム（Expression [ER]）がある場合は保持
if 'Expression [ER]' in filtered_xdf.columns:
    # 最初のカラムも平均化（通常はfitnessと同じ値のはず）
    expr_mean = filtered_xdf.groupby(groupby_cols)['Expression [ER]'].mean().reset_index()
    result_df = expr_mean.merge(result_df, on=groupby_cols)
    # カラムの順序を元のCSVに合わせる
    result_df = result_df[['Expression [ER]'] + groupby_cols + ['fitness']]

print(f"\n平均化後のデータ:")
print(f"  総行数: {len(result_df)}")
print(f"  ユニークな配列数: {result_df['heavy'].nunique() if 'heavy' in result_df.columns else len(result_df)}")

# 結果を保存（元のファイルを上書き）
output_path = input_path.replace(".csv", f"_filtered.csv")
result_df.to_csv(output_path, index=False)

print(f"\n平均化されたデータを保存しました: {output_path}")

# 統計情報を表示
print(f"\n平均化後の統計:")
print(f"  fitness平均: {result_df['fitness'].mean():.4f}")
print(f"  fitness標準偏差: {result_df['fitness'].std():.4f}")
print(f"  fitness最小値: {result_df['fitness'].min():.4f}")
print(f"  fitness最大値: {result_df['fitness'].max():.4f}")

xxx = xdf.groupby("seq")["fitness"].max()-xdf.groupby("seq")["fitness"].min()
yyy = filtered_xdf.groupby("seq")["fitness"].max()-filtered_xdf.groupby("seq")["fitness"].min()
print(xdf["fitness"].max())
print(xdf["fitness"].min())
print(xxx.max())
print(yyy.max())
