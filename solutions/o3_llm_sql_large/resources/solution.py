import pandas as pd

class Solution:
    def solve(
        self,
        df: pd.DataFrame,
        early_stop: int = 100000,
        row_stop: int = 4,
        col_stop: int = 2,
        col_merge: list = None,
        one_way_dep: list = None,
        distinct_value_threshold: float = 0.7,
        parallel: bool = True,
    ) -> pd.DataFrame:

        # 1. Apply column merges if specified
        if col_merge:
            merged_cols_to_drop = []
            for idx, group in enumerate(col_merge):
                valid_cols = [c for c in group if c in df.columns]
                if not valid_cols:
                    continue
                new_col_name = f"merged_{idx}"
                df[new_col_name] = df[valid_cols].astype(str).agg(''.join, axis=1)
                merged_cols_to_drop.extend(valid_cols)
            if merged_cols_to_drop:
                df = df.drop(columns=merged_cols_to_drop)

        n_rows = len(df)
        if n_rows == 0:
            return df  # empty dataframe, nothing to do

        # 2. Compute column statistics
        unique_ratio = {}
        adj_dup_ratio = {}
        avg_len = {}

        for col in df.columns:
            # Convert to string once
            col_str = df[col].astype(str)
            unique_ratio[col] = col_str.nunique(dropna=False) / n_rows if n_rows else 0.0
            adj_dup_ratio[col] = (col_str.shift() == col_str).mean()
            avg_len[col] = col_str.str.len().mean()

        # 3. Build sorting key
        def sort_key(c):
            return (
                1 if unique_ratio[c] > distinct_value_threshold else 0,  # push high-cardinality cols back
                unique_ratio[c],                                          # lower unique ratio first
                1 - adj_dup_ratio[c],                                     # higher adjacent dup ratio first
                -avg_len[c],                                              # longer average length first
            )

        sorted_cols = sorted(df.columns, key=sort_key)

        # 4. Return dataframe with reordered columns
        return df[sorted_cols]
