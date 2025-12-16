import pandas as pd
import itertools
from math import factorial

class Solution:
    def _merge_columns(self, df: pd.DataFrame, col_merge: list):
        if not col_merge:
            return df.copy()
        df = df.copy()
        original_cols = list(df.columns)
        for group in col_merge:
            # keep original order of columns inside the group
            group = [col for col in original_cols if col in group]
            if not group:
                continue
            new_name = "__MERGED__" + "_".join(group)
            merged_col = df[group[0]].astype(str)
            for col in group[1:]:
                merged_col = merged_col + df[col].astype(str)
            df[new_name] = merged_col
            df.drop(columns=group, inplace=True)
        return df

    def _column_stats(self, series: pd.Series):
        s = series.astype(str)
        n = len(s)
        # average string length
        avg_len = s.str.len().mean()
        # probability two random rows share the same value
        cnts = s.value_counts()
        p_match = (cnts.pow(2).sum()) / (n * n)
        return avg_len, p_match

    def _best_permutation(self, cols, avg_lens, p_matches):
        m = len(cols)
        if factorial(m) <= 2_000_000:  # exhaustive search threshold
            best_perm = None
            best_score = -1.0
            total_len = sum(avg_lens[c] for c in cols)
            for perm in itertools.permutations(cols):
                cumulative_p = 1.0
                expected_prefix = 0.0
                for c in perm:
                    expected_prefix += cumulative_p * avg_lens[c]
                    cumulative_p *= p_matches[c]
                score = expected_prefix / total_len if total_len else 0.0
                if score > best_score:
                    best_score = score
                    best_perm = perm
            return list(best_perm)
        # Greedy fallback
        remaining = set(cols)
        order = []
        cumulative_p = 1.0
        while remaining:
            best_col = None
            best_increment = -1.0
            for c in remaining:
                increment = cumulative_p * avg_lens[c]
                if increment > best_increment:
                    best_increment = increment
                    best_col = c
            order.append(best_col)
            cumulative_p *= p_matches[best_col]
            remaining.remove(best_col)
        return order

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
        # Step 1: apply column merges
        df_merged = self._merge_columns(df, col_merge)

        # Step 2: compute statistics on a sample (or full data if small)
        sample_df = df_merged.iloc[:early_stop]
        avg_lens = {}
        p_matches = {}
        for col in sample_df.columns:
            avg_len, p_match = self._column_stats(sample_df[col])
            avg_lens[col] = avg_len
            p_matches[col] = p_match

        # Step 3: determine best column order
        cols = list(df_merged.columns)
        best_order = self._best_permutation(cols, avg_lens, p_matches)

        # Step 4: return DataFrame with reordered columns
        return df_merged[best_order]
