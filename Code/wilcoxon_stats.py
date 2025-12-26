import re
import pandas as pd
from scipy.stats import ranksums, wilcoxon
from itertools import combinations

def Average(lst):
    return sum(lst) / len(lst)

def a12(lst1, lst2, rev=True):
    """
    Compute Vargha–Delaney A12 and interpret magnitude.
    """
    # decide direction
    if Average(lst1) < Average(lst2):
        rev = False

    more = same = 0.0
    for x in lst1:
        for y in lst2:
            if x == y:
                same += 1
            elif rev and x > y:
                more += 1
            elif not rev and x < y:
                more += 1

    res = (more + 0.5 * same) / (len(lst1) * len(lst2))

    # interpret
    if res > 0.71:
        desc = "Large"
    elif res > 0.64:
        desc = "Medium"
    elif res > 0.56:
        desc = "Small"
    else:
        desc = "negligible"

    if not rev:
        res = 1 - res
        # flip interpretation thresholds for reversed
        if res < 0.29:
            desc = "Large"
        elif res < 0.36:
            desc = "Medium"
        elif res < 0.44:
            desc = "Small"
        else:
            desc = "negligible"

    return res, desc


def compute_pair_wilcoxon(long_df, pair_size=2, rating_col="Rating (num)"):
    """
    Perform Wilcoxon rank-sum tests on user pairs for each criterion.
    
    Parameters:
      long_df (pd.DataFrame):
        A 'long' DataFrame with columns including:
          - "User": user identifier string (e.g., "P1_Parham_Sheet1")
          - "UserNum": numeric user ID (e.g., 1, 2, 3, ...)
          - "Criterion": e.g. "Clarity", "Simplicity", ...
          - "Rating (num)": numeric rating
      pair_size (int):
        Number of users per group (default is 2).
      rating_col (str):
        Name of the column in long_df that holds numeric ratings.
        
    Returns:
      dict:
        A dictionary where each key is a criterion (e.g. "Clarity")
        and the value is a DataFrame of Wilcoxon results with columns:
            - "Pair" (list of user strings)
            - "WilcoxonStat" (Wilcoxon rank-sum statistic)
            - "p-value" (p-value from the test)
    """
    # 1. Get unique users in the order they appear in the DataFrame
    users = list(dict.fromkeys(long_df["User"]))

    # 2. Sort users by their numeric identifier 
    #    (assuming "P1_Parham_Sheet1" => numeric ID is 1, etc.)
    #    Or, if "UserNum" is already in the DataFrame, you can rely on that.
    #    Example: P1 => 1, P2 => 2, etc.
    def extract_numeric_id(user_str):
        # E.g., "P2_Dipeeka_Sheet1" -> 2
        # If you already store user_num in the DataFrame, you can skip this step.
        match = re.match(r"P(\d+)_", user_str)
        return int(match.group(1)) if match else 999999  # fallback
    users.sort(key=extract_numeric_id)

    # 3. Group users in pairs
    user_pairs = [users[i:i+pair_size] for i in range(0, len(users), pair_size)]

    # 4. For each criterion, compute Wilcoxon rank-sum for each pair
    criteria_list = long_df["Criterion"].unique()
    results_by_criterion = {}

    for criterion in criteria_list:
        # Subset the DataFrame for just this criterion
        criterion_data = long_df[long_df["Criterion"] == criterion]

        pair_results = []
        for pair in user_pairs:
            if len(pair) < 2:
                # If there's an odd user left at the end, skip
                continue

            # Extract the two user strings
            user1, user2 = pair
            
            # Subset data for user1 and user2
            user1_data = criterion_data[criterion_data["User"] == user1][rating_col].dropna()
            user2_data = criterion_data[criterion_data["User"] == user2][rating_col].dropna()

            # If either is empty, skip
            if user1_data.empty or user2_data.empty:
                pair_results.append({
                "Pair":        pair,
                "WilcoxonStat": None,
                "p-value":     None,
                "A12":         None,
                "A12_desc":    None
            })
                continue

            # Perform Wilcoxon rank-sum test
            stat, pval = ranksums(user1_data, user2_data)
            a12_val, a12_desc = a12(user1_data.tolist(), user2_data.tolist())

            pair_results.append({
                "Pair":        pair,
                "WilcoxonStat": stat,
                "p-value":     pval,
                "A12":         a12_val,
                "A12_desc":    a12_desc
            })

        results_by_criterion[criterion] = pd.DataFrame(pair_results)
    
    return results_by_criterion

def compute_model_wilcoxon(long_df, rating_col="Rating (num)", model_pair=None):
    """
    Compute Wilcoxon rank-sum tests per criterion between models.
    
    This function groups the data by the "Criterion" column and then, for each criterion,
    it performs Wilcoxon rank-sum tests comparing the rating distributions of different models.
    
    Parameters:
      long_df (pd.DataFrame):
          A long-format DataFrame (e.g. output from read_long_data) that must include the columns:
              "User", "UserNum", "Task", "Criterion", "Rating (num)", "Rating (label)", and "Model".
      rating_col (str):
          Name of the column containing the numeric ratings. Default is "Rating (num)".
      model_pair (tuple or list of two str, optional):
          If provided, only the two specified models are compared (e.g., ("Llama", "Claude")).
          Otherwise, all pairwise comparisons among models within each criterion are performed.
    
    Returns:
      dict:
          A dictionary where each key is a Criterion (e.g., "Clarity", "Simplicity", etc.)
          and the corresponding value is a DataFrame that contains the pairwise Wilcoxon test results.
          Each DataFrame will have the columns:
              - "ModelPair"  : a string representation of the model pair (e.g., "Llama vs Claude")
              - "WilcoxonStat": the Wilcoxon rank-sum statistic (None if test was not performed)
              - "p-value"    : the p-value from the test (None if test was not performed)
              - "n1"         : number of samples in the first model group
              - "n2"         : number of samples in the second model group
    """
    criteria_list = long_df["Criterion"].unique()
    results_by_criterion = {}

    for criterion in criteria_list:
        crit_data = long_df[long_df["Criterion"] == criterion]
        models_present = crit_data["Model"].unique()

        # Decide which model pairs to compute:
        if model_pair is not None:
            # Validate that both models exist for this criterion.
            if model_pair[0] in models_present and model_pair[1] in models_present:
                pair_list = [model_pair]
            else:
                # If one of the requested models is missing, record a warning entry.
                pair_list = []
        else:
            # If no specific pair is requested, compute all unique pairwise comparisons
            # from the models present for this criterion.
            pair_list = list(combinations(sorted(models_present), 2))
        
        pair_results = []
        for pair in pair_list:
            model_a, model_b = pair

            group_a = crit_data[crit_data["Model"] == model_a][rating_col].dropna()
            group_b = crit_data[crit_data["Model"] == model_b][rating_col].dropna()

            n_a = group_a.shape[0]
            n_b = group_b.shape[0]

            # If one of the groups is empty, store None for the statistics.
            if n_a == 0 or n_b == 0:
                stat = pval = a12_val = a12_desc = None
            else:
                stat, pval = ranksums(group_a, group_b)
                a12_val, a12_desc = a12(group_a.tolist(), group_b.tolist())

            pair_results.append({
                "ModelPair": f"{model_a} vs {model_b}",
                "WilcoxonStat": stat,
                "p-value": pval,
                "n1": n_a,
                "n2": n_b,
                "A12":          a12_val,
                "A12_desc":     a12_desc
            })

        # Store the DataFrame of results for the criterion
        results_by_criterion[criterion] = pd.DataFrame(pair_results)
    
    return results_by_criterion

def compute_model_signed_wilcoxon(long_df, rating_col="Rating (num)", model_pair=None, min_pairs=1):
    """
    Compute paired Wilcoxon signed-rank tests per criterion between two models.

    For each criterion, this function finds users who have ratings for both models
    and performs the Wilcoxon signed-rank test on the paired ratings. If a user
    has multiple ratings for the same (criterion, model), their ratings are
    averaged so that each user contributes a single paired value per model.

    Parameters:
      long_df (pd.DataFrame): long-format DataFrame containing at least the columns
          "User", "Criterion", "Model", and the numeric rating column.
      rating_col (str): name of the column containing numeric ratings.
      model_pair (tuple of str, optional): pair of model names to compare. If None,
          the function will compute signed-rank tests for all pairwise combinations
          of models present for each criterion.
      min_pairs (int): minimum number of paired users required to run the test.

    Returns:
      dict: mapping from Criterion -> DataFrame with columns:
            - "ModelPair"
            - "WilcoxonStat"
            - "p-value"
            - "n_pairs"
    """
    criteria_list = long_df["Criterion"].unique()
    results_by_criterion = {}

    for criterion in criteria_list:
        crit_data = long_df[long_df["Criterion"] == criterion]
        models_present = sorted(crit_data["Model"].dropna().unique())

        if model_pair is not None:
            pair_list = [model_pair] if (model_pair[0] in models_present and model_pair[1] in models_present) else []
        else:
            # all unique pairs
            from itertools import combinations
            pair_list = list(combinations(models_present, 2))

        pair_results = []
        for pair in pair_list:
            model_a, model_b = pair

            # subset data for each model and compute per-user average rating
            a_df = crit_data[crit_data["Model"] == model_a]
            b_df = crit_data[crit_data["Model"] == model_b]

            # group by user and average rating to get one value per user per model
            a_by_user = a_df.groupby("User")[rating_col].mean()
            b_by_user = b_df.groupby("User")[rating_col].mean()

            # find users present in both
            common_users = a_by_user.index.intersection(b_by_user.index)

            n_pairs = len(common_users)

            if n_pairs < min_pairs:
                stat = pval = None
                n_nonzero = 0
                a12_val = a12_desc = None
            else:
                a_vals = a_by_user.loc[common_users].values
                b_vals = b_by_user.loc[common_users].values

                # Preprocess paired arrays: remove pairs with NaNs and optionally
                # zero differences so the signed-rank test is well-defined.
                import numpy as _np
                import warnings as _warnings

                # ensure numpy arrays
                a_vals = _np.asarray(a_vals)
                b_vals = _np.asarray(b_vals)

                # remove any pairs with NaN
                valid_mask = _np.isfinite(a_vals) & _np.isfinite(b_vals)
                a_clean = a_vals[valid_mask]
                b_clean = b_vals[valid_mask]

                # differences
                diffs = a_clean - b_clean

                # remove zero differences (these provide no information for signed test)
                nonzero_mask = diffs != 0
                diffs_filtered = diffs[nonzero_mask]
                a_filtered = a_clean[nonzero_mask]
                b_filtered = b_clean[nonzero_mask]

                # If after filtering we have too few observations, the test is not applicable.
                n_nonzero = diffs_filtered.size
                if n_nonzero < 1:
                    stat = pval = None
                    a12_val = a12_desc = None
                else:
                    try:
                        with _warnings.catch_warnings():
                            _warnings.simplefilter("ignore", category=RuntimeWarning)
                            # Pass the filtered differences directly to wilcoxon
                            stat, pval = wilcoxon(diffs_filtered, zero_method="wilcox")
                        # If result is NaN, treat as not-applicable
                        if not _np.isfinite(stat) or not _np.isfinite(pval):
                            stat = pval = None
                    except Exception:
                        stat = pval = None

                    # compute Vargha-Delaney A12 effect size on the filtered paired values
                    try:
                        a12_val, a12_desc = a12(a_filtered.tolist(), b_filtered.tolist())
                    except Exception:
                        a12_val = a12_desc = None

            pair_results.append({
                "ModelPair": f"{model_a} vs {model_b}",
                "WilcoxonStat": stat,
                "p-value": pval,
                "n_pairs": n_pairs,
                "n_nonzero": int(n_nonzero) if 'n_nonzero' in locals() else 0,
                "A12": a12_val if 'a12_val' in locals() else None,
                "A12_desc": a12_desc if 'a12_desc' in locals() else None
            })

        results_by_criterion[criterion] = pd.DataFrame(pair_results)

    return results_by_criterion

def compute_pair_signed_wilcoxon(long_df, pair_size=2, rating_col="Rating (num)"):
    """
    Perform paired Wilcoxon signed-rank tests for user pairs on each criterion.

    This function groups users into pairs (like `compute_pair_wilcoxon`) and for
    each pair and each criterion finds tasks that both users rated, then runs
    a Wilcoxon signed-rank test on their paired ratings (per-task).

    Parameters:
      long_df (pd.DataFrame): long-format DataFrame containing at least the columns
          "User", "Task", "Criterion", and the numeric rating column.
      pair_size (int): number of users per group when forming pairs (default 2).
      rating_col (str): name of the numeric rating column.

    Returns:
      dict: mapping Criterion -> DataFrame with columns: "Pair", "WilcoxonStat",
            "p-value", "n_pairs" (number of paired observations / tasks).
    """
    # 1. Get unique users in the order they appear
    users = list(dict.fromkeys(long_df["User"]))

    # Try to extract numeric id for stable ordering
    def extract_numeric_id(user_str):
        import re
        match = re.match(r"P(\d+)_", user_str)
        return int(match.group(1)) if match else 999999

    users.sort(key=extract_numeric_id)

    # 2. Group users into pairs of size `pair_size`
    user_pairs = [users[i:i+pair_size] for i in range(0, len(users), pair_size)]

    criteria_list = long_df["Criterion"].unique()
    results_by_criterion = {}

    for criterion in criteria_list:
        crit_data = long_df[long_df["Criterion"] == criterion]
        pair_results = []

        # create a pivot of Task x User for this criterion (averaging if multiple entries)
        pivot = crit_data.pivot_table(index="Task", columns="User", values=rating_col, aggfunc="mean")

        for pair in user_pairs:
            if len(pair) < 2:
                continue

            user1, user2 = pair

            if user1 not in pivot.columns or user2 not in pivot.columns:
                pair_results.append({
                    "Pair": pair,
                    "WilcoxonStat": None,
                    "p-value": None,
                    "n_pairs": 0,
                    "n_nonzero": 0,
                    "A12": None,
                    "A12_desc": None
                })
                continue

            paired = pivot[[user1, user2]].dropna()
            n_pairs = paired.shape[0]

            if n_pairs == 0:
                stat = pval = None
                n_nonzero = 0
                a12_val = a12_desc = None
            else:
                import numpy as _np
                import warnings as _warnings

                a_vals = paired[user1].values
                b_vals = paired[user2].values

                a_vals = _np.asarray(a_vals)
                b_vals = _np.asarray(b_vals)

                valid_mask = _np.isfinite(a_vals) & _np.isfinite(b_vals)
                a_clean = a_vals[valid_mask]
                b_clean = b_vals[valid_mask]

                diffs = a_clean - b_clean
                nonzero_mask = diffs != 0
                diffs_filtered = diffs[nonzero_mask]
                a_filtered = a_clean[nonzero_mask]
                b_filtered = b_clean[nonzero_mask]

                n_nonzero = diffs_filtered.size
                if n_nonzero < 1:
                    stat = pval = None
                    a12_val = a12_desc = None
                else:
                    try:
                        with _warnings.catch_warnings():
                            _warnings.simplefilter("ignore", category=RuntimeWarning)
                            # Pass the filtered differences directly to wilcoxon
                            stat, pval = wilcoxon(diffs_filtered, zero_method="wilcox")
                        if not _np.isfinite(stat) or not _np.isfinite(pval):
                            stat = pval = None
                    except Exception:
                        stat = pval = None

                    try:
                        a12_val, a12_desc = a12(a_filtered.tolist(), b_filtered.tolist())
                    except Exception:
                        a12_val = a12_desc = None

            pair_results.append({
                "Pair": pair,
                "WilcoxonStat": stat,
                "p-value": pval,
                "n_pairs": n_pairs,
                "n_nonzero": int(n_nonzero) if 'n_nonzero' in locals() else 0,
                "A12": a12_val if 'a12_val' in locals() else None,
                "A12_desc": a12_desc if 'a12_desc' in locals() else None
            })

        results_by_criterion[criterion] = pd.DataFrame(pair_results)

    return results_by_criterion


