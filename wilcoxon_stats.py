import re
import pandas as pd
from scipy.stats import ranksums
from itertools import combinations

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
                    "Pair": pair,
                    "WilcoxonStat": None,
                    "p-value": None
                })
                continue

            # Perform Wilcoxon rank-sum test
            stat, pval = ranksums(user1_data, user2_data)
            
            pair_results.append({
                "Pair": pair,
                "WilcoxonStat": stat,
                "p-value": pval
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
                stat = None
                pval = None
            else:
                stat, pval = ranksums(group_a, group_b)

            pair_results.append({
                "ModelPair": f"{model_a} vs {model_b}",
                "WilcoxonStat": stat,
                "p-value": pval,
                "n1": n_a,
                "n2": n_b
            })

        # Store the DataFrame of results for the criterion
        results_by_criterion[criterion] = pd.DataFrame(pair_results)
    
    return results_by_criterion

