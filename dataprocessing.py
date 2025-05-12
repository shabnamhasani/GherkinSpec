import os
import pandas as pd
import numpy as np
import re
import tiktoken
# GPT tokenizer setup (default: gpt-3.5-turbo; change if needed)
ENCODING = tiktoken.encoding_for_model("gpt-3.5-turbo")

# Rating conversion dictionary
rating_map = {
    "All Relevant": 5,
    "Mostly Relevant": 4,
    "Somewhat Relevant": 3,
    "Mostly Irrelevant": 2,
    "All Irrelevant": 1,

    "Fully Complete": 5,
    "Mostly Complete": 4,
    "Partially Complete": 3,
    "Mostly Incomplete": 2,
    "Fully Incomplete": 1,

    "Completely Clear": 5,
    "Mostly Clear": 4,
    "Somewhat Clear": 3,
    "Mostly Unclear": 2,
    "Completely Unclear": 1,

    "Completely Singular": 5,
    "Mostly Singular": 4,
    "Somewhat Singular": 3,
    "Mostly Mixed": 2,
    "Completely Mixed": 1,

    "Completely Helpful": 5,
    "Mostly Helpful": 4,
    "Somewhat Helpful": 3,
    "Mostly Unhelpful": 2,
    "Completely Unhelpful": 1,
}
# Helper function to convert model id to a name
def model_id_to_name(m_id):
    return "Claude" if m_id == 1 else "Llama" if m_id == 2 else None

# Mapping dictionary: keys are user numbers, values are dictionaries that map task number (1-12) to model id.
user_task_model_map = {
    1: {i: 1 for i in range(1, 13)}, 
    2: {i: 1 for i in range(1, 13)},  
    3: {**{i: 2 for i in range(1, 7)}, **{i: 1 for i in range(7, 13)}},  
    4: {**{i: 2 for i in range(1, 7)}, **{i: 1 for i in range(7, 13)}}, 
    5: {**{i: 1 for i in range(1, 7)}, **{i: 2 for i in range(7, 13)}},  
    6: {**{i: 1 for i in range(1, 7)}, **{i: 2 for i in range(7, 13)}},  
    7: {**{i: 2 for i in range(1, 7)}, **{i: 1 for i in range(7, 13)}}, 
    8: {**{i: 2 for i in range(1, 7)}, **{i: 1 for i in range(7, 13)}},  
    9: {i: 2 for i in range(1, 13)},
    10:{i: 2 for i in range(1, 13)}
}

def read_excel_data(input_dir, rating_map):
    """
    Reads Excel files from the input directory and extracts user data.
    Returns:
      - all_user_data: dict mapping each user (file_sheet) to their ratings per criterion.
      - task_ids: list of tasks (assumed common to all files).
      - criteria_names: list of criteria (assumed common to all files).
    """
    all_user_data = {}
    task_ids = []
    criteria_names = []
    
    for file in os.listdir(input_dir):
        if file.endswith(".xlsx") or file.endswith(".xls"):
            file_path = os.path.join(input_dir, file)
            xlsx = pd.ExcelFile(file_path)
            for sheet_name in xlsx.sheet_names:
                # print(f"Reading file: {file_path}, sheet: {sheet_name}")
                df = pd.read_excel(xlsx, sheet_name=sheet_name, header=None)
                task_rows = range(4, 16)
                criteria_columns = range(3, 8)
                
                if not task_ids:
                    task_ids = df.iloc[task_rows, 0].tolist()
                    # print(f"Extracted task IDs: {task_ids}")
                if not criteria_names:
                    criteria_names = df.iloc[0, criteria_columns].tolist()
                    # print(f"Extracted criteria names: {criteria_names}")
                    
                user_id = f"{os.path.splitext(file)[0]}_{sheet_name}"
                user_dict = {}
                for col_idx, criterion in zip(criteria_columns, criteria_names):
                    ratings = [rating_map.get(df.iloc[row_idx, col_idx], None) for row_idx in task_rows]
                    user_dict[criterion] = ratings
                    # print(f"Ratings for {criterion}: {ratings}")
                all_user_data[user_id] = user_dict
                # print(f"Processed {user_id} from {file_path}")
                # print(f"User data: {user_dict}")
                # print("-" * 40)
    # print(f"Total users processed: {len(all_user_data)}")
    return all_user_data, task_ids, criteria_names

def create_summary_dfs(all_user_data, task_ids, criteria_names):
    """
    Creates a dictionary mapping each criterion to its summary DataFrame.
    Each DataFrame uses tasks as rows and users as columns, adds an 'Average' column (across users)
    and a 'User Average' row (across tasks).
    """
    summary_dfs = {}
    for criterion in criteria_names:
        df = pd.DataFrame({
            user: ratings[criterion]
            for user, ratings in all_user_data.items()
        }, index=task_ids)
        # Add a column for the average rating per task across users
        df["Average"] = df.mean(axis=1)
        
        # Add a row for the average rating per user across tasks
        user_averages = df.mean(axis=0)
        df.loc["User Average"] = user_averages
        df.index.name = "Task"
        summary_dfs[criterion] = df
    return summary_dfs

def write_summary_excel(output_path, summary_dfs):
    """
    Writes each criterion's summary DataFrame to a separate sheet in the output Excel file.
    """
    with pd.ExcelWriter(output_path, engine="xlsxwriter") as writer:
        for criterion, df in summary_dfs.items():
            # print(f"Writing DataFrame for {criterion}:\n{df}")
            df.to_excel(writer, sheet_name=criterion)
    # print(f"✅ Saved summary to: {output_path}")

def read_long_data(input_dir, rating_map):
    all_data_long = []
    for file in os.listdir(input_dir):
        if file.endswith(".xlsx") or file.endswith(".xls"):
            file_path = os.path.join(input_dir, file)
            xlsx = pd.ExcelFile(file_path)
            for sheet_name in xlsx.sheet_names:
                df = pd.read_excel(xlsx, sheet_name=sheet_name, header=None)
                # Assuming tasks are in rows 4 to 16 (i.e., 12 rows for tasks)
                task_rows = range(4, 16)
                criteria_columns = range(3, 8)  # for your criteria columns
                # Extract criteria names from the sheet (first row in those columns)
                criteria_names = df.iloc[0, criteria_columns].tolist()
                
                # Construct user_id and extract numeric part (assumes file name like "User1.xlsx")
                user_id = f"{os.path.splitext(file)[0]}_{sheet_name}"
                match = re.search(r'(\d+)', os.path.splitext(file)[0])
                user_num = int(match.group(1)) if match else None
                
                for col_idx, criterion in zip(criteria_columns, criteria_names):
                    for row_idx in task_rows:
                        raw_val = df.iloc[row_idx, col_idx]
                        rating_num = rating_map.get(raw_val, None)
                        if rating_num is not None:
                            # Extract task number from the task cell (assumes value like "Task1", "Task2", …)
                            task_cell = str(df.iloc[row_idx, 0])
                            match_task = re.search(r'(\d+)', task_cell)
                            if match_task:
                                task_num = int(match_task.group(1))
                            else:
                                continue  # or set a default

                            # Determine the model using the mapping dictionary
                            m_id = user_task_model_map.get(user_num, {}).get(task_num, None)
                            model_name = model_id_to_name(m_id)
                            
                            all_data_long.append({
                                "User": user_id,
                                "UserNum": user_num,
                                "Task": task_num,  # now an integer 1..12
                                "Criterion": criterion,
                                "Rating (num)": rating_num,
                                "Rating (label)": raw_val,
                                "Model": model_name,
                            })
    return pd.DataFrame(all_data_long)

import os, re
import pandas as pd

def gpt_token_count(text):
    return len(ENCODING.encode(text))

def compute_unique_gherkin_counts(input_dir, user_task_model_map=user_task_model_map):
    """
    Reads every P<user>_*.xlsx, applies your user→(task→model) map,
    counts Gherkin tokens for each (Task,Model), then drops duplicates
    so each spec is counted exactly once.
    Returns a DataFrame with columns: Task, Model, Gherkin_tokens.
    """
    records = []
    for fname in os.listdir(input_dir):
        if not fname.lower().endswith(('.xls','.xlsx')):
            continue

        # extract user number (e.g. "P3_Parham.xlsx" → 3)
        m = re.search(r'P(\d+)_', fname)
        if not m:
            continue
        user = int(m.group(1))

        # open the workbook
        wb = pd.ExcelFile(os.path.join(input_dir, fname))
        for sheet in wb.sheet_names:
            df = pd.read_excel(wb, sheet_name=sheet, header=None)

            # your map tells us which 12 tasks this user actually saw
            for task_id, model_id in user_task_model_map[user].items():
                # rows 4–15 correspond to Task 1…12, so index = 3 + task_id
                row = 3 + task_id
                text = str(df.iat[row, 2])              # col 2 is the Gherkin spec
                # tokens = len(text.split())              # whitespace split
                tokens = gpt_token_count(text)

                records.append({
                    'Task':           task_id,
                    'Model':          model_id,
                    'Gherkin_tokens': tokens
                })

    # make a DataFrame and drop any duplicate (Task,Model) rows
    gdf = pd.DataFrame.from_records(records)
    return (
        gdf
        .drop_duplicates(subset=['Task','Model'])
        .sort_values(['Task','Model'])
        .reset_index(drop=True)
    )

import pandas as pd

def compute_unique_reg_tokens(reg_file_path):
    """
    Reads the sheet of unique regulatory provisions (30 rows):
      • Excel rows 2–31 (zero‐based index 1–30)
      • Col 0: Task ID (1–30)
      • Col 1: the provision text

    Returns a DataFrame with columns: Task (int), Reg_tokens (int).
    """
    df = pd.read_excel(reg_file_path, header=None)

    recs = []
    for idx in range(1, 31):                # Excel row 2 through 31
        # try to grab Task ID from col 0, else fall back to idx
        try:
            task = int(df.iat[idx, 0])
        except:
            task = idx

        text = str(df.iat[idx, 1])         # second column = the provision
        # tokens = len(text.split())
        tokens = gpt_token_count(text)     # use the GPT tokenizer
        recs.append({'Task': task, 'Reg_tokens': tokens})

    return pd.DataFrame(recs)

def compute_token_counts(input_dir):
    """
    Walks all Excel files and sheets, reads rows 4–15, columns 1 (reg text) 
    and 2 (Gherkin text), tokenizes on whitespace, and returns a DataFrame 
    with columns: UserNum, Task, Reg_tokens, Gherkin_tokens.
    """
    records = []
    for file in os.listdir(input_dir):
        if not file.lower().endswith(('.xlsx',' .xls')): continue
        wb = pd.ExcelFile(os.path.join(input_dir, file))
        user_match = re.search(r'(\d+)', os.path.splitext(file)[0])
        user_num = int(user_match.group(1)) if user_match else None

        for sheet in wb.sheet_names:
            df = pd.read_excel(wb, sheet_name=sheet, header=None)
            for row in range(4, 16):
                task_cell = str(df.iat[row, 0])
                m = re.search(r'(\d+)', task_cell)
                if not m: continue
                task = int(m.group(1))

                reg_txt     = str(df.iat[row, 1])
                gherkin_txt = str(df.iat[row, 2])

                records.append({
                    'UserNum':        user_num,
                    'Task':           task,
                    # 'Reg_tokens':     len(reg_txt.split()),
                    'Reg_tokens':gpt_token_count(reg_txt),
                    #'Gherkin_tokens': len(gherkin_txt.split()
                    'Gherkin_tokens': gpt_token_count(gherkin_txt)})
    return pd.DataFrame(records)

def compute_pair_stats(long_df, pair_size=2):
    """
    Groups users into pairs in the order of their numeric identifier (e.g., P1 with P2, P3 with P4, etc.)
    and computes statistics (mean, median, standard deviation) for each criterion for that pair.
    
    Parameters:
      long_df (pd.DataFrame): Long-format DataFrame with columns including "User", "Criterion", and "Rating (num)".
      pair_size (int): Number of users per group (default is 2).
      
    Returns:
      dict: A dictionary where each key is a criterion and the value is a DataFrame with pair-wise statistics.
    """
    # Get unique users in the order they appear
    users = list(dict.fromkeys(long_df["User"]))
    
    # Sort users by their numeric identifier extracted from the user string (e.g., "P1_Parham_Sheet1")
    users.sort(key=lambda x: int(x.split('_')[0].lstrip('P')))
    
    # Group users in pairs according to the sorted order
    user_pairs = [users[i:i+pair_size] for i in range(0, len(users), pair_size)]
    
    stats_by_criterion = {}
    for criterion in long_df["Criterion"].unique():
        criterion_data = long_df[long_df["Criterion"] == criterion]
        pair_stats = []
        for pair in user_pairs:
            pair_data = criterion_data[criterion_data["User"].isin(pair)]
            if not pair_data.empty:
                stats = {
                    "Pair": pair,
                    "Mean": pair_data["Rating (num)"].mean(),
                    "Median": pair_data["Rating (num)"].median(),
                    "Std": pair_data["Rating (num)"].std()
                }
                pair_stats.append(stats)
        stats_by_criterion[criterion] = pd.DataFrame(pair_stats)
    return stats_by_criterion

# Export the rating_map too in case other modules need it.
__all__ = [
    "rating_map",
    "read_excel_data",
    "create_summary_dfs",
    "write_summary_excel",
    "read_long_data",
    "compute_pair_stats"
]