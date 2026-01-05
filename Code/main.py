from boxplot import plot_boxplots_by_criterion,plot_boxplots_by_criterion_and_model,plot_heatmaps_by_criterion,plot_beeswarm_by_criterion,plot_beeswarm_by_criterion_and_model,plot_stacked_bar_by_criterion,plot_stacked_bar_by_criterion_and_model,plot_token_scatter,generate_token_step_boxplots
from wilcoxon_stats import (
    compute_pair_wilcoxon,
    compute_model_wilcoxon,
    compute_pair_signed_wilcoxon,
    compute_model_signed_wilcoxon,
    a12,
    ranksums,
)
import os
import numpy as np
import matplotlib.pyplot as plt
import argparse
import pandas as pd
from dataprocessing import (
    read_excel_data,
    create_summary_dfs,
    write_summary_excel,
    read_long_data,
    rating_map,
    compute_pair_stats,
    compute_token_counts,
    get_token_stats,
    average_steps_per_scenario,
    average_step_length,
    get_token_lists,
    get_steps_per_scenario_list,
    get_step_length_list
)

def main():
    input_dir = "/Gherkin/Data/input"
    output_path = "/Gherkin/Evaluation/long_dfs.xlsx"
    output_boxplots_dir = "/Gherkin/Evaluation/boxplots"
    output_wilcoxon_dir = "/Gherkin/Evaluation/wilcoxon"
    output_stacked_bar_dir = "/Gherkin/Evaluation/stacked_bar"
    output_tokenstats_dir= "/Gherkin/Evaluation/token_stats"

    # Step 1: Read and process data
    all_user_data, task_ids, criteria_names = read_excel_data(input_dir, rating_map)
    summary_dfs = create_summary_dfs(all_user_data, task_ids, criteria_names)

    # Step 2: Write summary to Excel
    # try:
    #     write_summary_excel(output_path, summary_dfs)
    # except Exception as e:
    #     print(f"Error writing summary: {e}")

    # Step 3: Read long-format data
    long_df = read_long_data(input_dir, rating_map)
    # print(long_df.head(5))  # Debugging: Check the structure of the DataFrame
    # print(long_df.columns)
    write_summary_excel(output_path, long_df)

    # Step 4: Generate boxplots
    
    qualitative_mappings = {
    "Relevance":{
    5:"Fully relevant",
    4:"Mostly relevant",
    3:"Partly relevant",
    2:"Mostly irrelevant",
    1:"Fully irrelevant"},

    "Completeness": {
    5:"Fully complete",
    4:"Mostly complete",
    3:"Somewhat incomplete",
    2:"Mostly incomplete",
    1:"Fully incomplete"},

    "Clarity":{
    5:"Fully clear",
    4:"Mostly clear",
    3:"Partly clear",
    2:"Mostly unclear",
    1:"Fully unclear"},

    "Singularity":{
    5:"Fully singular",
    4:"Mostly singular",
    3:"Partly singular",
    2:"Mostly mixed",
    1:"Fully mixed"},

    "Time Savings":{
    5:"Maximum time saved",
    4:"Significant time saved",
    3:"Moderate time saved",
    2:"Minimal time saved",
    1:"No time saved"}
    }
    # plot_boxplots_by_criterion(long_df, qualitative_mappings, output_boxplots_dir)
    # plot_boxplots_by_criterion_and_model(long_df, qualitative_mappings, output_boxplots_dir)

    # Step 5: Compute Wilcoxon rank-sum tests
    try:
        wilcoxon_results = compute_pair_wilcoxon(long_df)
        
        # For each criterion, save results to CSV (similar to what you do with pair_stats)

        for criterion, df in wilcoxon_results.items():
            out_file = os.path.join(output_wilcoxon_dir, f"{criterion}_pair_wilcoxon.csv")
            df.to_csv(out_file, index=False)
    
    except Exception as e:
        print(f"Error computing Wilcoxon stats: {e}")

    # Paired Wilcoxon signed-rank tests for participant pairs (per-criterion)
    try:
        pair_signed_results = compute_pair_signed_wilcoxon(long_df)
        for criterion, df in pair_signed_results.items():
            out_file = os.path.join(output_wilcoxon_dir, f"{criterion}_pair_signed_wilcoxon.csv")
            df.to_csv(out_file, index=False)
    except Exception as e:
        print(f"Error computing paired Wilcoxon (participants): {e}")

    # Option 1: Compare a specific pair of models (e.g., "Llama" vs "Claude")
    model_comparison = ("Llama", "Claude")
    wilcox_results = compute_model_wilcoxon(long_df, model_pair=model_comparison)

    for crit, result_df in wilcox_results.items():
        output_path = os.path.join(output_wilcoxon_dir, f"{crit}_model_wilcoxon.csv")
        result_df.to_csv(output_path, index=False)
    # Paired Wilcoxon signed-rank tests for model comparisons (per-criterion)
    try:
        model_signed_results = compute_model_signed_wilcoxon(long_df, model_pair=model_comparison)
        for crit, df in model_signed_results.items():
            out_file = os.path.join(output_wilcoxon_dir, f"{crit}_model_signed_wilcoxon.csv")
            df.to_csv(out_file, index=False)
    except Exception as e:
        print(f"Error computing paired Wilcoxon (models): {e}")
    # step 6: Generate stacked bar plots
    plot_stacked_bar_by_criterion(long_df, qualitative_mappings, output_stacked_bar_dir)
    plot_stacked_bar_by_criterion_and_model(long_df, qualitative_mappings, output_stacked_bar_dir)

    # Step 7: Generate token scatter plots

    # --- token‐count analysis ---
    #Summarize tokens per (User, Task) without dropping any overlapping pairs
    token_df = compute_token_counts(long_df)
    #Save raw token summary
    columns_to_save = ['User', 'UserNum', 'Task', 'Reg_tokens', 'Gherkin_tokens', 'Model']
    token_df[columns_to_save].to_csv(os.path.join(output_tokenstats_dir, "token_summary.csv"), index=False)
    
    #Scatter plot of Reg_tokens vs. Gherkin_tokens, colored by model
    plot_token_scatter(token_df, output_tokenstats_dir)

    #Compute and save overall token statistics
    stats = get_token_stats(token_df)
    stats.to_csv(os.path.join(output_tokenstats_dir, 'token_counts_summary.csv'))

    #Scenario step statistics (requires 'Gherkin_text' in long_df)
    avg_steps = average_steps_per_scenario(token_df, text_col='Gherkin_text')
    avg_word_len  = average_step_length(token_df, 'Gherkin_text', mode='words')
    avg_char_len  = average_step_length(token_df, 'Gherkin_text', mode='chars')
    avg_token_len = average_step_length(token_df, 'Gherkin_text', mode='gpt_tokens')

    #Get lists of tokens and step lengths
    regs, gherkin = get_token_lists(token_df)
    steps_per_scenario_list = get_steps_per_scenario_list(token_df, text_col='Gherkin_text')
    step_length_lists_words = get_step_length_list(token_df, text_col='Gherkin_text', mode='words')
    step_length_lists_chars = get_step_length_list(token_df, text_col='Gherkin_text', mode='chars')
    step_length_lists_token = get_step_length_list(token_df, text_col='Gherkin_text', mode='gpt_tokens')

    #Save token lists and step lengths
        # Convert list to DataFrame
    pd.DataFrame(regs).to_csv(os.path.join(output_tokenstats_dir, 'token_lists.csv'), index=False)
    pd.DataFrame(gherkin).to_csv(os.path.join(output_tokenstats_dir, 'gherkin_token_lists.csv'), index=False)
    pd.DataFrame(step_length_lists_words).to_csv(os.path.join(output_tokenstats_dir, 'step_length_lists_words.csv'), index=False)
    pd.DataFrame(step_length_lists_chars).to_csv(os.path.join(output_tokenstats_dir, 'step_length_lists_chars.csv'), index=False)
    pd.DataFrame(step_length_lists_token).to_csv(os.path.join(output_tokenstats_dir, 'step_length_lists_token.csv'), index=False)
    pd.DataFrame(steps_per_scenario_list).to_csv(os.path.join(output_tokenstats_dir, 'steps_per_scenario_list.csv'), index=False)

    # … your existing code that writes the CSVs …

    generate_token_step_boxplots(
        input_dir=output_tokenstats_dir,
        output_dir=os.path.join(output_tokenstats_dir, "boxplots")
    )
    #Save scenario step statistics
    with open(os.path.join(output_tokenstats_dir, 'scenario_step_stats.txt'), 'w') as f:
        f.write(f'Average steps per scenario: {avg_steps}\n')
        f.write(f'Average step length (words): {avg_word_len}\n')
        f.write(f'Average step length (chars): {avg_char_len}\n')
        f.write(f'Average step length (tokens): {avg_token_len}\n')

    # ---- debug prints ----
    # print("Total rows in token_df:", len(token_df))
    # print(token_df['Model'].value_counts(dropna=False))
    # print(token_df['Model'].unique())

    # bad = token_df[~token_df['Model'].isin(['Llama','Claude'])]
    # if not bad.empty:
    #     print("Rows with unmapped model (will not be plotted):")
    #     print(bad[['UserNum','Task','Model']].drop_duplicates())

    # ---- end debug prints ----
    # Step 8: Statistics for specific model comparison
    # Example: Wilcoxon rank-sum test between Llama and Claude token counts
    llama_tokens = token_df[token_df["Model"] == "Llama"]["Gherkin_tokens"]
    claude_tokens = token_df[token_df["Model"] == "Claude"]["Gherkin_tokens"]

    # Wilcoxon rank-sum test
    stat, p_value = ranksums(llama_tokens, claude_tokens)

    # Vargha–Delaney A12
    a12_value, magnitude = a12(llama_tokens.tolist(), claude_tokens.tolist())

    # print(f"Wilcoxon rank-sum: statistic={stat:.3f}, p={p_value:.4f}")
    # print(f"Vargha–Delaney A12 = {a12_value:.3f} ({magnitude} effect)")
    # print("llama median", np.median(llama_tokens)),
    # print("llama mean", np.mean(llama_tokens))
    # print("claude median", np.median(claude_tokens)),
    # print("claude mean", np.mean(claude_tokens))

if __name__ == "__main__":
    main()
