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
)
from boxplot import plot_boxplots_by_criterion,plot_boxplots_by_criterion_and_model,plot_heatmaps_by_criterion,plot_beeswarm_by_criterion,plot_beeswarm_by_criterion_and_model,plot_stacked_bar_by_criterion,plot_stacked_bar_by_criterion_and_model,plot_token_scatter
from wilcoxon_stats import compute_pair_wilcoxon,compute_model_wilcoxon
import os
import matplotlib.pyplot as plt

def main():

    input_dir = "/home/shabnam/Gherkin/input"
    output_path = "/home/shabnam/Gherkin/output/summary.xlsx"
    output_boxplots_dir = "/home/shabnam/Gherkin/output/boxplots"
    output_stats_dir = "/home/shabnam/Gherkin/output/pair_stats"
    output_heatmap_dir = "/home/shabnam/Gherkin/output/heatmaps"
    output_beeswarm_dir = "/home/shabnam/Gherkin/output/beeswarm"
    output_wilcoxon_dir = "/home/shabnam/Gherkin/output/wilcoxon"
    output_stacked_bar_dir = "/home/shabnam/Gherkin/output/stacked_bar"
    output_token_dir = "/home/shabnam/Gherkin/output/token"

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

    # Step 4: Compute pair statistics
    # try:
    #     pair_stats = compute_pair_stats(long_df)
    #     for criterion, stats_df in pair_stats.items():
    #         file_path = os.path.join(output_stats_dir, f"{criterion}_pair_stats.csv")
    #         stats_df.to_csv(file_path, index=False)
    #         print(f"Saved pair stats for {criterion} to {file_path}")
    # except Exception as e:
    #     print(f"Error computing pair stats: {e}")

    # Step 5: Generate boxplots
    # qualitative_mappings = {
    #     "Relevance": {5: "All Relevant", 4: "Mostly Relevant", 3: "Somewhat Relevant", 2: "Mostly Irrelevant", 1: "All Irrelevant"},
    #     "Completeness": {5: "Fully Complete", 4: "Mostly Complete", 3: "Partially Complete", 2: "Mostly Incomplete", 1: "Fully Incomplete"},
    #     "Clarity": {5: "Completely Clear", 4: "Mostly Clear", 3: "Somewhat Clear", 2: "Mostly Unclear", 1: "Completely Unclear"},
    #     "Singularity": {5: "Completely Singular", 4: "Mostly Singular", 3: "Somewhat Singular", 2: "Mostly Mixed", 1: "Completely Mixed"},
    #     "Time Savings": {5: "Completely Helpful", 4: "Mostly Helpful", 3: "Somewhat Helpful", 2: "Mostly Unhelpful", 1: "Completely Unhelpful"},
    # }
    # plot_boxplots_by_criterion(long_df, qualitative_mappings, output_boxplots_dir)
    # plot_boxplots_by_criterion_and_model(long_df, qualitative_mappings, output_boxplots_dir)

    # # Step 6: Generate heatmaps
    # plot_heatmaps_by_criterion(long_df, qualitative_mappings, output_heatmap_dir)
    # # Step 7: Generate beeswarm plots
    # plot_beeswarm_by_criterion(long_df, qualitative_mappings, output_beeswarm_dir)
    # plot_beeswarm_by_criterion_and_model(long_df, qualitative_mappings, output_beeswarm_dir)

    # Step 8: Compute Wilcoxon rank-sum tests
    # try:
    #     wilcoxon_results = compute_pair_wilcoxon(long_df)
        
    #     # For each criterion, save results to CSV (similar to what you do with pair_stats)

    #     for criterion, df in wilcoxon_results.items():
    #         out_file = os.path.join(output_wilcoxon_dir, f"{criterion}_pair_wilcoxon.csv")
    #         df.to_csv(out_file, index=False)
    
    # except Exception as e:
    #     print(f"Error computing Wilcoxon stats: {e}")

    # # Option 1: Compare a specific pair of models (e.g., "Llama" vs "Claude")
    # model_comparison = ("Llama", "Claude")
    # wilcox_results = compute_model_wilcoxon(long_df, model_pair=model_comparison)

    # for crit, result_df in wilcox_results.items():
    #     output_path = os.path.join(output_wilcoxon_dir, f"{crit}_model_wilcoxon.csv")
    #     result_df.to_csv(output_path, index=False)
    #step 9: Generate stacked bar plots
    # plot_stacked_bar_by_criterion(long_df, qualitative_mappings, output_stacked_bar_dir)
    # plot_stacked_bar_by_criterion_and_model(long_df, qualitative_mappings, output_stacked_bar_dir)

    # Step 10: Generate token scatter plots

    # --- token‐count analysis ---
    #Summarize tokens per (User, Task) without dropping any overlapping pairs
    token_df = compute_token_counts(long_df)
    #Save raw token summary
    columns_to_save = ['User', 'UserNum', 'Task', 'Reg_tokens', 'Gherkin_tokens', 'Model']
    token_df[columns_to_save].to_csv(os.path.join(output_token_dir, "token_summary.csv"), index=False)
    
    #Scatter plot of Reg_tokens vs. Gherkin_tokens, colored by model
    plot_token_scatter(token_df, output_token_dir)

    #Compute and save overall token statistics
    stats = get_token_stats(token_df)
    stats.to_csv(os.path.join(output_token_dir, 'token_counts_summary.csv'))

    #Scenario step statistics (requires 'Gherkin_text' in long_df)
    avg_steps = average_steps_per_scenario(token_df, text_col='Gherkin_text')
    avg_word_len  = average_step_length(token_df, 'Gherkin_text', mode='words')
    avg_char_len  = average_step_length(token_df, 'Gherkin_text', mode='chars')
    avg_token_len = average_step_length(token_df, 'Gherkin_text', mode='gpt_tokens')


    #Save scenario step statistics
    with open(os.path.join(output_token_dir, 'scenario_step_stats.txt'), 'w') as f:
        f.write(f'Average steps per scenario: {avg_steps}\n')
        f.write(f'Average step length (words): {avg_word_len}\n')
        f.write(f'Average step length (chars): {avg_char_len}\n')
        f.write(f'Average step length (tokens): {avg_token_len}\n')

    # ---- debug prints ----
    print("Total rows in token_df:", len(token_df))
    print(token_df['Model'].value_counts(dropna=False))
    print(token_df['Model'].unique())

    bad = token_df[~token_df['Model'].isin(['Llama','Claude'])]
    if not bad.empty:
        print("Rows with unmapped model (will not be plotted):")
        print(bad[['UserNum','Task','Model']].drop_duplicates())



if __name__ == "__main__":
    main()