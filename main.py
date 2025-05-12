from dataprocessing import (
    read_excel_data,
    create_summary_dfs,
    write_summary_excel,
    read_long_data,
    rating_map,
    compute_pair_stats,
    compute_unique_gherkin_counts,
    compute_unique_reg_tokens,
    compute_token_counts
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
    try:
        write_summary_excel(output_path, summary_dfs)
    except Exception as e:
        print(f"Error writing summary: {e}")

    # Step 3: Read long-format data
    long_df = read_long_data(input_dir, rating_map)
    # print(long_df.head(5))  # Debugging: Check the structure of the DataFrame
    # print(long_df.columns)

    # Step 4: Compute pair statistics
    try:
        pair_stats = compute_pair_stats(long_df)
        for criterion, stats_df in pair_stats.items():
            file_path = os.path.join(output_stats_dir, f"{criterion}_pair_stats.csv")
            stats_df.to_csv(file_path, index=False)
            print(f"Saved pair stats for {criterion} to {file_path}")
    except Exception as e:
        print(f"Error computing pair stats: {e}")

    # Step 5: Generate boxplots
    qualitative_mappings = {
        "Relevance": {5: "All Relevant", 4: "Mostly Relevant", 3: "Somewhat Relevant", 2: "Mostly Irrelevant", 1: "All Irrelevant"},
        "Completeness": {5: "Fully Complete", 4: "Mostly Complete", 3: "Partially Complete", 2: "Mostly Incomplete", 1: "Fully Incomplete"},
        "Clarity": {5: "Completely Clear", 4: "Mostly Clear", 3: "Somewhat Clear", 2: "Mostly Unclear", 1: "Completely Unclear"},
        "Singularity": {5: "Completely Singular", 4: "Mostly Singular", 3: "Somewhat Singular", 2: "Mostly Mixed", 1: "Completely Mixed"},
        "Time Savings": {5: "Completely Helpful", 4: "Mostly Helpful", 3: "Somewhat Helpful", 2: "Mostly Unhelpful", 1: "Completely Unhelpful"},
    }
    plot_boxplots_by_criterion(long_df, qualitative_mappings, output_boxplots_dir)
    plot_boxplots_by_criterion_and_model(long_df, qualitative_mappings, output_boxplots_dir)

    # Step 6: Generate heatmaps
    plot_heatmaps_by_criterion(long_df, qualitative_mappings, output_heatmap_dir)
    # Step 7: Generate beeswarm plots
    plot_beeswarm_by_criterion(long_df, qualitative_mappings, output_beeswarm_dir)
    plot_beeswarm_by_criterion_and_model(long_df, qualitative_mappings, output_beeswarm_dir)

    # Step 8: Compute Wilcoxon rank-sum tests
    try:
        wilcoxon_results = compute_pair_wilcoxon(long_df)
        
        # For each criterion, save results to CSV (similar to what you do with pair_stats)

        for criterion, df in wilcoxon_results.items():
            out_file = os.path.join(output_wilcoxon_dir, f"{criterion}_pair_wilcoxon.csv")
            df.to_csv(out_file, index=False)
    
    except Exception as e:
        print(f"Error computing Wilcoxon stats: {e}")

    # Option 1: Compare a specific pair of models (e.g., "Llama" vs "Claude")
    model_comparison = ("Llama", "Claude")
    wilcox_results = compute_model_wilcoxon(long_df, model_pair=model_comparison)

    for crit, result_df in wilcox_results.items():
        output_path = os.path.join(output_wilcoxon_dir, f"{crit}_model_wilcoxon.csv")
        result_df.to_csv(output_path, index=False)
    #step 9: Generate stacked bar plots
    plot_stacked_bar_by_criterion(long_df, qualitative_mappings, output_stacked_bar_dir)
    plot_stacked_bar_by_criterion_and_model(long_df, qualitative_mappings, output_stacked_bar_dir)

    # Step 10: Generate token scatter plots
    # 1) load the 60 unique Gherkin counts
    # gherkin_df = compute_unique_gherkin_counts(input_dir)
    # #    → columns: Task, Model, Gherkin_tokens

    # # 2) load the 30 unique regulatory counts
    # reg_df = compute_unique_reg_tokens('/home/shabnam/Gherkin/token input/Uniqe Tasks.xlsx')
    # #    → columns: Task, Reg_tokens

    # # 3) merge so each (Task,Model) row has its Reg_tokens
    # merged = gherkin_df.merge(reg_df, on='Task')

    # # 4) summary statistics
    # reg_stats     = merged['Reg_tokens'].agg(['mean','median','std'])
    # gherkin_stats = merged['Gherkin_tokens'].agg(['mean','median','std'])

    # print("Reg tokens:",     reg_stats.to_dict())
    # print("Gherkin tokens:", gherkin_stats.to_dict())

    # # 5) scatter plot
    # plt.figure(figsize=(8,6))
    # plt.scatter(merged['Reg_tokens'], merged['Gherkin_tokens'], alpha=0.7)
    # plt.xlabel('Regulatory Provision Token Count')
    # plt.ylabel('Gherkin Token Count')
    # plt.title('Reg vs. Gherkin Token Counts (60 specs)')
    # plt.tight_layout()
    # plt.savefig(os.path.join(output_token_dir, 'reg_vs_gherkin_scatter.pdf'))
    # plt.show()
    # --- token‐count analysis ---
    token_df = compute_token_counts(input_dir)

    # 1) summary stats
    stats = token_df[['Reg_tokens','Gherkin_tokens']].agg(['mean','median','std']).T
    stats.to_csv(os.path.join(output_token_dir, 'token_counts_summary.csv'))

    # 2) scatter plot
    token_plot_dir = os.path.join(output_boxplots_dir, 'token_scatter')
    plot_token_scatter(token_df, output_token_dir)



if __name__ == "__main__":
    main()