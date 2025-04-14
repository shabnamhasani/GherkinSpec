from dataprocessing import (
    read_excel_data,
    create_summary_dfs,
    write_summary_excel,
    read_long_data,
    rating_map,
    compute_pair_stats,
)
from boxplot import plot_boxplots_by_criterion,plot_boxplots_by_criterion_and_model,plot_heatmaps_by_criterion,plot_beeswarm_by_criterion,plot_beeswarm_by_criterion_and_model
import os

def main():

    input_dir = "/home/shabnam/Gherkin/input"
    output_path = "/home/shabnam/Gherkin/output/summary.xlsx"
    output_boxplots_dir = "/home/shabnam/Gherkin/output/boxplots"
    output_stats_dir = "/home/shabnam/Gherkin/output/pair_stats"
    output_heatmap_dir = "/home/shabnam/Gherkin/output/heatmaps"
    output_beeswarm_dir = "/home/shabnam/Gherkin/output/beeswarm"

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


if __name__ == "__main__":
    main()