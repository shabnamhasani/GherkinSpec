import seaborn as sns
import os
import matplotlib.pyplot as plt
import re
import pandas as pd

def plot_boxplots_by_criterion(long_df, qualitative_mappings, output_dir):
    """
    Generates separate boxplots per criterion.
    Saves each boxplot to output_dir.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # You can define a palette of colors to cycle through
    box_colors = [
        "#A6CEE3", "#1F78B4", "#B2DF8A", "#33A02C", 
        "#FB9A99", "#E31A1C", "#FDBF6F", "#FF7F00"
    ]

    for i, criterion in enumerate(long_df['Criterion'].unique()):
        criterion_df = long_df[long_df['Criterion'] == criterion]

        # Check pivot table for the "Model" information as well
        pivot_df = criterion_df.pivot_table(
            index=["Task", "Model"],
            columns="User",
            values="Rating (num)"
        )
        # print(f"Pivot for {criterion}:\n", pivot_df)
        pivot_df.to_csv(f"/home/shabnam/Gherkin/output/pivot_checks/{criterion}_pivot_check.csv")

        plt.figure(figsize=(10, 6))
        sns.set(style="whitegrid", font_scale=1.2)

        ax = sns.boxplot(
            data=criterion_df,
            x="Criterion",
            y="Rating (num)",
            showmeans=False,
            showcaps=True,
            medianprops={"color": "black", "linewidth": 2, "linestyle": "--"},
            boxprops={"facecolor": box_colors[i % len(box_colors)], "edgecolor": "gray"},
            whiskerprops={"color": "gray"},
        )
        
        # Use criterion-specific qualitative mapping
        mapping = qualitative_mappings.get(criterion.strip().title())
        if mapping:
            sorted_keys = sorted(mapping.keys())
            ax.set_yticks(sorted_keys)
            ax.set_yticklabels([mapping[k] for k in sorted_keys], fontsize=12)
        
        ax.set_title(f"Distribution of Ratings for {criterion}", fontsize=16)
        
        # Remove axis labels
        ax.set_xlabel("")
        ax.set_ylabel("")
        
        # Increase tick label font size
        ax.tick_params(axis='x', labelsize=13)
        ax.tick_params(axis='y', labelsize=13)

        plt.xticks(rotation=0)
        plt.tight_layout()

        plot_path = os.path.join(output_dir, f"{criterion}_boxplot.pdf")
        plt.savefig(plot_path)
        plt.close()
        # print(f"Saved boxplot for {criterion} to: {plot_path}")
def plot_boxplots_by_criterion_and_model(long_df, qualitative_mappings, output_dir):
    """
    Generates separate boxplots per criterion for each model.
    Saves each boxplot to output_dir.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Define the same palette as in the first function
    box_colors = [
        "#A6CEE3", "#1F78B4", "#B2DF8A", "#33A02C", 
        "#FB9A99", "#E31A1C", "#FDBF6F", "#FF7F00"
    ]
    
    # Get a consistent ordering of criteria
    criteria_list = long_df['Criterion'].unique()
    
    # Iterate over each criterion with its index to use for color selection.
    for idx, criterion in enumerate(criteria_list):
        # Get the color for this criterion from the palette
        color = box_colors[idx % len(box_colors)]
        
        # For each model, filter the data.
        for model in long_df['Model'].dropna().unique():
            subset_df = long_df[(long_df['Criterion'] == criterion) & (long_df['Model'] == model)]
            
            plt.figure(figsize=(10, 6))
            sns.set(style="whitegrid", font_scale=1.2)
            
            ax = sns.boxplot(
                data=subset_df,
                x="Criterion",
                y="Rating (num)",
                showmeans=False,
                showcaps=True,
                medianprops={"color": "black", "linewidth": 2, "linestyle": "--"},
                boxprops={"facecolor": color, "edgecolor": "gray"},
                whiskerprops={"color": "gray"},
            )
            
            # Apply criterion-specific qualitative mapping if provided.
            mapping = qualitative_mappings.get(criterion.strip().title())
            if mapping:
                sorted_keys = sorted(mapping.keys())
                ax.set_yticks(sorted_keys)
                ax.set_yticklabels([mapping[k] for k in sorted_keys], fontsize=12)
            
            ax.set_title(f"Distribution of {criterion} Ratings for {model}", fontsize=16)
            ax.set_xlabel("")
            ax.set_ylabel("")
            ax.tick_params(axis='x', labelsize=13)
            ax.tick_params(axis='y', labelsize=13)
            plt.xticks(rotation=0)
            plt.tight_layout()
            
            plot_path = os.path.join(output_dir, f"{criterion}_{model}_boxplot.pdf")
            plt.savefig(plot_path)
            plt.close()
            print(f"Saved boxplot for {criterion} for {model} to: {plot_path}")
def plot_heatmaps_by_criterion(long_df, qualitative_mappings, output_dir):
    """
    Generates a heatmap per criterion, showing task scores for each user using qualitative labels.
    Saves each heatmap to output_dir.
    """
    import matplotlib.colors as mcolors

    os.makedirs(output_dir, exist_ok=True)

    criteria = long_df["Criterion"].unique()

    for criterion in criteria:
        df = long_df[long_df["Criterion"] == criterion].copy()

        # Create a short user label like "User 1", "User 2", ...
        df["ShortUser"] = df["UserNum"].apply(lambda x: f"User {x}")

        # Create pivot table using short user IDs
        pivot = df.pivot_table(index="ShortUser", columns="Task", values="Rating (num)")

        # Sort users numerically
        pivot = pivot.reindex(sorted(pivot.index, key=lambda x: int(x.split()[-1])))

        # Replace numeric values with qualitative labels for display
        label_map = qualitative_mappings.get(criterion.strip().title(), {})
        pivot_labels = pivot.applymap(lambda x: label_map.get(x, "") if not pd.isna(x) else "")

        # Create custom colormap
        cmap_colors = sns.color_palette("YlGnBu", len(label_map))
        sorted_keys = sorted(label_map.keys(), reverse=False)
        bounds = sorted_keys + [max(sorted_keys) + 1]
        norm = mcolors.BoundaryNorm(bounds, len(bounds)-1)
        cmap = mcolors.ListedColormap(cmap_colors)

        plt.figure(figsize=(12, 8))
        sns.set(style="whitegrid", font_scale=1.1)

        # Draw heatmap without annotations
        ax = sns.heatmap(
            pivot,
            cmap=cmap,
            cbar_kws={"ticks": sorted_keys, "label": "Rating"},
            linewidths=0.5,
            linecolor="gray",
            vmin=min(sorted_keys),
            vmax=max(sorted_keys),
            annot=False,
            norm=norm
        )

        # Apply qualitative labels on colorbar
        colorbar = ax.collections[0].colorbar
        colorbar.set_ticks(sorted_keys)
        colorbar.set_ticklabels([label_map[k] for k in sorted_keys])

        ax.set_title(f"{criterion} Ratings (Qualitative)", fontsize=16)
        ax.set_xlabel("Task")
        ax.set_ylabel("User")
        plt.tight_layout()

        plot_path = os.path.join(output_dir, f"{criterion}_heatmap.pdf")
        plt.savefig(plot_path)
        plt.close()
        print(f"✅ Saved heatmap for {criterion} to: {plot_path}")
def plot_beeswarm_by_criterion(long_df, qualitative_mappings, output_dir):

    """
    Generates a beeswarm (strip) plot per criterion with jitter to show all individual ratings.
    Saves each plot to output_dir.
    """
    os.makedirs(output_dir, exist_ok=True)

    # You can define a palette of colors to cycle through
    point_colors = [
        "#A6CEE3", "#1F78B4", "#B2DF8A", "#33A02C", 
        "#FB9A99", "#E31A1C", "#FDBF6F", "#FF7F00"
    ]

    for i, criterion in enumerate(long_df['Criterion'].unique()):
        criterion_df = long_df[long_df['Criterion'] == criterion]

        plt.figure(figsize=(10, 6))
        sns.set(style="whitegrid", font_scale=1.2)

        ax = sns.stripplot(
            data=criterion_df,
            x="Criterion",
            y="Rating (num)",
            jitter=True,
            alpha=0.7,
            size=6,
            color=point_colors[i % len(point_colors)],
        )
        # Compute the median value for the current criterion
        median_val = criterion_df["Rating (num)"].median()
        # Overlay a horizontal dashed line at the median    
        ax.axhline(median_val, color='black', linestyle='--', linewidth=2)

        # Apply criterion-specific qualitative mapping if available
        mapping = qualitative_mappings.get(criterion.strip().title())
        if mapping:
            sorted_keys = sorted(mapping.keys())
            ax.set_yticks(sorted_keys)
            ax.set_yticklabels([mapping[k] for k in sorted_keys], fontsize=12)

        ax.set_title(f"Beeswarm Plot of Ratings for {criterion}", fontsize=16)
        ax.set_xlabel("")
        ax.set_ylabel("")

        ax.tick_params(axis='x', labelsize=13)
        ax.tick_params(axis='y', labelsize=13)

        plt.xticks(rotation=0)
        plt.tight_layout()

        plot_path = os.path.join(output_dir, f"{criterion}_beeswarm.pdf")
        plt.savefig(plot_path)
        plt.close()
def plot_beeswarm_by_criterion_and_model(long_df, qualitative_mappings, output_dir):
    """
    Generates beeswarm (strip) plots per criterion and per model.
    Saves each plot to output_dir.

    Parameters:
      long_df (pd.DataFrame): Long-format DataFrame with columns including "Criterion",
                              "Rating (num)", and "Model".
      qualitative_mappings (dict): Mapping for each criterion that converts numeric ratings
                                   to qualitative labels.
      output_dir (str): Directory where the plots will be saved.
    """
    import os
    import matplotlib.pyplot as plt
    import seaborn as sns

    os.makedirs(output_dir, exist_ok=True)

    # Define a palette of colors to cycle through -- using same colors as your beeswarm function.
    point_colors = [
        "#A6CEE3", "#1F78B4", "#B2DF8A", "#33A02C",
        "#FB9A99", "#E31A1C", "#FDBF6F", "#FF7F00"
    ]

    # Iterate through each unique criterion; use the index for color selection.
    for i, criterion in enumerate(long_df['Criterion'].unique()):
        color = point_colors[i % len(point_colors)]
        # For each model (e.g., "Claude", "Llama")
        for model in long_df['Model'].dropna().unique():
            # Filter the data by criterion and model.
            subset_df = long_df[(long_df['Criterion'] == criterion) & (long_df['Model'] == model)]
            if subset_df.empty:
                continue

            plt.figure(figsize=(10, 6))
            sns.set(style="whitegrid", font_scale=1.2)

            ax = sns.stripplot(
                data=subset_df,
                x="Criterion",
                y="Rating (num)",
                jitter=True,
                alpha=0.7,
                size=6,
                color=color
            )
            # Compute the median value for the current subset.
            median_val = subset_df["Rating (num)"].median()
            # Overlay a horizontal dashed line at the median.
            ax.axhline(median_val, color='black', linestyle='--', linewidth=2)

            # Apply qualitative mapping if available.
            mapping = qualitative_mappings.get(criterion.strip().title())
            if mapping:
                sorted_keys = sorted(mapping.keys())
                ax.set_yticks(sorted_keys)
                ax.set_yticklabels([mapping[k] for k in sorted_keys], fontsize=12)

            ax.set_title(f"Beeswarm Plot of {criterion} Ratings for {model}", fontsize=16)
            ax.set_xlabel("")
            ax.set_ylabel("")
            ax.tick_params(axis='x', labelsize=13)
            ax.tick_params(axis='y', labelsize=13)

            plt.xticks(rotation=0)
            plt.tight_layout()

            plot_path = os.path.join(output_dir, f"{criterion}_{model}_beeswarm.pdf")
            plt.savefig(plot_path)
            plt.close()
# Updated functions with directory checks, imports, and improved layout adjustments

import os
import matplotlib.pyplot as plt

def plot_stacked_bar_by_criterion(long_df, qualitative_mappings, output_dir):
    """
    For each criterion, draw a single horizontal stacked bar whose
    segments are the counts of each rating (label), ordered from
    highest→lowest. Saves one PDF per criterion in output_dir.
    """
    os.makedirs(output_dir, exist_ok=True)
    df = long_df.copy()
    df["Criterion_clean"] = df["Criterion"].str.strip()

    for criterion, mapping in qualitative_mappings.items():
        sub = df[df["Criterion_clean"] == criterion]
        if sub.empty:
            print(f"⚠️ No data for '{criterion}'")
            continue

        keys = sorted(mapping.keys(), reverse=True)
        labels = [mapping[k] for k in keys]
        counts = sub["Rating (num)"].value_counts().reindex(keys, fill_value=0)

        fig, ax = plt.subplots(figsize=(10, 2))
        left = 0
        for k, lbl in zip(keys, labels):
            cnt = counts.loc[k]
            ax.barh(0, cnt, left=left, height=0.6, label=lbl)
            if cnt > 0:
                ax.text(left + cnt / 2, 0, str(cnt),
                        va="center", ha="center",
                        color="white", fontweight="bold", fontsize=10)
            left += cnt

        ax.set_xlim(0, left)
        ax.set_ylim(-0.5, 0.5)
        ax.set_yticks([])
        ax.set_xlabel("Count")
        ax.set_title(f"{criterion} distribution", fontsize=14)

        # Legend outside at top-right
        legend = ax.legend(
            ncol=1,
            loc="upper left",
            bbox_to_anchor=(1.3, 1.0),
            borderaxespad=0
        )

        # Reserve space for the legend
        fig.subplots_adjust(left=0.05, right=0.60, top=0.85)

        out_path = os.path.join(output_dir, f"{criterion.replace(' ', '_')}_stacked_bar.pdf")
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)

def plot_stacked_bar_by_criterion_and_model(long_df, qualitative_mappings, output_dir):
    """
    For each criterion × model, draw a horizontal stacked‐bar of counts
    and save one PDF per (criterion, model).
    """
    os.makedirs(output_dir, exist_ok=True)
    df = long_df.copy()
    df["Criterion_clean"] = df["Criterion"].str.strip()
    models = df["Model"].dropna().unique()

    for criterion, mapping in qualitative_mappings.items():
        for model in models:
            sub = df[(df["Criterion_clean"] == criterion) & (df["Model"] == model)]
            if sub.empty:
                continue

            keys = sorted(mapping.keys(), reverse=True)
            labels = [mapping[k] for k in keys]
            counts = sub["Rating (num)"].value_counts().reindex(keys, fill_value=0)

            fig, ax = plt.subplots(figsize=(10, 2))
            left = 0
            for k, lbl in zip(keys, labels):
                cnt = counts.loc[k]
                ax.barh(0, cnt, left=left, height=0.6, label=lbl)
                if cnt > 0:
                    ax.text(left + cnt / 2, 0, str(cnt),
                            va="center", ha="center",
                            color="white", fontweight="bold", fontsize=10)
                left += cnt

            ax.set_xlim(0, left)
            ax.set_ylim(-0.5, 0.5)
            ax.set_yticks([])
            ax.set_xlabel("Count")
            ax.set_title(f"{criterion} distribution — {model}", fontsize=14)

            ax.legend(
                ncol=1,
                loc="upper left",
                bbox_to_anchor=(1.3, 1.0),
                borderaxespad=0
            )
            fig.subplots_adjust(left=0.05, right=0.60, top=0.85)

            fname = f"{criterion.replace(' ', '_')}_{model}_stacked_bar.pdf"
            out_path = os.path.join(output_dir, fname)
            fig.savefig(out_path, bbox_inches="tight")
            plt.close(fig)

