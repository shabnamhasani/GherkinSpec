import seaborn as sns
import os
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.colors as mcolors
import numpy as np


def plot_stacked_bar_by_criterion(long_df, qualitative_mappings, output_dir):
    """
    For each criterion, draw a single horizontal stacked bar whose
    segments are the counts of each rating (label), ordered from
    highest→lowest. Saves one PDF per criterion in output_dir.
    """
    df = long_df.copy()
    df["Criterion_clean"] = df["Criterion"].str.strip()

    for criterion, mapping in qualitative_mappings.items():
        sub = df[df["Criterion_clean"] == criterion]
        if sub.empty:
            continue

        keys = sorted(mapping.keys(), reverse=True)
        labels = [mapping[k] for k in keys]
        counts = sub["Rating (num)"].value_counts().reindex(keys, fill_value=0)

        fig, ax = plt.subplots(figsize=(16, 4))  # Wider landscape format
        left = 0
        for k, lbl in zip(keys, labels):
            cnt = counts.loc[k]
            ax.barh(0, cnt, left=left, height=0.8, label=lbl)  # Increased bar height
            if cnt > 0:
                ax.text(left + cnt / 2, 0, str(cnt),
                        va="center", ha="center",
                        color="white", fontweight="bold", fontsize=24)  # Larger numbers
            left += cnt

        ax.set_xlim(0, left)
        ax.set_ylim(-0.7, 0.7)  # Adjusted for larger bar height
        ax.set_yticks([])
        ax.set_xlabel(" ", fontsize=22, fontweight="bold")  # Empty label but preserved space
        ax.set_title(f"{criterion} Distribution", fontsize=28, pad=20)

        # Legend with larger text and boxes
        ax.legend(
            ncol=1,
            loc="center left",
            bbox_to_anchor=(1.01, 0.5),
            borderaxespad=0,
            fontsize=20,
            frameon=False,
            handlelength=1.0,
            handleheight=1.0
        )

        fig.subplots_adjust(left=0.06, right=0.85, top=0.82, bottom=0.22)
        out_path = os.path.join(output_dir, f"{criterion.replace(' ', '_')}_stacked_bar.pdf")
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)

def plot_stacked_bar_by_criterion_and_model(long_df, qualitative_mappings, output_dir):
    """
    For each criterion × model, draw a horizontal stacked-bar of counts
    and save one PDF per (criterion, model).
    """
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

            fig, ax = plt.subplots(figsize=(16, 4))  # Same size as combined plot
            left = 0
            for k, lbl in zip(keys, labels):
                cnt = counts.loc[k]
                ax.barh(0, cnt, left=left, height=0.8, label=lbl)  # Same bar height
                if cnt > 0:
                    ax.text(left + cnt / 2, 0, str(cnt),
                            va="center", ha="center",
                            color="white", fontweight="bold", fontsize=24)  # Same font size
                left += cnt

            ax.set_xlim(0, left)
            ax.set_ylim(-0.7, 0.7)  # Same y-limits
            ax.set_yticks([])
            ax.set_xlabel(" ", fontsize=22, fontweight="bold")
            ax.set_title(f"{criterion} Distribution ({model})", fontsize=28, pad=20)

            # Same legend style as combined plot
            ax.legend(
                ncol=1,
                loc="center left",
                bbox_to_anchor=(1.01, 0.5),
                borderaxespad=0,
                fontsize=20,
                frameon=False,
                handlelength=1.0,
                handleheight=1.0
            )

            fig.subplots_adjust(left=0.06, right=0.85, top=0.82, bottom=0.22)
            fname = f"{criterion.replace(' ', '_')}_{model}_stacked_bar.pdf"
            out_path = os.path.join(output_dir, fname)
            fig.savefig(out_path, bbox_inches="tight")
            plt.close(fig)

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
        pivot_df.to_csv(f"/home/shabnam/Gherkin/Evaluation/pivot_checks/{criterion}_pivot_check.csv")

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
        print(f"Saved heatmap for {criterion} to: {plot_path}")
        
def plot_beeswarm_by_criterion(long_df, qualitative_mappings, output_dir):
    """
    Generates a beeswarm (strip) plot per criterion with jitter to show all individual ratings.
    Saves each plot to output_dir.
    """

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

    df = long_df.copy()
    df["Criterion_clean"] = df["Criterion"].str.strip()
    models = df["Model"].dropna().unique()

    for criterion, mapping in qualitative_mappings.items():
        for model in models:
            sub = df[(df["Criterion_clean"] == criterion) & (df["Model"] == model)]
            if sub.empty:
                continue

            keys   = sorted(mapping.keys(), reverse=True)
            labels = [mapping[k] for k in keys]
            counts = sub["Rating (num)"].value_counts().reindex(keys, fill_value=0)

            fig, ax = plt.subplots(figsize=FIGSIZE)
            ax.set_position(AX_POS)  # <- identical plot area
            _draw(ax,
                  counts=[int(counts.loc[k]) for k in keys],
                  labels=labels,
                  title=f"{criterion} distribution — {model}")

            leg = fig.legend(handles=ax.patches[:len(labels)], labels=labels,
                             loc="center left", bbox_to_anchor=LEGEND_POS,
                             frameon=False, bbox_transform=fig.transFigure)
            for t in leg.get_texts():
                t.set_fontsize(TICK_FONTSIZE)

            fname = f"{criterion.replace(' ', '_')}_{model}_stacked_bar.pdf"
            out_path = os.path.join(output_dir, fname)
            fig.savefig(out_path, bbox_inches="tight")
            plt.close(fig)

    """
    For each criterion × model, draw a horizontal stacked‐bar of counts
    and save one PDF per (criterion, model).
    """
    df = long_df.copy()
    df["Criterion_clean"] = df["Criterion"].str.strip()
    models = df["Model"].dropna().unique()

    # total ratings per criterion across ALL models (locks width to match the other function)
    totals = df.groupby("Criterion_clean").size().to_dict()

    for criterion, mapping in qualitative_mappings.items():
        for model in models:
            sub = df[(df["Criterion_clean"] == criterion) & (df["Model"] == model)]
            if sub.empty:
                continue

            keys   = sorted(mapping.keys(), reverse=True)
            labels = [mapping[k] for k in keys]
            counts = sub["Rating (num)"].value_counts().reindex(keys, fill_value=0)

            fig, ax = plt.subplots(figsize=FIGSIZE)
            left = 0
            for k, lbl in zip(keys, labels):
                cnt = int(counts.loc[k])
                ax.barh(0, cnt, left=left, height=BAR_HEIGHT, label=lbl)
                if cnt > 0:
                    ax.text(left + cnt/2, 0, str(cnt),
                            va="center", ha="center",
                            color="white", fontweight="bold", fontsize=COUNT_FONTSIZE)
                left += cnt

            # same fixed width as the all-models chart (e.g., 120)
            ax.set_xlim(0, totals.get(criterion, left))
            ax.set_ylim(-0.5, 0.5)
            ax.set_yticks([])
            ax.set_xlabel("Count", fontsize=LABEL_FONTSIZE)
            ax.set_title(f"{criterion} distribution — {model}", fontsize=TITLE_FONTSIZE)
            ax.tick_params(axis="x", labelsize=TICK_FONTSIZE)
            ax.margins(x=0)

            leg = ax.legend(ncol=1, loc=LEGEND_LOC, bbox_to_anchor=LEGEND_BBOX,
                            borderaxespad=0.0, frameon=False)
            for t in leg.get_texts():
                t.set_fontsize(TICK_FONTSIZE)

            fig.subplots_adjust(**SUBPLOT_ADJUST)

            fname = f"{criterion.replace(' ', '_')}_{model}_stacked_bar.pdf"
            out_path = os.path.join(output_dir, fname)
            fig.savefig(out_path, bbox_inches="tight")
            plt.close(fig)

    """
    For each criterion × model, draw a horizontal stacked‐bar of counts
    and save one PDF per (criterion, model).
    """
    df = long_df.copy()
    df["Criterion_clean"] = df["Criterion"].str.strip()
    models = df["Model"].dropna().unique()

    for criterion, mapping in qualitative_mappings.items():
        for model in models:
            sub = df[(df["Criterion_clean"] == criterion) & (df["Model"] == model)]
            if sub.empty:
                continue

            keys   = sorted(mapping.keys(), reverse=True)
            labels = [mapping[k] for k in keys]
            counts = sub["Rating (num)"].value_counts().reindex(keys, fill_value=0)

            fig, ax = plt.subplots(figsize=(11, 2.8))
            left = 0
            for k, lbl in zip(keys, labels):
                cnt = counts.loc[k]
                ax.barh(0, cnt, left=left, height=0.6, label=lbl)
                if cnt > 0:
                    ax.text(left + cnt/2, 0, str(cnt),
                            va="center", ha="center",
                            color="white", fontweight="bold", fontsize=12)
                left += cnt

            ax.set_xlim(0, left)
            ax.set_ylim(-0.5, 0.5)
            ax.set_yticks([])
            ax.set_xlabel("Count", fontsize=14)
            ax.set_title(f"{criterion} distribution — {model}", fontsize=16)
            ax.tick_params(axis="x", labelsize=12)
            ax.margins(x=0)

            legend = ax.legend(
                ncol=1,
                loc="center left",
                bbox_to_anchor=(1.08, 0.5),  # not too close
                borderaxespad=0.0,
                frameon=False
            )
            for text in legend.get_texts():
                text.set_fontsize(12)

            fig.subplots_adjust(left=0.08, right=0.84, top=0.88, bottom=0.22)

            fname = f"{criterion.replace(' ', '_')}_{model}_stacked_bar.pdf"
            out_path = os.path.join(output_dir, fname)
            fig.savefig(out_path, bbox_inches="tight")
            plt.close(fig)

    """
    For each criterion × model, draw a horizontal stacked‐bar of counts
    and save one PDF per (criterion, model).
    """
    df = long_df.copy()
    df["Criterion_clean"] = df["Criterion"].str.strip()
    models = df["Model"].dropna().unique()

    for criterion, mapping in qualitative_mappings.items():
        for model in models:
            sub = df[(df["Criterion_clean"] == criterion) & (df["Model"] == model)]
            if sub.empty:
                continue

            keys   = sorted(mapping.keys(), reverse=True)
            labels = [mapping[k] for k in keys]
            counts = sub["Rating (num)"].value_counts().reindex(keys, fill_value=0)

            fig, ax = plt.subplots(figsize=(11, 2.8))
            left = 0
            for k, lbl in zip(keys, labels):
                cnt = counts.loc[k]
                ax.barh(0, cnt, left=left, height=0.6, label=lbl)
                if cnt > 0:
                    ax.text(left + cnt/2, 0, str(cnt),
                            va="center", ha="center",
                            color="white", fontweight="bold", fontsize=12)
                left += cnt

            ax.set_xlim(0, left)
            ax.set_ylim(-0.5, 0.5)
            ax.set_yticks([])
            ax.set_xlabel("Count", fontsize=14)
            ax.set_title(f"{criterion} distribution — {model}", fontsize=16)
            ax.tick_params(axis="x", labelsize=12)
            ax.margins(x=0)

            legend = ax.legend(
                ncol=1,
                loc="center left",
                bbox_to_anchor=(1.08, 0.5),  # not too close
                borderaxespad=0.0,
                frameon=False
            )
            for text in legend.get_texts():
                text.set_fontsize(12)

            fig.subplots_adjust(left=0.08, right=0.84, top=0.88, bottom=0.22)

            fname = f"{criterion.replace(' ', '_')}_{model}_stacked_bar.pdf"
            out_path = os.path.join(output_dir, fname)
            fig.savefig(out_path, bbox_inches="tight")
            plt.close(fig)

    """
    For each criterion × model, draw a horizontal stacked‐bar of counts
    and save one PDF per (criterion, model).
    """
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

            fig, ax = plt.subplots(figsize=(13.5, 2.2))
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
            ax.set_xlabel("Count",fontsize=14)
            ax.set_title(f"{criterion} distribution — {model}", fontsize=16)

            legend = ax.legend(
                ncol=1,
                loc="center left",
                bbox_to_anchor=(0.995, 0.5),
                borderaxespad=0.0,
                frameon=False
            )

            for text in legend.get_texts():
                text.set_fontsize(12)

            fig.subplots_adjust(left=0.06, right=0.90, top=0.88, bottom=0.22)

            fname = f"{criterion.replace(' ', '_')}_{model}_stacked_bar.pdf"
            out_path = os.path.join(output_dir, fname)
            fig.savefig(out_path, bbox_inches="tight")
            plt.close(fig)
            
def plot_token_scatter(token_df, output_dir, model_colors=None):
    """
    Scatter-plot Reg_tokens (x) vs. Gherkin_tokens (y), coloring by 'Model'.
    Adds a regression line for each model.
    """
    plt.figure(figsize=(8, 6))

    if model_colors is None:
        model_colors = {'Llama': 'blue', 'Claude': 'red'}

    for model, color in model_colors.items():
        sub = token_df[token_df['Model'] == model]
        if sub.empty:
            continue

        # Scatter plot
        plt.scatter(sub['Reg_tokens'], sub['Gherkin_tokens'],
                    alpha=0.7, label=model, color=color)

        # Regression line
        x = sub['Reg_tokens']
        y = sub['Gherkin_tokens']
        coeffs = np.polyfit(x, y, deg=1)
        x_vals = np.array([x.min(), x.max()])
        y_vals = coeffs[0] * x_vals + coeffs[1]
        plt.plot(x_vals, y_vals, linestyle='--', color=color, linewidth=2)

    plt.xlabel('Legal Provision Token Count', fontsize=19)
    plt.ylabel('Gherkin Specification Token Count', fontsize=19)
    # plt.title('Reg vs. Gherkin Token Counts by Model')
    plt.legend()
    plt.tight_layout()

    out_path = os.path.join(output_dir, 'token_scatter_colored.pdf')
    plt.savefig(out_path)
    plt.close()
    
def generate_token_step_boxplots(input_dir: str, output_dir: str):
    """
    Reads the following CSVs from input_dir:
      - token_lists.csv
      - gherkin_token_lists.csv
      - step_length_lists_words.csv
      - step_length_lists_chars.csv
      - step_length_lists_token.csv
      - steps_per_scenario_list.csv

    And generates five boxplots, saving them as PNGs into output_dir:
      1) Token counts (Reg vs Gherkin)
      2) Steps per scenario
      3) Step length (words)
      4) Step length (chars)
      5) Step length (token)
    """
    os.makedirs(output_dir, exist_ok=True)

    # common style for boxplots: bright blue fill, red '+' outliers,
    # and make the box itself very thin (width=0.4)
    bp_kwargs = dict(
        color="#18DEF8",
        flierprops=dict(
            marker='+',
            markerfacecolor='red',
            markeredgecolor='red',
            markersize=6
        )
    )
    mean_color = "#67045D"
    
    def annotate_mean(ax, x_pos, values):
        """Place a dark-green square at the mean, with its value printed next to it."""
        m = values.mean()
        ax.scatter(x_pos, m, marker='^', color=mean_color, zorder=3)
        ax.text(x_pos + 0.05, m, f'{m:.2f}', color=mean_color, ha='left', va='center', fontsize=9, fontweight='bold')

    median_color = "#FF8C00"   # orange

    def annotate_median(ax, x_pos, values):
            """Place an orange square at the median, with its value printed next to it."""
            med = values.median()
            ax.scatter(x_pos, med, marker='s', color=median_color, zorder=3)
            ax.text(x_pos + 0.05, med, f'{med:.0f}', 
                    color=median_color, ha='left', va='center',
                    fontsize=9, fontweight='bold')

    # --- 1) Token counts (Reg vs Gherkin) ---
    reg = pd.read_csv(os.path.join(input_dir, 'token_lists.csv'), header=None, names=['Count'])
    reg['Type'] = 'Reg_tokens'
    gherkin = pd.read_csv(os.path.join(input_dir, 'gherkin_token_lists.csv'), header=None, names=['Count'])
    gherkin['Type'] = 'Gherkin_tokens'
    tokens_df = pd.concat([reg, gherkin], ignore_index=True)

    fig, ax = plt.subplots(figsize=(3.5, 4))  # Slightly wider for spacing
    sns.boxplot(x='Type', y='Count', data=tokens_df, ax=ax, **bp_kwargs, width=0.1)

    annotate_mean(ax, 0, reg['Count'])
    annotate_mean(ax, 1, gherkin['Count'])
    # For the token-counts plot, after annotate_mean:
    # annotate_median(ax, 0, reg['Count'])
    # annotate_median(ax, 1, gherkin['Count'])

    # for i, (_, grp) in enumerate(tokens_df.groupby('Type')):
    #     annotate_mean(ax, i, grp['Count'])

    ax.set_title('Token Count\nDistribution', pad=12,fontsize=12)
    ax.set_xlabel('')
    # ax.set_ylabel('Token Count')
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Legal\nProvision', 'Gherkin\nSpecification'],fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(os.path.join(output_dir, 'token_counts_boxplot.pdf'))
    plt.close(fig)

    # --- 2) Steps per scenario ---
    steps = pd.read_csv(os.path.join(input_dir, 'steps_per_scenario_list.csv'),
                        header=None, names=['Steps'])

    fig, ax = plt.subplots(figsize=(3.5, 4))  
    sns.boxplot(y='Steps', data=steps, ax=ax, **bp_kwargs, width=0.1)

    annotate_mean(ax, 0, steps['Steps'])
    # For the steps-per-scenario plot:
    # annotate_median(ax, 0, steps['Steps'])

    ax.set_title('Number of Steps per\nScenario Distribution', pad=12, fontsize=12)
    # ax.set_ylabel('Number of Steps', fontsize=15)
    ax.set_xticks([])  # Hide x-axis ticks
    ax.set_xlim(-0.5, 0.4)  # shrink visible x-axis range
    ax.set_ylabel('')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(os.path.join(output_dir, 'steps_per_scenario_boxplot.pdf'))
    plt.close(fig)

    # --- 3) Step-length distributions (words, chars, token) ---
    for mode in ['words', 'chars', 'token']:
        fname = f'step_length_lists_{mode}.csv'
        lengths = pd.read_csv(os.path.join(input_dir, fname),
                              header=None, names=['Length'])
        lengths['X'] = ' '  # Use a single blank category

        fig, ax = plt.subplots(figsize=(3.5, 4))  # Narrower plot
        sns.boxplot(x='X', y='Length', data=lengths, ax=ax, **bp_kwargs, width=0.1)

        annotate_mean(ax, 0, lengths['Length'])
        # And in the loop over step-lengths:
        # annotate_median(ax, 0, lengths['Length'])

        ax.set_title(f'Step Length\nDistribution ({mode.capitalize()})', pad=12,fontsize=15)
        ax.set_ylabel('')
        ax.set_xlabel('')
        ax.set_xticks([])  # Hide x-axis tick
        ax.set_xlim(-0.5, 0.4)  # Shrink visible width like module 2
        plt.tight_layout(rect=[0, 0, 1, 0.95])

        fig.savefig(os.path.join(output_dir, f'step_length_{mode}_boxplot.pdf'))
        plt.close(fig)
    # Finished generating token/step boxplots — return early to avoid
    # executing accidental duplicated code that appears later in this file
    # (that re-imports modules and references variables not in scope).
    return

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

            fig, ax = plt.subplots(figsize=(16, 4))  # Landscape, wide
            left = 0
            for k, lbl in zip(keys, labels):
                cnt = counts.loc[k]
                ax.barh(0, cnt, left=left, height=0.8, label=lbl)
                if cnt > 0:
                    ax.text(left + cnt / 2, 0, str(cnt),
                            va="center", ha="center",
                            color="white", fontweight="bold", fontsize=24)
                left += cnt

            ax.set_xlim(0, left)
            ax.set_ylim(-0.7, 0.7)
            ax.set_yticks([])
            ax.set_xlabel(" ", fontsize=22, fontweight="bold")
            ax.set_title(f"{criterion} Distribution ({model})", fontsize=28, pad=20)

            ax.legend(
                ncol=1,
                loc="center left",
                bbox_to_anchor=(1.01, 0.5),
                borderaxespad=0,
                fontsize=20,
                frameon=False,
                handlelength=1.0,  # Make legend color boxes longer
                handleheight=1.0   # Make legend color boxes taller

            )

            fig.subplots_adjust(left=0.06, right=0.85, top=0.82, bottom=0.22)
            fname = f"{criterion.replace(' ', '_')}_{model}_stacked_bar.pdf"
            out_path = os.path.join(output_dir, fname)
            fig.savefig(out_path, bbox_inches="tight")
            plt.close(fig)
    """
    For each criterion × model, draw a horizontal stacked bar with the same
    dimensions and styling as the aggregated plot above.
    """
    df = long_df.copy()
    df["Criterion_clean"] = df["Criterion"].str.strip()
    models = df["Model"].dropna().unique()

    # Get total counts per criterion to maintain consistent width scaling
    totals = df.groupby("Criterion_clean").size().to_dict()

    for criterion, mapping in qualitative_mappings.items():
        for model in models:
            sub = df[(df["Criterion_clean"] == criterion) & (df["Model"] == model)]
            if sub.empty:
                continue

            keys = sorted(mapping.keys(), reverse=True)
            labels = [mapping[k] for k in keys]
            counts = sub["Rating (num)"].value_counts().reindex(keys, fill_value=0)

            fig, ax = plt.subplots(figsize=(16, 4))  # Same size as above
            left = 0
            for k, lbl in zip(keys, labels):
                cnt = counts.loc[k]
                ax.barh(0, cnt, left=left, height=0.8, label=lbl)  # Same bar height
                if cnt > 0:
                    ax.text(left + cnt / 2, 0, str(cnt),
                            va="center", ha="center",
                            color="white", fontweight="bold", fontsize=24)  # Same text size
                left += cnt

            # Use same width as the all-models plot for consistency
            ax.set_xlim(0, totals.get(criterion, left))
            ax.set_ylim(-0.7, 0.7)
            ax.set_yticks([])
            ax.set_xlabel(" ", fontsize=22, fontweight="bold")
            ax.set_title(f"{criterion} Distribution ({model})", fontsize=28, pad=20)

            # Same legend formatting
            ax.legend(
                ncol=1,
                loc="center left",
                bbox_to_anchor=(1.01, 0.5),
                borderaxespad=0,
                fontsize=20,
                frameon=False,
                handlelength=1.0,
                handleheight=1.0
            )

            fig.subplots_adjust(left=0.06, right=0.85, top=0.82, bottom=0.22)
            fname = f"{criterion.replace(' ', '_')}_{model}_stacked_bar.pdf"
            out_path = os.path.join(output_dir, fname)
            fig.savefig(out_path, bbox_inches="tight")
            plt.close(fig)
