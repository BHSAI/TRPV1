"""
Visualization utilities for TRPV1 ML benchmark.

Common plotting functions for:
- Heatmaps
- Box plots
- Bar charts
- Dashboard layouts
- SHAP visualizations
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from pathlib import Path

# ============================================================================
# Heatmap Utilities
# ============================================================================

def plot_heatmap(data, title="Heatmap", xlabel=None, ylabel=None,
                 cmap="RdYlGn", annot=True, fmt=".3f", figsize=(10, 8),
                 vmin=None, vmax=None, cbar_label=None, save_path=None):
    """
    Create a heatmap visualization.

    Args:
        data: 2D array or DataFrame
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        cmap: Colormap name
        annot: Whether to annotate cells
        fmt: Format string for annotations
        figsize: Figure size tuple
        vmin: Minimum value for colorbar
        vmax: Maximum value for colorbar
        cbar_label: Colorbar label
        save_path: Path to save figure (None = don't save)

    Returns:
        Figure and axes objects
    """
    fig, ax = plt.subplots(figsize=figsize)

    sns.heatmap(
        data,
        annot=annot,
        fmt=fmt,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        cbar_kws={'label': cbar_label} if cbar_label else None,
        ax=ax
    )

    ax.set_title(title, fontsize=14, fontweight='bold')
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=12)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=12)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig, ax


def plot_pvalue_heatmap(pvalue_matrix, title="P-value Heatmap",
                        alpha_levels=[0.001, 0.01, 0.05, 1.0],
                        colors=["#00441b", "#238b45", "#99d8c9", "#fee0d2"],
                        figsize=(10, 8), save_path=None):
    """
    Create p-value heatmap with custom binning.

    Args:
        pvalue_matrix: Symmetric matrix of p-values
        title: Plot title
        alpha_levels: Significance level boundaries
        colors: Colors for each bin
        figsize: Figure size
        save_path: Path to save figure

    Returns:
        Figure and axes objects
    """
    cmap = mpl.colors.ListedColormap(colors)
    norm = mpl.colors.BoundaryNorm(alpha_levels, ncolors=cmap.N)

    fig, ax = plt.subplots(figsize=figsize)

    im = ax.imshow(pvalue_matrix.values, cmap=cmap, norm=norm, aspect='auto')

    # Set ticks
    ax.set_xticks(range(len(pvalue_matrix.columns)))
    ax.set_yticks(range(len(pvalue_matrix.index)))
    ax.set_xticklabels(pvalue_matrix.columns, rotation=45, ha='right')
    ax.set_yticklabels(pvalue_matrix.index)

    # Annotate cells
    for i in range(len(pvalue_matrix.index)):
        for j in range(len(pvalue_matrix.columns)):
            val = pvalue_matrix.iloc[i, j]
            text_color = 'white' if val < 0.05 else 'black'
            ax.text(j, i, f'{val:.3f}', ha='center', va='center',
                   color=text_color, fontsize=9)

    ax.set_title(title, fontsize=14, fontweight='bold')

    # Colorbar
    cbar = plt.colorbar(im, ax=ax, boundaries=alpha_levels, ticks=alpha_levels)
    cbar.set_label('p-value', fontsize=12)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig, ax


# ============================================================================
# Box Plot Utilities
# ============================================================================

def plot_boxplots(data, x, y, hue=None, title="Box Plot",
                 xlabel=None, ylabel=None, figsize=(12, 6),
                 palette="Set2", order=None, save_path=None):
    """
    Create box plot visualization.

    Args:
        data: DataFrame
        x: Column name for x-axis
        y: Column name for y-axis
        hue: Column name for grouping
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        figsize: Figure size
        palette: Color palette
        order: Order for x-axis categories
        save_path: Path to save figure

    Returns:
        Figure and axes objects
    """
    fig, ax = plt.subplots(figsize=figsize)

    sns.boxplot(
        data=data,
        x=x,
        y=y,
        hue=hue,
        palette=palette,
        order=order,
        ax=ax
    )

    ax.set_title(title, fontsize=14, fontweight='bold')
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=12)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=12)

    plt.xticks(rotation=45, ha='right')
    plt.legend(title=hue if hue else None)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig, ax


# ============================================================================
# Bar Chart Utilities
# ============================================================================

def plot_bar_chart(data, x, y, title="Bar Chart", xlabel=None, ylabel=None,
                  color="steelblue", figsize=(10, 6), save_path=None):
    """
    Create bar chart visualization.

    Args:
        data: DataFrame or dict
        x: X-axis values or column name
        y: Y-axis values or column name
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        color: Bar color
        figsize: Figure size
        save_path: Path to save figure

    Returns:
        Figure and axes objects
    """
    fig, ax = plt.subplots(figsize=figsize)

    if isinstance(data, pd.DataFrame):
        ax.bar(data[x], data[y], color=color)
    else:
        ax.bar(x, y, color=color)

    ax.set_title(title, fontsize=14, fontweight='bold')
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=12)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=12)

    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig, ax


# ============================================================================
# Multi-panel Dashboard
# ============================================================================

def create_dashboard(plot_functions, nrows=2, ncols=2, figsize=(15, 12),
                    title=None, save_path=None):
    """
    Create multi-panel dashboard.

    Args:
        plot_functions: List of functions that create plots, each takes (ax) as argument
        nrows: Number of rows
        ncols: Number of columns
        figsize: Figure size
        title: Overall title
        save_path: Path to save figure

    Returns:
        Figure and axes array
    """
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    axes = axes.flatten() if nrows * ncols > 1 else [axes]

    for i, plot_func in enumerate(plot_functions):
        if i < len(axes):
            plot_func(axes[i])

    # Hide unused subplots
    for i in range(len(plot_functions), len(axes)):
        axes[i].axis('off')

    if title:
        fig.suptitle(title, fontsize=16, fontweight='bold', y=0.995)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig, axes


# ============================================================================
# Style Configuration
# ============================================================================

def set_publication_style():
    """Set matplotlib style for publication-quality figures."""
    plt.style.use('seaborn-v0_8-paper')
    plt.rcParams.update({
        'font.size': 11,
        'axes.labelsize': 12,
        'axes.titlesize': 14,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.titlesize': 16,
        'figure.dpi': 100,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'axes.grid': True,
        'grid.alpha': 0.3,
    })


def reset_style():
    """Reset matplotlib to default style."""
    mpl.rcParams.update(mpl.rcParamsDefault)
