"""
Graphical visualization of relationship tensors using matplotlib.
"""

import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from typing import Optional

from context_tracker.model.module.relationship_tensor import parse_latex, ParsedLatex, RelationType


def plot_relationship_tensor(
    parsed: ParsedLatex,
    save_path: Optional[str] = None,
    figsize: tuple = (14, 10),
    title: Optional[str] = None
):
    """
    Create a comprehensive visualization of the relationship tensor.
    
    Includes:
    1. Token list with indices
    2. Tree structure
    3. Tensor heatmaps for each relation type
    """
    tokens = parsed.tokens
    relations = parsed.relations
    n = len(tokens)
    
    if n == 0:
        print("No tokens to visualize")
        return
    
    # Create figure with subplots
    fig = plt.figure(figsize=figsize)
    fig.suptitle(title or "Relationship Tensor Visualization", fontsize=14, fontweight='bold')
    
    # Determine active relation types
    active_rels = set()
    for _, _, rel in relations:
        active_rels.add(rel)
    active_rels = sorted(active_rels)
    
    num_active = len(active_rels)
    
    # Layout: left side for tree, right side for tensor slices
    gs = fig.add_gridspec(2, max(num_active, 2) + 1, width_ratios=[1.5] + [1] * max(num_active, 2))
    
    # --- Left panel: Token list and relations ---
    ax_info = fig.add_subplot(gs[:, 0])
    ax_info.axis('off')
    
    info_text = "TOKENS:\n"
    info_text += "-" * 20 + "\n"
    for i, tok in enumerate(tokens):
        info_text += f"[{i}] {tok}\n"
    
    info_text += "\nRELATIONS:\n"
    info_text += "-" * 20 + "\n"
    for child_idx, parent_idx, rel_type in relations:
        child_tok = tokens[child_idx]
        parent_tok = tokens[parent_idx]
        info_text += f"{child_tok} -> {parent_tok}\n  ({RelationType(rel_type).name})\n"
    
    ax_info.text(0.05, 0.95, info_text, transform=ax_info.transAxes,
                 fontsize=10, verticalalignment='top', fontfamily='monospace',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax_info.set_title("Structure", fontweight='bold')
    
    # --- Right panels: Tensor slices ---
    if parsed.tensor is not None:
        tensor = parsed.tensor.numpy()
        
        # Color map
        cmap = plt.cm.Blues
        
        for idx, rel_type in enumerate(active_rels):
            row = idx // max(num_active, 2)
            col = 1 + (idx % max(num_active, 2))
            ax = fig.add_subplot(gs[row, col])
            
            slice_data = tensor[:, :, rel_type]
            
            # Plot heatmap
            im = ax.imshow(slice_data, cmap=cmap, vmin=0, vmax=1, aspect='equal')
            
            # Labels
            ax.set_xticks(range(n))
            ax.set_yticks(range(n))
            ax.set_xticklabels([t[:4] for t in tokens], rotation=45, ha='right', fontsize=8)
            ax.set_yticklabels([t[:4] for t in tokens], fontsize=8)
            
            ax.set_xlabel("Parent", fontsize=9)
            ax.set_ylabel("Child", fontsize=9)
            ax.set_title(f"{RelationType(rel_type).name}", fontweight='bold', fontsize=11)
            
            # Add X markers for non-zero entries
            for i in range(n):
                for j in range(n):
                    if slice_data[i, j] > 0:
                        ax.text(j, i, 'X', ha='center', va='center', 
                               color='red', fontweight='bold', fontsize=12)
            
            # Grid
            ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
            ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
            ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.5)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"Saved to {save_path}")
    
    plt.close()
    return fig


def plot_tree_structure(
    parsed: ParsedLatex,
    save_path: Optional[str] = None,
    figsize: tuple = (12, 8),
    title: Optional[str] = None
):
    """
    Visualize the parsed LaTeX as a tree structure.
    """
    tokens = parsed.tokens
    relations = parsed.relations
    n = len(tokens)
    
    if n == 0:
        print("No tokens to visualize")
        return
    
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_title(title or "LaTeX Tree Structure", fontsize=14, fontweight='bold')
    
    # Build parent-child relationships
    children_map = {}  # parent_idx -> [(child_idx, rel_type), ...]
    has_parent = set()
    
    for child_idx, parent_idx, rel_type in relations:
        has_parent.add(child_idx)
        if parent_idx not in children_map:
            children_map[parent_idx] = []
        children_map[parent_idx].append((child_idx, rel_type))
    
    # Find roots
    roots = [i for i in range(n) if i not in has_parent]
    
    # Calculate positions using a simple layout algorithm
    positions = {}
    
    def layout_subtree(node_idx: int, x: float, y: float, width: float):
        positions[node_idx] = (x, y)
        
        children = children_map.get(node_idx, [])
        if not children:
            return
        
        # Sort children by relation type
        children.sort(key=lambda c: (c[1], c[0]))
        
        # Distribute children horizontally
        child_width = width / len(children)
        for i, (child_idx, _) in enumerate(children):
            child_x = x - width/2 + child_width * (i + 0.5)
            layout_subtree(child_idx, child_x, y - 1.5, child_width)
    
    # Layout each root
    total_width = n * 2
    root_width = total_width / len(roots)
    for i, root_idx in enumerate(roots):
        root_x = -total_width/2 + root_width * (i + 0.5)
        layout_subtree(root_idx, root_x, 0, root_width)
    
    # Relation type colors
    rel_colors = {
        RelationType.RIGHT: '#2196F3',   # Blue
        RelationType.SUP: '#4CAF50',     # Green
        RelationType.SUB: '#FF9800',     # Orange
        RelationType.ABOVE: '#9C27B0',   # Purple
        RelationType.BELOW: '#E91E63',   # Pink
        RelationType.INSIDE: '#795548',  # Brown
    }
    
    # Draw edges
    for parent_idx, children in children_map.items():
        px, py = positions[parent_idx]
        for child_idx, rel_type in children:
            cx, cy = positions[child_idx]
            color = rel_colors.get(RelationType(rel_type), 'gray')
            ax.annotate('', xy=(cx, cy + 0.3), xytext=(px, py - 0.3),
                       arrowprops=dict(arrowstyle='->', color=color, lw=2))
            
            # Edge label
            mid_x, mid_y = (px + cx) / 2, (py + cy) / 2
            ax.text(mid_x + 0.1, mid_y, RelationType(rel_type).name, 
                   fontsize=8, color=color, style='italic')
    
    # Draw nodes
    for idx, (x, y) in positions.items():
        token = tokens[idx]
        
        # Node circle
        circle = plt.Circle((x, y), 0.35, facecolor='lightblue', 
                            edgecolor='black', linewidth=2, zorder=3)
        ax.add_patch(circle)
        
        # Token text
        ax.text(x, y, token, ha='center', va='center', fontsize=10, 
               fontweight='bold', zorder=4)
        
        # Index
        ax.text(x + 0.4, y + 0.3, f'[{idx}]', fontsize=7, color='gray')
    
    # Legend
    legend_patches = []
    for rel_type, color in rel_colors.items():
        if any(r == rel_type for _, _, r in relations):
            patch = mpatches.Patch(color=color, label=rel_type.name)
            legend_patches.append(patch)
    
    if legend_patches:
        ax.legend(handles=legend_patches, loc='upper right', fontsize=9)
    
    # Adjust view
    if positions:
        xs = [p[0] for p in positions.values()]
        ys = [p[1] for p in positions.values()]
        margin = 1.5
        ax.set_xlim(min(xs) - margin, max(xs) + margin)
        ax.set_ylim(min(ys) - margin, max(ys) + margin)
    
    ax.set_aspect('equal')
    ax.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"Saved to {save_path}")
    
    plt.close()
    return fig


def plot_action_space(
    parsed: ParsedLatex,
    save_path: Optional[str] = None,
    figsize: tuple = (14, 8),
    title: Optional[str] = None
):
    """
    Visualize the action space (all possible parent-relation combinations).
    """
    tokens = parsed.tokens
    n = len(tokens)
    num_rels = RelationType.num_types()
    
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_title(title or "Action Space: (Parent, Relation) Pairs", fontsize=14, fontweight='bold')
    
    # Create grid
    action_matrix = np.zeros((n, num_rels))
    
    # All actions are valid (for visualization)
    action_matrix[:, :] = 0.3  # Light background for all possible actions
    
    # Mark existing relations
    for child_idx, parent_idx, rel_type in parsed.relations:
        action_matrix[parent_idx, rel_type] = 1.0  # Dark for existing
    
    # Plot
    im = ax.imshow(action_matrix.T, cmap='YlOrRd', vmin=0, vmax=1, aspect='auto')
    
    # Labels
    ax.set_xticks(range(n))
    ax.set_yticks(range(num_rels))
    ax.set_xticklabels(tokens, rotation=45, ha='right', fontsize=10)
    ax.set_yticklabels(RelationType.names(), fontsize=10)
    
    ax.set_xlabel("Parent Token", fontsize=12)
    ax.set_ylabel("Relation Type", fontsize=12)
    
    # Grid
    ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, num_rels, 1), minor=True)
    ax.grid(which='minor', color='white', linestyle='-', linewidth=2)
    
    # Action indices
    for i in range(n):
        for j in range(num_rels):
            action_idx = i * num_rels + j
            ax.text(i, j, f'{action_idx}', ha='center', va='center', 
                   fontsize=7, color='black' if action_matrix[i, j] < 0.5 else 'white')
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.6)
    cbar.set_label('Action Status\n(0.3=possible, 1.0=existing)', fontsize=9)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"Saved to {save_path}")
    
    plt.close()
    return fig


def visualize_latex(latex: str, output_dir: str = "relationship_viz"):
    """
    Generate all visualizations for a LaTeX expression.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    parsed = parse_latex(latex)
    
    # Safe filename
    safe_name = latex.replace('\\', '_').replace('{', '').replace('}', '')
    safe_name = ''.join(c if c.isalnum() or c in '+-_' else '_' for c in safe_name)[:30]
    
    # Generate visualizations
    plot_relationship_tensor(
        parsed, 
        save_path=os.path.join(output_dir, f"{safe_name}_tensor.png"),
        title=f"Relationship Tensor: {latex}"
    )
    
    plot_tree_structure(
        parsed,
        save_path=os.path.join(output_dir, f"{safe_name}_tree.png"),
        title=f"Tree Structure: {latex}"
    )
    
    plot_action_space(
        parsed,
        save_path=os.path.join(output_dir, f"{safe_name}_actions.png"),
        title=f"Action Space: {latex}"
    )
    
    print(f"Generated visualizations for: {latex}")
    return parsed


if __name__ == "__main__":
    # Test cases
    test_cases = [
        r"a_t",
        r"\frac{a}{b}",
        r"\frac{a_t+b}{c}",
        r"x^2 + y^2 = z^2",
    ]
    
    # Output to same directory as this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, "relationship_viz")
    
    for latex in test_cases:
        visualize_latex(latex, output_dir)
    
    print(f"\nAll visualizations saved to: {output_dir}")

