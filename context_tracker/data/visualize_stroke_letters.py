"""
Visualize Stroke-Letter Pairs with Line Images

This script creates visualizations showing:
1. Original line image (from lineImages)
2. Rendered strokes with color-coded letters
3. Legend showing stroke -> letter mapping
"""

import os
import xml.etree.ElementTree as ET
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import hsv_to_rgb
import colorsys
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass


# ============================================================
# Data Classes (same as build script)
# ============================================================

@dataclass
class Point:
    x: float
    y: float
    time: float


@dataclass 
class Stroke:
    points: List[Point]
    start_time: float
    end_time: float
    
    @property
    def center_x(self) -> float:
        return np.mean([p.x for p in self.points]) if self.points else 0
    
    @property
    def min_x(self) -> float:
        return min(p.x for p in self.points) if self.points else 0
    
    @property
    def max_x(self) -> float:
        return max(p.x for p in self.points) if self.points else 0


# ============================================================
# Parser
# ============================================================

class IAMOnDBParser:
    def __init__(self, data_root: str):
        self.data_root = Path(data_root)
        self.line_strokes_dir = self.data_root / "lineStrokes-all" / "lineStrokes"
        self.line_images_dir = self.data_root / "lineImages-all" / "lineImages"
        self.ascii_dir = self.data_root / "ascii-all" / "ascii"
    
    def parse_stroke_xml(self, xml_path: str) -> List[Stroke]:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        strokes = []
        for stroke_elem in root.findall('.//Stroke'):
            points = []
            for point_elem in stroke_elem.findall('Point'):
                points.append(Point(
                    x=float(point_elem.get('x', 0)),
                    y=float(point_elem.get('y', 0)),
                    time=float(point_elem.get('time', 0))
                ))
            
            if points:
                strokes.append(Stroke(
                    points=points,
                    start_time=float(stroke_elem.get('start_time', 0)),
                    end_time=float(stroke_elem.get('end_time', 0))
                ))
        
        return strokes
    
    def get_line_image_path(self, line_id: str) -> Optional[Path]:
        """Get path to line image file"""
        parts = line_id.split('-')
        group = parts[0]
        form_base = parts[0] + '-' + ''.join(c for c in parts[1] if c.isdigit())
        
        img_path = self.line_images_dir / group / form_base / f"{line_id}.tif"
        if img_path.exists():
            return img_path
        
        # Try alternative structure
        form_with_variant = parts[0] + '-' + parts[1]
        img_path = self.line_images_dir / group / form_with_variant / f"{line_id}.tif"
        if img_path.exists():
            return img_path
        
        return None
    
    def get_stroke_xml_path(self, line_id: str) -> Optional[Path]:
        """Get path to stroke XML file"""
        parts = line_id.split('-')
        group = parts[0]
        form_base = parts[0] + '-' + ''.join(c for c in parts[1] if c.isdigit())
        
        xml_path = self.line_strokes_dir / group / form_base / f"{line_id}.xml"
        if xml_path.exists():
            return xml_path
        
        form_with_variant = parts[0] + '-' + parts[1]
        xml_path = self.line_strokes_dir / group / form_with_variant / f"{line_id}.xml"
        if xml_path.exists():
            return xml_path
        
        return None
    
    def get_line_text(self, line_id: str) -> Optional[str]:
        parts = line_id.split('-')
        if len(parts) < 3:
            return None
        
        group = parts[0]
        form_with_variant = parts[1]
        line_num = int(parts[2]) - 1
        
        form_id = f"{group}-{form_with_variant}"
        form_base = ''.join(c for c in form_with_variant if c.isdigit())
        form_dir = f"{group}-{form_base}"
        
        ascii_path = self.ascii_dir / group / form_dir / f"{form_id}.txt"
        
        if not ascii_path.exists():
            return None
        
        with open(ascii_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        if 'CSR:' in content:
            text = content.split('CSR:')[1].strip()
        else:
            text = content.replace('OCR:', '').strip()
        
        lines = [l.strip() for l in text.split('\n') if l.strip()]
        
        if 0 <= line_num < len(lines):
            return lines[line_num]
        
        return None
    
    def iter_line_files(self, max_count=None):
        """Iterate through line files that have both strokes and images"""
        count = 0
        for group_dir in sorted(self.line_strokes_dir.iterdir()):
            if not group_dir.is_dir():
                continue
            
            for form_dir in sorted(group_dir.iterdir()):
                if not form_dir.is_dir():
                    continue
                
                for xml_file in sorted(form_dir.glob("*.xml")):
                    if max_count and count >= max_count:
                        return
                    
                    line_id = xml_file.stem
                    img_path = self.get_line_image_path(line_id)
                    
                    if img_path:
                        yield line_id, str(xml_file), str(img_path)
                        count += 1


# ============================================================
# Stroke to Letter Mapper (simplified)
# ============================================================

class StrokeToLetterMapper:
    def __init__(self, char_gap_threshold=0.08, word_gap_threshold=0.15):
        self.char_gap_threshold = char_gap_threshold
        self.word_gap_threshold = word_gap_threshold
    
    def map_strokes_to_letters(self, strokes: List[Stroke], text: str) -> List[Tuple[int, str]]:
        """Returns list of (stroke_index, letter) pairs"""
        if not strokes or not text:
            return []
        
        text = ' '.join(text.split())
        non_space_chars = [c for c in text if c != ' ']
        
        if not non_space_chars:
            return []
        
        # Sort strokes by x position
        sorted_indices = sorted(range(len(strokes)), key=lambda i: strokes[i].center_x)
        
        # Calculate line width
        all_xs = [p.x for s in strokes for p in s.points]
        line_width = max(all_xs) - min(all_xs) if all_xs else 1
        
        char_gap = line_width * self.char_gap_threshold
        word_gap = line_width * self.word_gap_threshold
        
        # Group strokes
        groups = []
        current_group = [sorted_indices[0]]
        
        for i in range(1, len(sorted_indices)):
            prev_stroke = strokes[sorted_indices[i-1]]
            curr_stroke = strokes[sorted_indices[i]]
            
            gap = curr_stroke.min_x - prev_stroke.max_x
            
            if gap > word_gap or gap > char_gap:
                groups.append(current_group)
                current_group = [sorted_indices[i]]
            else:
                current_group.append(sorted_indices[i])
        
        groups.append(current_group)
        
        # Map groups to characters
        result = {}
        chars_per_group = len(non_space_chars) / len(groups) if groups else 1
        
        for group_idx, group in enumerate(groups):
            char_idx = min(int(group_idx * chars_per_group), len(non_space_chars) - 1)
            letter = non_space_chars[char_idx]
            
            for stroke_idx in group:
                result[stroke_idx] = letter
        
        return [(i, result.get(i, '?')) for i in range(len(strokes))]


# ============================================================
# Visualizer
# ============================================================

class StrokeLetterVisualizer:
    def __init__(self, data_root: str, output_dir: str = None):
        self.parser = IAMOnDBParser(data_root)
        self.mapper = StrokeToLetterMapper()
        self.output_dir = Path(output_dir) if output_dir else Path("visualizations")
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def get_distinct_colors(self, n: int) -> List[Tuple[float, float, float]]:
        """Generate n visually distinct colors"""
        colors = []
        for i in range(n):
            hue = i / n
            saturation = 0.7 + 0.3 * (i % 2)  # Alternate saturation
            value = 0.8 + 0.2 * ((i // 2) % 2)  # Alternate brightness
            rgb = colorsys.hsv_to_rgb(hue, saturation, value)
            colors.append(rgb)
        return colors
    
    def render_strokes_colored(self, strokes: List[Stroke], 
                                stroke_letters: List[Tuple[int, str]],
                                figsize: Tuple[int, int] = (12, 4)) -> plt.Figure:
        """Render strokes with colors based on letters"""
        
        # Get unique letters and assign colors
        unique_letters = sorted(set(letter for _, letter in stroke_letters))
        colors = self.get_distinct_colors(len(unique_letters))
        letter_to_color = {letter: colors[i] for i, letter in enumerate(unique_letters)}
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Get all points for bounds
        all_x = [p.x for s in strokes for p in s.points]
        all_y = [p.y for s in strokes for p in s.points]
        
        if not all_x:
            return fig
        
        # Draw each stroke
        for stroke_idx, letter in stroke_letters:
            stroke = strokes[stroke_idx]
            color = letter_to_color.get(letter, (0.5, 0.5, 0.5))
            
            xs = [p.x for p in stroke.points]
            ys = [p.y for p in stroke.points]
            
            # Plot stroke
            ax.plot(xs, ys, color=color, linewidth=2, alpha=0.8)
            
            # Add small marker at start of stroke
            if xs:
                ax.scatter([xs[0]], [ys[0]], color=color, s=20, zorder=5)
        
        # Invert y-axis (image coordinates)
        ax.invert_yaxis()
        ax.set_aspect('equal')
        ax.axis('off')
        
        # Create legend
        patches = [mpatches.Patch(color=letter_to_color[l], label=f"'{l}'") 
                   for l in unique_letters]
        ax.legend(handles=patches, loc='upper right', fontsize=8, ncol=min(len(unique_letters), 10))
        
        plt.tight_layout()
        return fig
    
    def visualize_line(self, line_id: str, save: bool = True, show: bool = True) -> Optional[plt.Figure]:
        """
        Create visualization for a specific line showing:
        - Original line image
        - Strokes colored by letter assignment
        - Text transcription
        """
        # Get paths
        xml_path = self.parser.get_stroke_xml_path(line_id)
        img_path = self.parser.get_line_image_path(line_id)
        
        if not xml_path:
            print(f"Stroke XML not found for {line_id}")
            return None
        
        # Parse data
        strokes = self.parser.parse_stroke_xml(str(xml_path))
        text = self.parser.get_line_text(line_id)
        
        if not strokes:
            print(f"No strokes found for {line_id}")
            return None
        
        if not text:
            text = "[Text not available]"
        
        # Map strokes to letters
        stroke_letters = self.mapper.map_strokes_to_letters(strokes, text)
        
        # Create figure
        has_image = img_path and Path(img_path).exists()
        n_rows = 3 if has_image else 2
        
        fig = plt.figure(figsize=(14, 3 * n_rows))
        
        row = 1
        
        # Row 1: Original image (if available)
        if has_image:
            ax1 = fig.add_subplot(n_rows, 1, row)
            try:
                img = Image.open(img_path)
                ax1.imshow(img, cmap='gray')
                ax1.set_title(f"Original Line Image: {line_id}", fontsize=12, fontweight='bold')
            except Exception as e:
                ax1.text(0.5, 0.5, f"Could not load image: {e}", ha='center', va='center')
                ax1.set_title("Original Line Image (Error)")
            ax1.axis('off')
            row += 1
        
        # Row 2: Colored strokes
        ax2 = fig.add_subplot(n_rows, 1, row)
        
        # Get unique letters and assign colors
        unique_letters = sorted(set(letter for _, letter in stroke_letters))
        colors = self.get_distinct_colors(len(unique_letters))
        letter_to_color = {letter: colors[i] for i, letter in enumerate(unique_letters)}
        
        # Draw strokes
        for stroke_idx, letter in stroke_letters:
            stroke = strokes[stroke_idx]
            color = letter_to_color.get(letter, (0.5, 0.5, 0.5))
            
            xs = [p.x for p in stroke.points]
            ys = [p.y for p in stroke.points]
            
            ax2.plot(xs, ys, color=color, linewidth=2.5, alpha=0.85)
            ax2.scatter([xs[0]], [ys[0]], color=color, s=30, zorder=5, edgecolor='black', linewidth=0.5)
        
        ax2.invert_yaxis()
        ax2.set_aspect('equal')
        ax2.axis('off')
        ax2.set_title("Strokes Colored by Letter Assignment", fontsize=12, fontweight='bold')
        
        # Add legend
        patches = [mpatches.Patch(color=letter_to_color[l], label=f"'{l}'") 
                   for l in unique_letters]
        ax2.legend(handles=patches, loc='upper right', fontsize=7, 
                   ncol=min(len(unique_letters), 12), framealpha=0.9)
        row += 1
        
        # Row 3: Text and stroke details
        ax3 = fig.add_subplot(n_rows, 1, row)
        ax3.axis('off')
        
        # Build info text
        info_lines = [
            f"Transcription: \"{text}\"",
            f"",
            f"Total Strokes: {len(strokes)} | Unique Letters: {len(unique_letters)}",
            f"",
            "Stroke -> Letter Mapping (first 20):"
        ]
        
        # Show first 20 mappings
        mapping_strs = []
        for i, (stroke_idx, letter) in enumerate(stroke_letters[:20]):
            n_points = len(strokes[stroke_idx].points)
            mapping_strs.append(f"S{stroke_idx}({n_points}pts)->'{letter}'")
        
        # Format in rows of 5
        for i in range(0, len(mapping_strs), 5):
            info_lines.append("  " + "  |  ".join(mapping_strs[i:i+5]))
        
        if len(stroke_letters) > 20:
            info_lines.append(f"  ... and {len(stroke_letters) - 20} more strokes")
        
        info_text = "\n".join(info_lines)
        ax3.text(0.02, 0.95, info_text, transform=ax3.transAxes, fontsize=9,
                 verticalalignment='top', fontfamily='monospace',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        ax3.set_title("Transcription & Mapping Details", fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        
        if save:
            save_path = self.output_dir / f"{line_id}_visualization.png"
            fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
            print(f"Saved: {save_path}")
        
        if show:
            plt.show()
        
        return fig
    
    def visualize_multiple(self, n_samples: int = 5, save: bool = True, show: bool = True):
        """Visualize multiple random samples"""
        line_files = list(self.parser.iter_line_files(max_count=100))
        
        if not line_files:
            print("No line files found!")
            return
        
        # Select random samples
        np.random.seed(42)
        indices = np.random.choice(len(line_files), min(n_samples, len(line_files)), replace=False)
        
        for idx in indices:
            line_id, xml_path, img_path = line_files[idx]
            print(f"\n{'='*60}")
            print(f"Visualizing: {line_id}")
            print(f"{'='*60}")
            
            self.visualize_line(line_id, save=save, show=show)
    
    def create_grid_visualization(self, n_samples: int = 6, save: bool = True):
        """Create a grid showing multiple line samples"""
        line_files = list(self.parser.iter_line_files(max_count=50))
        
        if not line_files:
            print("No line files found!")
            return
        
        # Select samples
        np.random.seed(123)
        indices = np.random.choice(len(line_files), min(n_samples, len(line_files)), replace=False)
        
        # Create grid
        n_cols = 2
        n_rows = (n_samples + 1) // 2
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4 * n_rows))
        axes = axes.flatten() if n_samples > 2 else [axes] if n_samples == 1 else axes
        
        for i, idx in enumerate(indices):
            line_id, xml_path, img_path = line_files[idx]
            ax = axes[i]
            
            # Parse
            strokes = self.parser.parse_stroke_xml(xml_path)
            text = self.parser.get_line_text(line_id)
            
            if not strokes:
                ax.text(0.5, 0.5, "No strokes", ha='center', va='center')
                ax.set_title(line_id)
                continue
            
            # Map
            stroke_letters = self.mapper.map_strokes_to_letters(strokes, text or "")
            
            # Colors
            unique_letters = sorted(set(letter for _, letter in stroke_letters))
            colors = self.get_distinct_colors(len(unique_letters))
            letter_to_color = {letter: colors[j] for j, letter in enumerate(unique_letters)}
            
            # Draw
            for stroke_idx, letter in stroke_letters:
                stroke = strokes[stroke_idx]
                color = letter_to_color.get(letter, (0.5, 0.5, 0.5))
                
                xs = [p.x for p in stroke.points]
                ys = [p.y for p in stroke.points]
                
                ax.plot(xs, ys, color=color, linewidth=2, alpha=0.8)
            
            ax.invert_yaxis()
            ax.set_aspect('equal')
            ax.axis('off')
            
            title = f"{line_id}\n\"{text[:40]}{'...' if text and len(text) > 40 else ''}\""
            ax.set_title(title, fontsize=9)
        
        # Hide unused axes
        for i in range(len(indices), len(axes)):
            axes[i].axis('off')
        
        plt.suptitle("Stroke-Letter Visualization Grid", fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save:
            save_path = self.output_dir / "grid_visualization.png"
            fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
            print(f"Saved grid: {save_path}")
        
        plt.show()
        return fig


# ============================================================
# Main
# ============================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Visualize Stroke-Letter Pairs')
    parser.add_argument('--data_root', type=str,
                        default=r'D:\Projects\Personal\IR\data\words.tgz',
                        help='Path to IAM-OnDB data root')
    parser.add_argument('--output_dir', type=str,
                        default=r'D:\Projects\Personal\IR\data\visualizations',
                        help='Output directory for visualizations')
    parser.add_argument('--line_id', type=str, default=None,
                        help='Specific line ID to visualize (e.g., a01-000u-01)')
    parser.add_argument('--n_samples', type=int, default=5,
                        help='Number of random samples to visualize')
    parser.add_argument('--grid', action='store_true',
                        help='Create grid visualization')
    parser.add_argument('--no_show', action='store_true',
                        help='Do not display plots (only save)')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Stroke-Letter Visualization")
    print("=" * 60)
    
    visualizer = StrokeLetterVisualizer(args.data_root, args.output_dir)
    
    if args.line_id:
        # Visualize specific line
        visualizer.visualize_line(args.line_id, save=True, show=not args.no_show)
    elif args.grid:
        # Create grid
        visualizer.create_grid_visualization(n_samples=args.n_samples, save=True)
    else:
        # Visualize multiple random samples
        visualizer.visualize_multiple(n_samples=args.n_samples, save=True, show=not args.no_show)


if __name__ == "__main__":
    main()

