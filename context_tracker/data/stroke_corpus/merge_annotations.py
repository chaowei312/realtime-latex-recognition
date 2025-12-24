"""
Merge Stroke Data and Analyze Redundancy

This script merges stroke data from multiple annotators (hou, huang, zhang)
and analyzes redundancy between the files.
"""

import json
import sys
from pathlib import Path
from collections import defaultdict
import copy

# Fix Windows console encoding
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')


def load_stroke_data(file_path: str) -> dict:
    """Load stroke data from a JSON file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def analyze_files(data_files: dict[str, dict]) -> dict:
    """
    Analyze stroke data files for redundancy.
    
    Args:
        data_files: Dictionary mapping source name to stroke data dict
        
    Returns:
        Analysis results dictionary
    """
    # Track which symbols appear in which files
    symbol_sources = defaultdict(list)
    
    for source_name, data in data_files.items():
        for symbol_key in data.keys():
            symbol_sources[symbol_key].append(source_name)
    
    # Categorize symbols
    unique_symbols = {}  # symbol -> source
    shared_symbols = {}  # symbol -> [sources]
    
    for symbol, sources in symbol_sources.items():
        if len(sources) == 1:
            unique_symbols[symbol] = sources[0]
        else:
            shared_symbols[symbol] = sources
    
    return {
        'symbol_sources': dict(symbol_sources),
        'unique_symbols': unique_symbols,
        'shared_symbols': shared_symbols,
        'total_unique_symbols': len(symbol_sources),
        'symbols_in_one_file': len(unique_symbols),
        'symbols_in_multiple_files': len(shared_symbols)
    }


def compare_symbol_data(data_files: dict[str, dict], symbol: str) -> dict:
    """
    Compare stroke data for a symbol across different files.
    
    Args:
        data_files: Dictionary mapping source name to stroke data dict
        symbol: The symbol to compare
        
    Returns:
        Comparison results
    """
    comparison = {}
    
    for source_name, data in data_files.items():
        if symbol in data:
            symbol_data = data[symbol]
            strokes_info = {}
            
            for stroke_name, stroke_data in symbol_data.get('strokes', {}).items():
                variations = stroke_data.get('variations', [])
                strokes_info[stroke_name] = {
                    'num_variations': len(variations),
                    'points_per_variation': [len(v) for v in variations]
                }
            
            comparison[source_name] = {
                'latex': symbol_data.get('latex', ''),
                'num_strokes': len(symbol_data.get('strokes', {})),
                'strokes': strokes_info
            }
    
    return comparison


def merge_stroke_data(data_files: dict[str, dict], strategy: str = 'combine_variations') -> dict:
    """
    Merge stroke data from multiple files.
    
    Args:
        data_files: Dictionary mapping source name to stroke data dict
        strategy: Merge strategy
            - 'combine_variations': Combine all variations from all sources
            - 'first_wins': Keep the first occurrence
            - 'most_variations': Keep the source with most variations
            
    Returns:
        Merged stroke data dictionary
    """
    merged = {}
    
    # Get all unique symbols
    all_symbols = set()
    for data in data_files.values():
        all_symbols.update(data.keys())
    
    for symbol in all_symbols:
        sources_with_symbol = [
            (name, data[symbol]) 
            for name, data in data_files.items() 
            if symbol in data
        ]
        
        if len(sources_with_symbol) == 1:
            # Only one source has this symbol
            merged[symbol] = copy.deepcopy(sources_with_symbol[0][1])
        else:
            # Multiple sources have this symbol
            if strategy == 'first_wins':
                merged[symbol] = copy.deepcopy(sources_with_symbol[0][1])
                
            elif strategy == 'most_variations':
                # Find source with most total variations
                best_source = max(
                    sources_with_symbol,
                    key=lambda x: sum(
                        len(stroke.get('variations', []))
                        for stroke in x[1].get('strokes', {}).values()
                    )
                )
                merged[symbol] = copy.deepcopy(best_source[1])
                
            elif strategy == 'combine_variations':
                # Combine variations from all sources
                merged[symbol] = copy.deepcopy(sources_with_symbol[0][1])
                
                for _, other_data in sources_with_symbol[1:]:
                    for stroke_name, stroke_data in other_data.get('strokes', {}).items():
                        if stroke_name in merged[symbol].get('strokes', {}):
                            # Add variations from other source
                            existing_variations = merged[symbol]['strokes'][stroke_name].get('variations', [])
                            new_variations = stroke_data.get('variations', [])
                            
                            # Check for duplicate variations
                            for new_var in new_variations:
                                is_duplicate = False
                                for existing_var in existing_variations:
                                    if new_var == existing_var:
                                        is_duplicate = True
                                        break
                                if not is_duplicate:
                                    existing_variations.append(new_var)
                        else:
                            # New stroke not in merged, add it
                            if 'strokes' not in merged[symbol]:
                                merged[symbol]['strokes'] = {}
                            merged[symbol]['strokes'][stroke_name] = copy.deepcopy(stroke_data)
    
    return merged


def print_analysis_report(analysis: dict, data_files: dict[str, dict]):
    """Print a detailed analysis report."""
    print("=" * 70)
    print("STROKE DATA ANALYSIS REPORT")
    print("=" * 70)
    
    # File statistics
    print("\n[FILE STATISTICS]")
    print("-" * 50)
    for source_name, data in data_files.items():
        total_variations = sum(
            len(stroke.get('variations', []))
            for symbol_data in data.values()
            for stroke in symbol_data.get('strokes', {}).values()
        )
        print(f"  {source_name}:")
        print(f"    - Symbols: {len(data)}")
        print(f"    - Total stroke variations: {total_variations}")
    
    # Redundancy statistics
    print("\n[REDUNDANCY ANALYSIS]")
    print("-" * 50)
    print(f"  Total unique symbols across all files: {analysis['total_unique_symbols']}")
    print(f"  Symbols in only one file: {analysis['symbols_in_one_file']}")
    print(f"  Symbols in multiple files (redundant): {analysis['symbols_in_multiple_files']}")
    
    if analysis['symbols_in_multiple_files'] > 0:
        redundancy_rate = analysis['symbols_in_multiple_files'] / analysis['total_unique_symbols'] * 100
        print(f"  Redundancy rate: {redundancy_rate:.1f}%")
    
    # Symbols unique to each file
    print("\n[SYMBOLS PER SOURCE (Unique)]")
    print("-" * 50)
    source_unique_counts = defaultdict(int)
    for symbol, source in analysis['unique_symbols'].items():
        source_unique_counts[source] += 1
    
    for source_name in data_files.keys():
        count = source_unique_counts.get(source_name, 0)
        print(f"  {source_name}: {count} unique symbols")
    
    # Shared symbols details
    if analysis['shared_symbols']:
        print("\n[SHARED SYMBOLS (Appearing in Multiple Files)]")
        print("-" * 50)
        
        # Group by which files share them
        sharing_groups = defaultdict(list)
        for symbol, sources in analysis['shared_symbols'].items():
            sources_key = tuple(sorted(sources))
            sharing_groups[sources_key].append(symbol)
        
        for sources, symbols in sorted(sharing_groups.items(), key=lambda x: -len(x[1])):
            sources_str = " + ".join(sources)
            print(f"\n  Shared between [{sources_str}]: {len(symbols)} symbols")
            # Show first 10 symbols as examples
            sample_symbols = symbols[:10]
            symbols_display = ", ".join(f"'{s}'" for s in sample_symbols)
            if len(symbols) > 10:
                symbols_display += f", ... ({len(symbols) - 10} more)"
            print(f"    Examples: {symbols_display}")


def main():
    # Define file paths
    data_dir = Path(__file__).parent
    files = {
        'hou': data_dir / 'stroke_data-hou.json',
        'huang': data_dir / 'stroke_data--huang.json',
        'zhang': data_dir / 'stroke_data---zhang.json'
    }
    
    # Load all data files
    print("Loading stroke data files...")
    data_files = {}
    for name, path in files.items():
        if path.exists():
            data_files[name] = load_stroke_data(path)
            print(f"  [OK] Loaded {name}: {len(data_files[name])} symbols")
        else:
            print(f"  [MISSING] File not found: {path}")
    
    if not data_files:
        print("No data files found!")
        return
    
    # Analyze redundancy
    print("\nAnalyzing redundancy...")
    analysis = analyze_files(data_files)
    
    # Print report
    print_analysis_report(analysis, data_files)
    
    # Compare a few shared symbols in detail
    if analysis['shared_symbols']:
        print("\n" + "=" * 70)
        print("DETAILED COMPARISON OF SHARED SYMBOLS")
        print("=" * 70)
        
        sample_shared = list(analysis['shared_symbols'].keys())[:5]
        for symbol in sample_shared:
            print(f"\n[Symbol]: '{symbol}'")
            print("-" * 40)
            comparison = compare_symbol_data(data_files, symbol)
            for source, info in comparison.items():
                print(f"  {source}:")
                print(f"    LaTeX: {info['latex']}")
                print(f"    Strokes: {info['num_strokes']}")
                for stroke_name, stroke_info in info['strokes'].items():
                    print(f"      {stroke_name}: {stroke_info['num_variations']} variations")
    
    # Merge data - keep person with more stroke variations for redundant symbols
    print("\n" + "=" * 70)
    print("MERGING STROKE DATA")
    print("=" * 70)
    print("\n  Strategy: Keep person with more stroke variations for redundant symbols")
    
    # Show which source wins for each redundant symbol
    if analysis['shared_symbols']:
        print("\n  Redundant symbol resolution:")
        for symbol, sources in analysis['shared_symbols'].items():
            best_source = None
            best_count = 0
            for source in sources:
                if symbol in data_files[source]:
                    count = sum(
                        len(stroke.get('variations', []))
                        for stroke in data_files[source][symbol].get('strokes', {}).values()
                    )
                    if count > best_count:
                        best_count = count
                        best_source = source
            print(f"    '{symbol}': keeping {best_source} ({best_count} variations)")
    
    merged_data = merge_stroke_data(data_files, strategy='most_variations')
    
    # Calculate merged statistics
    total_symbols = len(merged_data)
    total_strokes = sum(
        len(symbol_data.get('strokes', {}))
        for symbol_data in merged_data.values()
    )
    total_variations = sum(
        len(stroke.get('variations', []))
        for symbol_data in merged_data.values()
        for stroke in symbol_data.get('strokes', {}).values()
    )
    
    print(f"\n  Merged dataset statistics:")
    print(f"    - Total symbols: {total_symbols}")
    print(f"    - Total strokes: {total_strokes}")
    print(f"    - Total variations: {total_variations}")
    
    # Save merged data
    output_path = data_dir / 'stroke_data_merged.json'
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(merged_data, f, ensure_ascii=False, indent=2)
    print(f"\n  [OK] Merged data saved to: {output_path}")
    
    return analysis, merged_data


if __name__ == '__main__':
    analysis, merged_data = main()

