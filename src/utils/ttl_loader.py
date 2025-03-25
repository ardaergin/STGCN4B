import os
from typing import List, Dict, Union
from pathlib import Path
from rdflib import Graph


def get_ttl_files(directory: Union[str, Path], recursive: bool = False) -> List[str]:
    """
    Get all .ttl files in a directory.
    
    Args:
        directory: Directory path to search
        recursive: Whether to search recursively
        
    Returns:
        List of file paths
    """
    if isinstance(directory, str):
        directory = Path(directory)
        
    if recursive:
        return [str(path) for path in directory.glob('**/*.ttl')]
    else:
        return [str(path) for path in directory.glob('*.ttl')]

def categorize_ttl_files(base_dir: Union[str, Path]) -> Dict[str, List[str]]:
    """
    Categorize TTL files in the OfficeGraph directory structure.
    
    Args:
        base_dir: Base directory of the OfficeGraph dataset
        
    Returns:
        Dictionary with categories and file lists
    """
    if isinstance(base_dir, str):
        base_dir = Path(base_dir)
    
    result = {
        'devices': [],
        'enrichments': {
            'devices_in_rooms': [],
            # 'wikidata_days': [],
            # 'graph_learning': []
        }
    }
    
    # Device files
    device_dir = base_dir / 'devices'
    if device_dir.exists():
        for device_type_dir in device_dir.iterdir():
            if device_type_dir.is_dir():
                for file_path in device_type_dir.glob('*.ttl'):
                    result['devices'].append(str(file_path))
    
    # Enrichment files
    enrichment_dir = base_dir / 'enrichments'
    if enrichment_dir.exists():
        # Devices in rooms
        room_file = enrichment_dir / 'devices_in_rooms_enrichment.ttl'
        if room_file.exists():
            result['enrichments']['devices_in_rooms'].append(str(room_file))
        
        # Wikidata days
        # wikidata_file = enrichment_dir / 'wikidata_days_enrichment.ttl'
        # if wikidata_file.exists():
        #     result['enrichments']['wikidata_days'].append(str(wikidata_file))
            
        # Graph learning enrichments
        # gl_dir = enrichment_dir / 'floor7_graph_learning_enrichment'
        # if gl_dir.exists() and gl_dir.is_dir():
        #     for file_path in gl_dir.glob('*.ttl'):
        #         result['enrichments']['graph_learning'].append(str(file_path))
    
    return result

def load_ttl_file(file_path: str) -> Graph:
    """
    Load a TTL file into an RDFLib graph.
    
    Args:
        file_path: Path to the TTL file
        
    Returns:
        RDFLib Graph object
    """
    graph = Graph()
    graph.parse(file_path, format="turtle")
    return graph

def load_multiple_ttl_files(file_paths: List[str]) -> Graph:
    """
    Load multiple TTL files into a single RDFLib graph.
    
    Args:
        file_paths: List of paths to TTL files
        
    Returns:
        Combined RDFLib Graph object
    """
    combined_graph = Graph()
    for file_path in file_paths:
        combined_graph.parse(file_path, format="turtle")
    return combined_graph
