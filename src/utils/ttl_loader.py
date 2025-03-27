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

def get_device_files(base_dir: Union[str, Path]) -> List[str]:
    """
    Get all device TTL files from the OfficeGraph dataset.
    
    Args:
        base_dir: Base directory of the OfficeGraph dataset
        
    Returns:
        List of device file paths
    """
    if isinstance(base_dir, str):
        base_dir = Path(base_dir)
        
    device_files = []
    device_dir = base_dir / 'devices'
    if device_dir.exists():
        for device_type_dir in device_dir.iterdir():
            if device_type_dir.is_dir():
                for file_path in device_type_dir.glob('*.ttl'):
                    device_files.append(str(file_path))
                    
    return device_files

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
    for i, file_path in enumerate(file_paths, start=1):
        print(f"[{i}/{len(file_paths)}] Loading: {file_path}")
        combined_graph.parse(file_path, format="turtle")
    return combined_graph

def load_device_files(base_dir: Union[str, Path]) -> Graph:
    """
    Load all device files into an RDFLib graph.
    
    Args:
        base_dir: Base directory of the OfficeGraph dataset
        
    Returns:
        RDFLib Graph with all device data
    """
    file_paths = get_device_files(base_dir)
    if not file_paths:
        print("Warning: No device files found")
        return Graph()
    return load_multiple_ttl_files(file_paths)

def load_devices_in_rooms_enrichment(base_dir: Union[str, Path]) -> Graph:
    """
    Load the devices in rooms enrichment into an RDFLib graph.
    
    Args:
        base_dir: Base directory of the OfficeGraph dataset
        
    Returns:
        RDFLib Graph with the devices in rooms enrichment
    """
    if isinstance(base_dir, str):
        base_dir = Path(base_dir)
        
    enrichment_path = base_dir / 'enrichments' / 'devices_in_rooms_enrichment.ttl'
    if not enrichment_path.exists():
        print("Warning: Devices in rooms enrichment file not found")
        return Graph()
    
    return load_ttl_file(str(enrichment_path))

def load_wikidata_days_enrichment(base_dir: Union[str, Path]) -> Graph:
    """
    Load the wikidata days enrichment into an RDFLib graph.
    
    Args:
        base_dir: Base directory of the OfficeGraph dataset
        
    Returns:
        RDFLib Graph with the wikidata days enrichment
    """
    if isinstance(base_dir, str):
        base_dir = Path(base_dir)
        
    enrichment_path = base_dir / 'enrichments' / 'wikidata_days_enrichment.ttl'
    if not enrichment_path.exists():
        print("Warning: Wikidata days enrichment file not found")
        return Graph()
    
    return load_ttl_file(str(enrichment_path))

def load_floor7_graph_learning_enrichments(base_dir: Union[str, Path]) -> Graph:
    """
    Load all floor7 graph learning enrichments into an RDFLib graph.
    
    Args:
        base_dir: Base directory of the OfficeGraph dataset
        
    Returns:
        RDFLib Graph with all floor7 graph learning enrichments
    """
    if isinstance(base_dir, str):
        base_dir = Path(base_dir)
        
    gl_dir = base_dir / 'enrichments' / 'floor7_graph_learning_enrichment'
    if not gl_dir.exists() or not gl_dir.is_dir():
        print("Warning: Floor7 graph learning enrichment directory not found")
        return Graph()
    
    file_paths = [str(file_path) for file_path in gl_dir.glob('*.ttl')]
    if not file_paths:
        print("Warning: No floor7 graph learning enrichment files found")
        return Graph()
    
    return load_multiple_ttl_files(file_paths)
