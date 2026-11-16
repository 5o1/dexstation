"""Numerical/asset helpers extracted unchanged from the original collection backend dependency."""
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np

def project_inertia_tensor(tensor):
    """Nearest Frobenius projection onto the positive inertia triangle cone.

    Used only for the single evidenced G20 triangle violation; non-positive
    inputs are rejected rather than inventing mass distribution.
    """
    eigenvalues, basis = np.linalg.eigh(tensor)
    if eigenvalues[0] <= 0:
        raise ValueError("non-positive input inertia")
    deficit = eigenvalues[2] - eigenvalues[0] - eigenvalues[1]
    if deficit <= 0:
        return np.asarray(tensor).copy()
    eigenvalues += (deficit / 3 + np.finfo(float).eps * eigenvalues[2]) * np.array([1.,1.,-1.])
    return (basis * eigenvalues) @ basis.T

def resolve_asset_files(xml_text, resource_root):
    """Keep composed XML reloadable after child-spec attachment loses its base path."""
    directory = Path(resource_root)
    if not (directory/'manifest.json').is_file():
        directory = directory/'mujoco/full_controls'
    root = ET.fromstring(xml_text)
    for asset in root.findall('asset/*[@file]'):
        filename = Path(asset.get('file'))
        if not filename.is_absolute():
            asset.set('file',str((directory/filename).resolve()))
    return ET.tostring(root,encoding='unicode')
