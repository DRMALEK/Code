    #!/usr/bin/env python3
"""
Package Compatibility Finder - Automates finding compatible package versions
for multiple Python versions (e.g., 3.6 and 3.12)
"""
import subprocess
import json
import sys
import re
from packaging import version
import requests

def get_package_python_compatibility(package_name):
    """Get Python version compatibility for a package using PyPI API"""
    url = f"https://pypi.org/pypi/{package_name}/json"
    response = requests.get(url)
    if response.status_code != 200:
        print(f"Error: Could not fetch data for {package_name}")
        return {}
    
    data = response.json()
    version_info = {}
    
    # Get all versions
    for pkg_version in data["releases"]:
        # Skip pre-releases if in production
        if re.search(r'[a-zA-Z]', pkg_version):  # alpha, beta, rc, etc.
            continue
            
        try:
            release_data = data["releases"][pkg_version]
            if not release_data:
                continue
                
            # Get the Python requires info from metadata
            requires_python = None
            for file_info in release_data:
                if "requires_python" in file_info and file_info["requires_python"]:
                    requires_python = file_info["requires_python"]
                    break
                    
            version_info[pkg_version] = requires_python
        except Exception as e:
            print(f"Error processing {package_name} {pkg_version}: {e}")
    
    return version_info

def check_package_compatibility(package_name, min_python="3.6", max_python="3.12"):
    """Find compatible versions for a package between Python versions"""
    compatibility_data = get_package_python_compatibility(package_name)
    
    compatible_versions = []
    min_py_version = version.parse(min_python)
    max_py_version = version.parse(max_python)
    
    for pkg_version, requires_python in compatibility_data.items():
        # If no Python requirement is specified, assume it's compatible
        if not requires_python:
            compatible_versions.append(pkg_version)
            continue
            
        is_compatible = True
        
        # Parse the requires_python string
        # Example: ">=3.6, <4.0"
        for constraint in requires_python.split(','):
            constraint = constraint.strip()
            if not constraint:
                continue
                
            # Extract operator and version
            match = re.match(r'([<>=!~]+)?\s*(\d+(?:\.\d+)*)', constraint)
            if not match:
                continue
                
            operator, ver = match.groups()
            if not operator:
                operator = "=="
                
            py_ver = version.parse(ver)
            
            # Check if the constraint affects our target Python versions
            if operator == "==" and py_ver != min_py_version and py_ver != max_py_version:
                is_compatible = False
            elif operator == "!=" and (py_ver == min_py_version or py_ver == max_py_version):
                is_compatible = False
            elif operator == ">" and max_py_version <= py_ver:
                is_compatible = False
            elif operator == ">=" and max_py_version < py_ver:
                is_compatible = False
            elif operator == "<" and min_py_version >= py_ver:
                is_compatible = False
            elif operator == "<=" and min_py_version > py_ver:
                is_compatible = False
            
            if not is_compatible:
                break
                
        if is_compatible:
            compatible_versions.append(pkg_version)
    
    # Sort by version
    compatible_versions.sort(key=lambda x: version.parse(x))
    return compatible_versions

def generate_compatible_requirements(requirements_file, output_file, min_python="3.6", max_python="3.12"):
    """Generate a requirements file with compatible version ranges"""
    with open(requirements_file, 'r') as f:
        packages = [line.strip() for line in f if line.strip() and not line.startswith('#')]
    
    compatible_requirements = []
    
    for package in packages:
        # Strip version specifiers if present
        package_name = re.match(r'^([a-zA-Z0-9_\-]+)', package).group(1)
        print(f"Processing {package_name}...")
        
        compatible_versions = check_package_compatibility(package_name, min_python, max_python)
        
        if compatible_versions:
            min_compatible = compatible_versions[0]
            max_compatible = compatible_versions[-1]
            spec = f"{package_name}>={min_compatible},<{version.parse(max_compatible).major}.{version.parse(max_compatible).minor + 1}.0"
            compatible_requirements.append(spec)
            print(f"  Compatible range: {spec}")
        else:
            print(f"  No compatible versions found for {package_name}")
            compatible_requirements.append(f"# No compatible version found: {package_name}")
    
    with open(output_file, 'w') as f:
        f.write("# Compatible requirements for Python {min_python} to {max_python}\n")
        for req in compatible_requirements:
            f.write(f"{req}\n")
    
    print(f"\nCompatible requirements written to {output_file}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python compatibility_finder.py input_requirements.txt output_requirements.txt [min_python_version] [max_python_version]")
        sys.exit(1)
        
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    min_python = sys.argv[3] if len(sys.argv) > 3 else "3.6"
    max_python = sys.argv[4] if len(sys.argv) > 4 else "3.12"
    
    generate_compatible_requirements(input_file, output_file, min_python, max_python)