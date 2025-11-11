"""
Setup Verification Script
Tests that all dependencies are installed correctly.
"""

import sys
from pathlib import Path

def test_imports():
    """Test that all required packages can be imported."""
    print("Testing imports...")
    
    try:
        import gensim
        print(f"  ✓ gensim {gensim.__version__}")
    except ImportError as e:
        print(f"  ✗ gensim: {e}")
        return False
    
    try:
        import tokenizers
        print(f"  ✓ tokenizers {tokenizers.__version__}")
    except ImportError as e:
        print(f"  ✗ tokenizers: {e}")
        return False
    
    try:
        import numpy as np
        print(f"  ✓ numpy {np.__version__}")
    except ImportError as e:
        print(f"  ✗ numpy: {e}")
        return False
    
    try:
        import sklearn
        print(f"  ✓ scikit-learn {sklearn.__version__}")
    except ImportError as e:
        print(f"  ✗ scikit-learn: {e}")
        return False
    
    try:
        import matplotlib
        print(f"  ✓ matplotlib {matplotlib.__version__}")
    except ImportError as e:
        print(f"  ✗ matplotlib: {e}")
        return False
    
    try:
        import umap
        print(f"  ✓ umap-learn")
    except ImportError as e:
        print(f"  ✗ umap-learn: {e}")
        return False
    
    try:
        import requests
        print(f"  ✓ requests {requests.__version__}")
    except ImportError as e:
        print(f"  ✗ requests: {e}")
        return False
    
    try:
        import tqdm
        print(f"  ✓ tqdm {tqdm.__version__}")
    except ImportError as e:
        print(f"  ✗ tqdm: {e}")
        return False
    
    return True


def test_directories():
    """Test that required directories exist."""
    print("\nTesting directory structure...")
    
    dirs = ['src', 'data', 'models', 'output', 'env']
    all_exist = True
    
    for dir_name in dirs:
        dir_path = Path(dir_name)
        if dir_path.exists():
            print(f"  ✓ {dir_name}/")
        else:
            print(f"  ✗ {dir_name}/ (missing)")
            all_exist = False
    
    return all_exist


def test_source_files():
    """Test that all source files exist."""
    print("\nTesting source files...")
    
    files = [
        'src/data_loader.py',
        'src/tokenizer.py',
        'src/trainer.py',
        'src/evaluator.py',
        'src/visualizer.py',
        'src/main.py',
        'src/interactive.py',
        'src/config.py'
    ]
    
    all_exist = True
    
    for file_name in files:
        file_path = Path(file_name)
        if file_path.exists():
            print(f"  ✓ {file_name}")
        else:
            print(f"  ✗ {file_name} (missing)")
            all_exist = False
    
    return all_exist


def test_documentation():
    """Test that documentation files exist."""
    print("\nTesting documentation...")
    
    docs = [
        'README.md',
        'USAGE.md',
        'QUICKREF.md',
        'PROJECT_SUMMARY.md',
        'requirements.txt'
    ]
    
    all_exist = True
    
    for doc_name in docs:
        doc_path = Path(doc_name)
        if doc_path.exists():
            print(f"  ✓ {doc_name}")
        else:
            print(f"  ✗ {doc_name} (missing)")
            all_exist = False
    
    return all_exist


def test_python_version():
    """Check Python version."""
    print("\nChecking Python version...")
    
    version = sys.version_info
    print(f"  Python {version.major}.{version.minor}.{version.micro}")
    
    if version.major >= 3 and version.minor >= 8:
        print("  ✓ Python version OK (3.8+)")
        return True
    else:
        print("  ✗ Python version too old (need 3.8+)")
        return False


def main():
    """Run all verification tests."""
    print("=" * 60)
    print("  SETUP VERIFICATION")
    print("=" * 60)
    
    results = []
    
    # Run tests
    results.append(("Python Version", test_python_version()))
    results.append(("Package Imports", test_imports()))
    results.append(("Directory Structure", test_directories()))
    results.append(("Source Files", test_source_files()))
    results.append(("Documentation", test_documentation()))
    
    # Summary
    print("\n" + "=" * 60)
    print("  VERIFICATION SUMMARY")
    print("=" * 60)
    
    all_passed = True
    for test_name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{test_name:<25} {status}")
        if not passed:
            all_passed = False
    
    print("=" * 60)
    
    if all_passed:
        print("\n🎉 SUCCESS! All checks passed.")
        print("\nYou can now run:")
        print("  python src/main.py          # Complete pipeline")
        print("  python src/interactive.py   # Interactive interface")
    else:
        print("\n⚠️  WARNING: Some checks failed.")
        print("\nPlease fix the issues above before running the pipeline.")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
