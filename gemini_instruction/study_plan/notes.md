# VLA Journey Notes

## Technical Insights

### NumPy `.ravel()` in Data Collection
* Flattens multi-dimensional arrays (like environment observations) into a 1D vector.
* Returns a **view** of the data instead of a copy (unlike `.flatten()`), saving memory and processing time during high-frequency collection.

### Python Module execution (`python -m`)
- **Issue**: Running a script directly (e.g., `python scripts/collect_data.py`) fails when it imports modules from the project root (e.g., `from utils import ...`) because Python treats the script's directory (`scripts/`) as the import root.
- **Solution**: Running as a module (`python -m scripts.collect_data`) sets the current working directory (project root) as the import root.
