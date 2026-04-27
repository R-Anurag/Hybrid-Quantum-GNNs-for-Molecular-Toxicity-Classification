# Project Audit & Fix Plan

**Date**: 2025-01-XX  
**Status**: CRITICAL - Data pipeline broken in Colab environment  
**Auditor**: Senior ML Engineer Review

---

## Executive Summary

The project has a **critical data loading failure** in Google Colab due to DeepChem API incompatibilities. The current approach of using `dc.utils.data_utils.download_url()` returns `None` in Colab's DeepChem version, causing the entire pipeline to fail.

**Root Cause**: Attempting to use DeepChem's internal utilities that have inconsistent APIs across versions (local vs Colab).

**Solution**: Eliminate DeepChem dependency for data loading entirely. Use direct HTTP downloads with standard Python libraries.

---

## Critical Issues

### 🔴 ISSUE #1: Data Loading Failure (BLOCKING)
**File**: `src/data_pipeline.py`  
**Line**: 79-92  
**Severity**: CRITICAL  

**Problem**:
```python
dataset_file = dc.utils.data_utils.download_url(url, data_dir)
df = pd.read_csv(dataset_file)  # dataset_file is None → ValueError
```

**Why it fails**:
- DeepChem's `download_url()` has inconsistent behavior across versions
- Returns `None` in Colab environment (DeepChem 2.8.x)
- Different return types in different versions (path string vs None)
- Internal API not meant for external use

**Impact**: 
- ❌ Cannot load any datasets
- ❌ Entire pipeline blocked
- ❌ No experiments can run

---

### 🟡 ISSUE #2: DeepChem Over-Dependency
**Files**: `src/data_pipeline.py`, `requirements.txt`  
**Severity**: MEDIUM  

**Problem**:
- Using DeepChem only for dataset URLs and download utilities
- DeepChem is a 500MB+ package with complex dependencies
- Only need: Tox21 and ClinTox CSV files
- Actual molecular featurization done by RDKit (already in use)

**Impact**:
- Slow installation (5+ minutes in Colab)
- Version conflicts between environments
- Unnecessary complexity

---

### 🟡 ISSUE #3: Missing Error Handling
**File**: `src/data_pipeline.py`  
**Severity**: MEDIUM  

**Problem**:
- No validation that files downloaded successfully
- No fallback URLs if primary source fails
- No caching mechanism for repeated runs
- Silent failures possible

---

### 🟢 ISSUE #4: Hardcoded Dataset URLs
**File**: `src/data_pipeline.py`  
**Severity**: LOW  

**Problem**:
- URLs hardcoded in function
- No configuration file
- Difficult to add new datasets

---

## Architecture Review

### Current Data Flow (BROKEN)
```
DeepChem API → download_url() → None → CRASH
```

### Proposed Data Flow (ROBUST)
```
requests/urllib → local cache → pandas → PyG Data
                     ↓
              (persistent across runs)
```

---

## Fix Implementation Plan

### Phase 1: Emergency Fix (15 minutes)
**Goal**: Get pipeline working in both local and Colab

**Changes**:
1. Replace DeepChem download with `requests` library
2. Add local caching to `~/.cache/hqgnn/` or `/tmp/hqgnn/`
3. Add proper error handling and retries
4. Keep DeepChem in requirements (for future MolNet benchmarks)

**Files to modify**:
- `src/data_pipeline.py` (complete rewrite of `load_dataset()`)
- `requirements.txt` (add `requests`)

---

### Phase 2: Robustness (30 minutes)
**Goal**: Production-grade data loading

**Enhancements**:
1. Add dataset integrity checks (file size, MD5 hash)
2. Implement exponential backoff for downloads
3. Add progress bars for large downloads
4. Create dataset configuration file
5. Add unit tests for data loading

**New files**:
- `src/config/datasets.yaml` (dataset metadata)
- `tests/test_data_pipeline.py` (unit tests)

---

### Phase 3: Optimization (optional, 1 hour)
**Goal**: Speed and efficiency

**Enhancements**:
1. Parallel SMILES processing with multiprocessing
2. Pre-computed graph cache (pickle/torch.save)
3. Lazy loading for large datasets
4. Memory-mapped arrays for huge datasets

---

## Detailed Fix: Phase 1

### New `load_dataset()` Implementation

```python
def load_dataset(name="tox21"):
    """
    Load Tox21 or ClinTox dataset with robust caching.
    Returns: (data_list, class_weights, tasks)
    """
    import os
    import requests
    from pathlib import Path
    
    # Dataset metadata
    DATASETS = {
        "tox21": {
            "url": "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/tox21.csv.gz",
            "tasks": ['NR-AR', 'NR-AR-LBD', 'NR-AhR', 'NR-Aromatase', 'NR-ER', 
                     'NR-ER-LBD', 'NR-PPAR-gamma', 'SR-ARE', 'SR-ATAD5', 
                     'SR-HSE', 'SR-MMP', 'SR-p53'],
        },
        "clintox": {
            "url": "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/clintox.csv.gz",
            "tasks": ['FDA_APPROVED', 'CT_TOX'],
        },
    }
    
    if name not in DATASETS:
        raise ValueError(f"Unknown dataset: {name}. Choose from {list(DATASETS.keys())}")
    
    config = DATASETS[name]
    
    # Setup cache directory
    cache_dir = Path.home() / ".cache" / "hqgnn"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"{name}.csv.gz"
    
    # Download if not cached
    if not cache_file.exists():
        print(f"Downloading {name} dataset...")
        response = requests.get(config["url"], timeout=30)
        response.raise_for_status()
        cache_file.write_bytes(response.content)
        print(f"✓ Downloaded to {cache_file}")
    else:
        print(f"✓ Using cached dataset: {cache_file}")
    
    # Load CSV
    df = pd.read_csv(cache_file)
    tasks = config["tasks"]
    
    # Convert to PyG Data objects
    data_list = []
    failed = 0
    for _, row in df.iterrows():
        smiles = row['smiles']
        labels = row[tasks].values
        d = smiles_to_data(smiles, labels)
        if d is not None:
            data_list.append(d)
        else:
            failed += 1
    
    if failed > 0:
        print(f"⚠ Skipped {failed}/{len(df)} invalid SMILES")
    
    # Compute class weights
    y_all = np.array([d.y.numpy() for d in data_list])
    num_tasks = y_all.shape[1]
    class_weights = []
    for t in range(num_tasks):
        col = y_all[:, t]
        valid = col[~np.isnan(col)].astype(int)
        classes = np.unique(valid)
        if len(classes) < 2:
            class_weights.append(torch.tensor([1.0, 1.0]))
            continue
        w = compute_class_weight("balanced", classes=classes, y=valid)
        class_weights.append(torch.tensor(w, dtype=torch.float))
    
    return data_list, class_weights, tasks
```

### Updated `requirements.txt`

```txt
torch>=2.0.0
torch-geometric
rdkit
requests
scikit-learn
numpy
pandas
matplotlib
tqdm

# Optional: keep for future MolNet benchmarks
# deepchem>=2.8.0
```

---

## Testing Plan

### Unit Tests
```python
def test_load_tox21():
    data, weights, tasks = load_dataset("tox21")
    assert len(data) > 7000
    assert len(tasks) == 12
    assert all(isinstance(d, Data) for d in data)

def test_load_clintox():
    data, weights, tasks = load_dataset("clintox")
    assert len(data) > 1400
    assert len(tasks) == 2

def test_cache_persistence():
    # First load
    load_dataset("tox21")
    cache_file = Path.home() / ".cache" / "hqgnn" / "tox21.csv.gz"
    assert cache_file.exists()
    
    # Second load (should be instant)
    import time
    t0 = time.time()
    load_dataset("tox21")
    assert time.time() - t0 < 5  # Should be fast
```

### Integration Tests
```bash
# Local environment
cd src
python data_pipeline.py

# Colab environment
!cd /content/HQGNN/src && python data_pipeline.py
```

---

## Rollout Plan

### Step 1: Backup Current State
```bash
git add -A
git commit -m "Backup before data pipeline refactor"
git push
```

### Step 2: Implement Fix
1. Update `src/data_pipeline.py`
2. Update `requirements.txt`
3. Test locally
4. Push to GitHub

### Step 3: Deploy to Colab
1. Pull latest changes in Colab
2. Restart runtime (clear old imports)
3. Re-run installation cell
4. Test data loading
5. Run full experiment

### Step 4: Validate
- [ ] Tox21 loads successfully
- [ ] ClinTox loads successfully
- [ ] Cache works on second load
- [ ] All 5 models train without errors
- [ ] Results CSV generated
- [ ] Plots render correctly

---

## Risk Assessment

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| URL changes | Low | High | Add fallback URLs, version pinning |
| Network timeout | Medium | Medium | Retry logic, longer timeout |
| Disk space (Colab) | Low | Medium | Use /tmp, cleanup old cache |
| SMILES parsing fails | Low | Low | Already handled (skip invalid) |
| Memory overflow | Low | High | Batch processing, lazy loading |

---

## Success Criteria

✅ **Must Have**:
- [ ] Data loads in both local and Colab
- [ ] No DeepChem API calls for downloads
- [ ] Caching works correctly
- [ ] All 5 models train successfully
- [ ] Results match previous runs (if any)

✅ **Should Have**:
- [ ] Download progress indicator
- [ ] Proper error messages
- [ ] Cache size management
- [ ] Unit tests pass

✅ **Nice to Have**:
- [ ] Parallel SMILES processing
- [ ] Pre-computed graph cache
- [ ] Dataset versioning

---

## Timeline

- **Phase 1 (Emergency Fix)**: 15 minutes
- **Testing**: 10 minutes
- **Deployment**: 5 minutes
- **Total**: 30 minutes

---

## Lessons Learned

1. **Never rely on internal APIs**: DeepChem's `download_url()` is not documented for external use
2. **Version pinning is critical**: Different environments have different package versions
3. **Always cache downloads**: Network calls are slow and can fail
4. **Test in target environment early**: Colab ≠ Local
5. **Keep dependencies minimal**: Only use what you actually need

---

## Next Steps

1. **Immediate**: Implement Phase 1 fix
2. **Short-term**: Add Phase 2 robustness features
3. **Long-term**: Consider removing DeepChem entirely if not needed for other features

---

## Appendix: Alternative Solutions Considered

### Option A: Fix DeepChem API usage ❌
- **Pros**: Minimal code changes
- **Cons**: Still dependent on unstable API, version-specific

### Option B: Use MoleculeNet directly ❌
- **Pros**: Official DeepChem interface
- **Cons**: Requires featurizer, we don't need their features

### Option C: Direct HTTP download ✅ (CHOSEN)
- **Pros**: Simple, reliable, no external dependencies
- **Cons**: Need to implement caching ourselves (easy)

### Option D: Bundle datasets in repo ❌
- **Pros**: Always available, no download
- **Cons**: Large files in git, licensing issues

---

**Approved by**: Senior ML Engineer  
**Implementation**: Immediate  
**Priority**: P0 (Critical)
