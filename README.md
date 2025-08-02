# Single-Cell RNA Sequencing Model Comparison

This repository contains a comprehensive analysis comparing different machine learning models for single-cell RNA sequencing (scRNA-seq) data analysis, specifically focusing on cell type classification and clustering performance.

## Overview

The project evaluates and compares three different approaches for scRNA-seq data analysis:
- **Geneformer**: A transformer-based model for gene expression analysis
- **scGPT**: A large language model specifically designed for single-cell data
- **HVG (Highly Variable Genes)**: Traditional approach using highly variable genes for dimensionality reduction

## Datasets

The analysis is performed on multiple scRNA-seq datasets:
- **Ahern_covid_UK**: COVID-19 related dataset with major subset annotations
- **Green_BCL**: B-cell lymphoma dataset with author-defined cell types
- **Kock_AIDAv2**: AIDA v2 dataset with level 1 annotations
- **Perez_lupus**: Lupus dataset with author-defined cell types
- **Tabula_immune**: Immune cell dataset with broad cell class annotations

## Project Structure

```
├── run_geneformer.py          # Geneformer model execution
├── run_scGPT.py              # scGPT model execution
├── run_hvg_seuratv3.py       # HVG/Seurat v3 analysis
├── plot_umaps.py             # UMAP visualization generation
├── plot_avgbio_performance.py # Average biological performance metrics
├── dotplot_avgbio_per_ds.py  # Dot plot generation for biological metrics
├── dotplot_umap_per_ds       # Dot plot generation for UMAP metrics
├── avgbio_baseline.py        # Baseline biological metrics
├── avgbio_per_group.py       # Group-specific biological analysis
├── umap_dist_baseline.py     # Baseline UMAP distance metrics
└── umap_dist_per_group.py    # Group-specific UMAP analysis
```

## Key Features

### Model Comparison
- **Embedding Generation**: Each model generates embeddings from raw scRNA-seq data
- **UMAP Visualization**: 2D UMAP plots for visual comparison of clustering quality
- **Performance Metrics**: Comprehensive evaluation using multiple metrics

### Biological Analysis
- **Cell Type Classification**: Evaluation of model performance in identifying cell types
- **Clustering Quality**: Assessment using silhouette scores, adjusted rand index, and normalized mutual information
- **Group-wise Analysis**: Performance analysis across different biological groups (sex, ethnicity, age, etc.)

### Visualization
- **UMAP Plots**: High-quality visualizations of cell embeddings
- **Dot Plots**: Comparative analysis across datasets and models
- **Performance Heatmaps**: Visual representation of model performance metrics

## Requirements

The project requires the following key dependencies:
- `scanpy` - Single-cell analysis in Python
- `anndata` - Annotated data matrices
- `torch` - PyTorch for deep learning models
- `helical` - Framework for single-cell models
- `umap-learn` - UMAP dimensionality reduction
- `scikit-learn` - Machine learning utilities
- `matplotlib` & `seaborn` - Visualization
- `pandas` & `numpy` - Data manipulation

## Usage

### Running Models

1. **Geneformer Analysis**:
   ```bash
   python run_geneformer.py
   ```

2. **scGPT Analysis**:
   ```bash
   python run_scGPT.py
   ```

3. **HVG Analysis**:
   ```bash
   python run_hvg_seuratv3.py
   ```

### Generating Visualizations

1. **UMAP Plots**:
   ```bash
   python plot_umaps.py
   ```

2. **Performance Analysis**:
   ```bash
   python plot_avgbio_performance.py
   ```

3. **Dot Plots**:
   ```bash
   python dotplot_avgbio_per_ds.py
   python dotplot_umap_per_ds
   ```

## Output

The analysis generates:
- **Embeddings**: Model-specific embeddings stored in `.h5ad` files
- **UMAP Coordinates**: 2D UMAP projections for visualization
- **Performance Metrics**: Quantitative evaluation of model performance
- **Visualizations**: High-quality plots for publication and presentation

