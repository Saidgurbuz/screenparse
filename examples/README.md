# Examples

This directory contains example configurations and scripts for using Webshot.

## Quick Start Examples

### 1. Basic Dataset Generation

```bash
# Create a simple URL list
cat > urls_sample.csv << EOF
https://github.com
https://stackoverflow.com
https://reddit.com
https://news.ycombinator.com
https://python.org
EOF

# Run the pipeline
wsd pipeline --urls urls_sample.csv --workers 4
```

### 2. Custom Configuration

See [`pipeline_config_example.sh`](pipeline_config_example.sh) for a complete pipeline with custom settings.

### 3. VLM-Based Refinement

See [`vlm_refinement_example.sh`](vlm_refinement_example.sh) for VLM-based relabeling and quality scoring.

### 4. Multi-Dataset Evaluation

See [`evaluation_example.sh`](evaluation_example.sh) for evaluating models on multiple benchmarks.

## Configuration Files

- [`urls_sample.csv`](urls_sample.csv) - Sample URL list
- [`train_config_example.yaml`](train_config_example.yaml) - YOLO training configuration (if needed)

## Scripts

- [`pipeline_config_example.sh`](pipeline_config_example.sh) - Complete dataset generation pipeline
- [`vlm_refinement_example.sh`](vlm_refinement_example.sh) - VLM-based refinement workflow
- [`evaluation_example.sh`](evaluation_example.sh) - Model evaluation examples
- [`quick_start.sh`](quick_start.sh) - Minimal quick start script

## Usage

All scripts are meant to be customized for your specific use case. Copy and modify them as needed:

```bash
# Copy example script
cp examples/quick_start.sh my_pipeline.sh

# Edit with your settings
nano my_pipeline.sh

# Run
bash my_pipeline.sh
```

## Notes

- Replace placeholder paths with your actual data directories
- Adjust worker counts based on your system resources
- For VLM operations, ensure you have sufficient GPU memory
- Start with small samples (--limit flag) to test configurations