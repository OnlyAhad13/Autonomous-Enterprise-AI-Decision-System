#!/bin/bash
# ============================================================================
# Run Jupyter Notebooks Headlessly using nbconvert
# ============================================================================
# Usage:
#   ./scripts/run_notebooks.sh           # Run all notebooks
#   ./scripts/run_notebooks.sh --eda     # Run only EDA notebook
#   ./scripts/run_notebooks.sh --model   # Run only baselines notebook
# ============================================================================

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
NOTEBOOKS_DIR="$PROJECT_ROOT/notebooks"
OUTPUT_DIR="$PROJECT_ROOT/notebooks/executed"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "========================================"
echo "Headless Notebook Execution"
echo "========================================"
echo "Project root: $PROJECT_ROOT"
echo "Notebooks dir: $NOTEBOOKS_DIR"
echo "Output dir: $OUTPUT_DIR"
echo ""

run_notebook() {
    local notebook_name=$1
    local input_path="$NOTEBOOKS_DIR/$notebook_name"
    local output_name="${notebook_name%.ipynb}_executed.ipynb"
    local output_path="$OUTPUT_DIR/$output_name"
    
    echo -e "${YELLOW}Running: $notebook_name${NC}"
    
    if [ ! -f "$input_path" ]; then
        echo -e "${RED}Error: Notebook not found: $input_path${NC}"
        return 1
    fi
    
    # Run with nbconvert
    jupyter nbconvert \
        --to notebook \
        --execute \
        --ExecutePreprocessor.timeout=600 \
        --ExecutePreprocessor.kernel_name=python3 \
        --output "$output_name" \
        --output-dir "$OUTPUT_DIR" \
        "$input_path"
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ Success: Output saved to $output_path${NC}"
    else
        echo -e "${RED}✗ Failed: $notebook_name${NC}"
        return 1
    fi
    echo ""
}

# Parse arguments
RUN_EDA=false
RUN_MODEL=false

if [ $# -eq 0 ]; then
    RUN_EDA=true
    RUN_MODEL=true
else
    while [[ $# -gt 0 ]]; do
        case $1 in
            --eda)
                RUN_EDA=true
                shift
                ;;
            --model)
                RUN_MODEL=true
                shift
                ;;
            --help|-h)
                echo "Usage: $0 [--eda] [--model]"
                echo ""
                echo "Options:"
                echo "  --eda     Run only 01_EDA.ipynb"
                echo "  --model   Run only 02_baselines.ipynb"
                echo ""
                echo "If no options provided, runs all notebooks."
                exit 0
                ;;
            *)
                echo "Unknown option: $1"
                exit 1
                ;;
        esac
    done
fi

# Run selected notebooks
START_TIME=$(date +%s)

if [ "$RUN_EDA" = true ]; then
    run_notebook "01_EDA.ipynb"
fi

if [ "$RUN_MODEL" = true ]; then
    run_notebook "02_baselines.ipynb"
fi

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

echo "========================================"
echo -e "${GREEN}All notebooks executed successfully!${NC}"
echo "Total time: ${DURATION}s"
echo "========================================"
echo ""
echo "Executed notebooks saved to: $OUTPUT_DIR"
echo "Check MLflow experiments with:"
echo "  mlflow ui --backend-store-uri $PROJECT_ROOT/mlruns"
