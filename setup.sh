#!/bin/bash

# Exit on error
set -e

echo "Setting up webshop..."
echo "NOTE: please run scripts/setup_ragen.sh before running this script"

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Print step with color
print_step() {
    echo -e "${BLUE}[Step] ${1}${NC}"
}

pip install -r requirements.txt
# Main installation process
# TODO: merge this with the main setup script with an option to install webshop
# Install if you want to use webshop
# conda install -c pytorch faiss-cpu -y
conda install -c conda-forge openjdk=21 maven -y

# Install remaining requirements
print_step "Installing additional requirements..."
pip install -r external/webshop-minimal/requirements.txt

# webshop installation, model loading
pip install -e external/webshop-minimal/ --no-dependencies
pip install en_core_web_sm-3.8.0.tar.gz 



echo -e "${GREEN}Installation completed successfully!${NC}"
echo "To activate the environment, run: conda activate ragen"

