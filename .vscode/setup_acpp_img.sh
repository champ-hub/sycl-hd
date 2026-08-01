set -e
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd $SCRIPT_DIR
cd ..

docker build --target dev -t acpp-syclhd .

docker run --rm -it \
    -v "$(pwd)":/workspace \
    -w /workspace \
    acpp-syclhd \
    sh .vscode/setup_acpp.sh