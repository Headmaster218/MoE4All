git clone -b linux https://github.com/c00jsw00/MoE4All.git && cd MoE4All

./scripts/install-linux.sh

cargo build --release --locked -p infr-cli

./target/release/infr run '模型.gguf'
