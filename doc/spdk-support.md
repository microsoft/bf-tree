## SPDK support

Bf-Tree experimentally supports SPDK just for developers of Bf-Tree to understand the performance gap between optimal (SPDK) and current IO handling.
It is not intended for end-user to use the SPDK feature.

Unless you know what you are doing, you should stop reading the rest of this doc.

### Configure SPDK dev environment

#### Setup SSD
You need a machine with NVMe SSDs, and you need to detach it from Linux, so that the SPDK can take over the NVMe SSDs.
https://spdk.io/doc/getting_started.html

The script from SPDK will by default detach all the NVMe SSDs from Linux, so you want to run with allow list:
```bash
env PCI_ALLOWED="10000:01:00.0" sudo scripts/setup.sh config
```

#### Setup dev environment with Nix-shell
Install nix-shell here: https://nix.dev/install-nix.html
```bash
curl -L https://nixos.org/nix/install | sh -s -- --daemon
```

Then run the following command to enter the dev environment:
```bash
nix-shell
```
The nix scripts will automatically install all the dependencies we need and connect SPDK with a Rust wrapper.

#### (Optional) Vscode dev plugin
Nix-shell works by setting up a virtual environment with all the environment variables and dependencies we need.
For vscode to aware of the nix env variables, you need to install the plugin `nix-env-selector` and select the nix-shell environment (`shell.nix`).

#### Run benchmark with SPDK

Two steps: (1) enable the `spdk` feature, it will automatically apply to Bf-Tree (2) run the benchmark with the `sudo` command, because SPDK needs root access to run.

```bash
cargo build --release --features "spdk" && env SHUMAI_FILTER="overall" RUST_BACKTRACE=1 sudo target/release/benchmark
```
