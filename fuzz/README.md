
# Fuzzing quick start


### Setup
```bash
rustup component add llvm-tools-preview

sudo apt install llvm-dev

cargo install cargo-fuzz
```

### Run
```bash
cargo +nightly fuzz run bf_tree --release --debug-assertions -s address --jobs 30 -- -max_len=655360
```

Bf-Tree fuzzing write to /tmp directory, to accelerate IO, you want to mount /tmp to a tmpfs.
```bash
sudo umount /tmp
sudo mount -t tmpfs -o size=4G tmpfs /tmp
```

Build a debug version of the fuzz target.
```bash
cargo +nightly fuzz run bf_tree -D --debug-assertions -s address --jobs 1 -- -max_len=655360
```

### Coverage
```bash
cargo +nightly fuzz coverage bf_tree

cargo cov -- show target/x86_64-unknown-linux-gnu/coverage/x86_64-unknown-linux-gnu/release/bf_tree \
                                                      --format=html \
                                                      -instr-profile=coverage/bf_tree/coverage.profdata \
                                                      > index.html
```


### Debugging

`.vscode/launch.json` has the debug configurations for lldb and rr.

#### Debug with lldb
Fuzzing often run with release mode, once you get the crash artifact, you want to debug it in debug mode.

```bash
cargo +nightly fuzz run bf_tree -D
```


#### Debug with rr
You need to install the latest [rr](https://github.com/rr-debugger/rr/releases).
On Ubuntu:
```bash
sudo apt install ./rr-xxx.deb
```

Then run rr record:
```bash
rr record cargo fuzz ...
```

Then run rr replay:
```bash
rr replay -s 50505 -k
```

Then click debug with rr in vscode.
