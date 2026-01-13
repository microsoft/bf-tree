use benchmark::bench_e2e::{run_e2e_bench, E2EBench};
use mimalloc::MiMalloc;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

fn fork(use_fork: bool) -> i32 {
    if use_fork {
        unsafe { libc::fork() }
    } else {
        0
    }
}

/// We use fork to make sure each run starts with a clean state.
/// Specifically, each config with run in a separate process.
/// But if the config run multiple threads (share the same loading phase), they will run in the same process.
fn main() {
    let config = E2EBench::load().expect("Failed to parse config!");
    for c in config {
        match fork(true) {
            -1 => {
                panic!("Failed to fork!");
            }
            0 => {
                // child process
                run_e2e_bench(c);
                std::process::exit(0);
            }
            child_pid => {
                // parent process
                let mut status = 0;
                unsafe {
                    libc::waitpid(child_pid, &mut status, 0);
                }

                if libc::WIFEXITED(status) && libc::WEXITSTATUS(status) != 0 {
                    panic!(
                        "Child process exited with status {}",
                        libc::WEXITSTATUS(status)
                    );
                }
            }
        }
    }
}
