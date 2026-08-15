//! Benchmarks for the `atree` public API.
//!
//! The suite covers the three cost centers of a real `atree` invocation:
//!
//! 1. the parallel filesystem scan ([`build_graph`]) over a generated fixture tree,
//! 2. the graph algorithms ([`compute_depths`], [`astar`], [`bfs_expanded`]),
//! 3. the output pipelines (tree rendering, Graphviz DOT, JSON report assembly).
//!
//! Graph/output benchmarks run on synthetic, in-memory [`ScanResult`]s so they
//! measure pure CPU work and stay independent of the machine's filesystem.
//! Fixture construction always happens outside the measured region.

use std::fs;
use std::path::PathBuf;
use std::sync::OnceLock;

use atree::{
    astar, bfs_expanded, build_graph, build_json_report, build_path_report, compute_depths,
    generate_dot, human_size, print_tree, sanitize_name, JsonReport, NodeMeta, ScanOptions,
    ScanResult, Stats,
};
use divan::{black_box, Bencher};
use rustc_hash::{FxHashMap, FxHashSet};

fn main() {
    divan::main();
}

// =====================================================================
// Fixtures
// =====================================================================

const ROOT_NAME: &str = "bench_root";

fn child_key(parent: &str, name: &str) -> String {
    if parent == ROOT_NAME {
        name.to_string()
    } else {
        format!("{}/{}", parent, name)
    }
}

fn dir_meta(name: &str) -> NodeMeta {
    NodeMeta {
        is_dir: true,
        is_symlink: false,
        is_hidden: false,
        is_exec: false,
        mode: 0o755,
        size: 0,
        name: name.to_string(),
    }
}

fn file_meta(name: &str, size: u64) -> NodeMeta {
    NodeMeta {
        is_dir: false,
        is_symlink: false,
        is_hidden: false,
        is_exec: false,
        mode: 0o644,
        size,
        name: name.to_string(),
    }
}

/// Build an in-memory scan result shaped like a real project tree:
/// `dirs_per_dir` subdirectories and `files_per_dir` files per directory,
/// nested `depth` levels deep.
fn synthetic_scan(depth: usize, dirs_per_dir: usize, files_per_dir: usize) -> ScanResult {
    let root_name = ROOT_NAME.to_string();
    let mut adj: FxHashMap<String, Vec<String>> = FxHashMap::default();
    let mut meta: FxHashMap<String, NodeMeta> = FxHashMap::default();
    let mut stats = Stats::default();

    meta.insert(root_name.clone(), dir_meta(&root_name));
    adj.entry(root_name.clone()).or_default();
    stats.total_nodes = 1;
    stats.folders = 1;

    let mut frontier = vec![(root_name.clone(), 0usize)];
    while let Some((parent, level)) = frontier.pop() {
        if level >= depth {
            continue;
        }
        for i in 0..dirs_per_dir {
            let name = format!("dir_{:02}", i);
            let key = child_key(&parent, &name);
            meta.insert(key.clone(), dir_meta(&name));
            adj.entry(key.clone()).or_default().push(parent.clone());
            adj.entry(parent.clone()).or_default().push(key.clone());
            stats.total_nodes += 1;
            stats.folders += 1;
            frontier.push((key, level + 1));
        }
        for i in 0..files_per_dir {
            let name = format!("file_{:02}.rs", i);
            let key = child_key(&parent, &name);
            let size = 1024 * (i as u64 + 1);
            meta.insert(key.clone(), file_meta(&name, size));
            adj.entry(key.clone()).or_default().push(parent.clone());
            adj.entry(parent.clone()).or_default().push(key);
            stats.total_nodes += 1;
            stats.files += 1;
            stats.total_size_bytes += size;
        }
    }

    ScanResult {
        adj,
        root_name,
        meta,
        stats,
        truncated: false,
    }
}

/// ~120 nodes — a small project directory.
fn small_scan() -> &'static ScanResult {
    static SMALL: OnceLock<ScanResult> = OnceLock::new();
    SMALL.get_or_init(|| synthetic_scan(3, 3, 4))
}

/// ~1000 nodes — a mid-sized repository.
fn large_scan() -> &'static ScanResult {
    static LARGE: OnceLock<ScanResult> = OnceLock::new();
    LARGE.get_or_init(|| synthetic_scan(5, 3, 5))
}

fn depths_of(scan: &ScanResult) -> &'static FxHashMap<String, i32> {
    // Depth maps are keyed per scan size; both fixtures are process-global.
    if scan.stats.total_nodes == small_scan().stats.total_nodes {
        static SMALL_DEPTHS: OnceLock<FxHashMap<String, i32>> = OnceLock::new();
        SMALL_DEPTHS.get_or_init(|| compute_depths(&small_scan().adj, &small_scan().root_name))
    } else {
        static LARGE_DEPTHS: OnceLock<FxHashMap<String, i32>> = OnceLock::new();
        LARGE_DEPTHS.get_or_init(|| compute_depths(&large_scan().adj, &large_scan().root_name))
    }
}

/// The deepest node of a fixture, used as a worst-case pathfinding target.
fn deepest_node(scan: &ScanResult, depths: &FxHashMap<String, i32>) -> String {
    let mut best: (i32, &str) = (-1, scan.root_name.as_str());
    for (node, depth) in depths {
        if *depth > best.0 || (*depth == best.0 && node.as_str() < best.1) {
            best = (*depth, node.as_str());
        }
    }
    best.1.to_string()
}

/// Create (once per process) a real directory tree on disk for scan benchmarks.
fn fixture_dir() -> &'static PathBuf {
    static DIR: OnceLock<PathBuf> = OnceLock::new();
    DIR.get_or_init(|| {
        let root = std::env::temp_dir().join(format!("atree_bench_fixture_{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(&root).expect("create fixture root");
        create_tree(&root, 3, 3, 4);
        root
    })
}

fn create_tree(dir: &PathBuf, depth: usize, dirs_per_dir: usize, files_per_dir: usize) {
    for i in 0..files_per_dir {
        fs::write(
            dir.join(format!("file_{:02}.rs", i)),
            b"// atree bench fixture\n",
        )
        .expect("write fixture file");
    }
    if depth == 0 {
        return;
    }
    for i in 0..dirs_per_dir {
        let sub = dir.join(format!("dir_{:02}", i));
        fs::create_dir_all(&sub).expect("create fixture dir");
        create_tree(&sub, depth - 1, dirs_per_dir, files_per_dir);
    }
}

fn scan_options(root: PathBuf, include_files: bool, tree_mode: bool) -> ScanOptions {
    ScanOptions {
        root,
        max_depth: 64,
        max_nodes: 100_000,
        include_files,
        // Single-threaded so the measurement reflects the scan work itself and
        // not the non-deterministic interleaving of the work-stealing queues.
        threads: 1,
        tree_mode,
    }
}

fn json_report_fixture(scan: &ScanResult) -> JsonReport {
    let depths = depths_of(scan);
    let goal = deepest_node(scan, depths);
    let path = astar(&scan.adj, &scan.root_name, &goal, depths).map(|(nodes, expanded)| {
        let bfs = bfs_expanded(&scan.adj, &scan.root_name, &goal);
        build_path_report(&scan.root_name, &goal, &nodes, expanded, bfs)
    });
    build_json_report(
        scan,
        &scan_options(PathBuf::from("/tmp/atree-bench"), true, false),
        depths,
        path,
        12.34,
    )
}

// =====================================================================
// Filesystem scan
// =====================================================================

mod scan {
    use super::*;

    /// Full scan: directories, files and a `stat` call per entry.
    #[divan::bench]
    fn build_graph_full(bencher: Bencher) {
        let opts = scan_options(fixture_dir().clone(), true, false);
        bencher.bench(|| black_box(build_graph(black_box(&opts)).expect("scan")));
    }

    /// `--tree` mode: skips the per-file `stat`, the fast path of the CLI.
    #[divan::bench]
    fn build_graph_tree_mode(bencher: Bencher) {
        let opts = scan_options(fixture_dir().clone(), true, true);
        bencher.bench(|| black_box(build_graph(black_box(&opts)).expect("scan")));
    }

    /// Directories only, as used by the default depth-limited overview.
    #[divan::bench]
    fn build_graph_dirs_only(bencher: Bencher) {
        let opts = scan_options(fixture_dir().clone(), false, false);
        bencher.bench(|| black_box(build_graph(black_box(&opts)).expect("scan")));
    }
}

// =====================================================================
// Graph algorithms
// =====================================================================

mod graph {
    use super::*;

    #[divan::bench]
    fn compute_depths_small(bencher: Bencher) {
        let scan = small_scan();
        bencher.bench(|| black_box(compute_depths(&scan.adj, black_box(&scan.root_name))));
    }

    #[divan::bench]
    fn compute_depths_large(bencher: Bencher) {
        let scan = large_scan();
        bencher.bench(|| black_box(compute_depths(&scan.adj, black_box(&scan.root_name))));
    }

    /// A* from the root to the deepest node of a ~1000-node tree.
    #[divan::bench]
    fn astar_root_to_deepest(bencher: Bencher) {
        let scan = large_scan();
        let depths = depths_of(scan);
        let goal = deepest_node(scan, depths);
        bencher.bench(|| {
            black_box(astar(
                &scan.adj,
                black_box(&scan.root_name),
                black_box(&goal),
                depths,
            ))
        });
    }

    /// A* between two leaves in different subtrees: the heuristic has to work
    /// harder because the path goes up through a common ancestor.
    #[divan::bench]
    fn astar_leaf_to_leaf(bencher: Bencher) {
        let scan = large_scan();
        let depths = depths_of(scan);
        let start = "dir_00/dir_00/dir_00/dir_00/file_00.rs";
        let goal = "dir_02/dir_02/dir_02/dir_02/file_04.rs";
        assert!(scan.adj.contains_key(start) && scan.adj.contains_key(goal));
        bencher.bench(|| black_box(astar(&scan.adj, black_box(start), black_box(goal), depths)));
    }

    /// Blind BFS baseline that A* is compared against in the efficiency report.
    #[divan::bench]
    fn bfs_expanded_root_to_deepest(bencher: Bencher) {
        let scan = large_scan();
        let depths = depths_of(scan);
        let goal = deepest_node(scan, depths);
        bencher.bench(|| {
            black_box(bfs_expanded(
                &scan.adj,
                black_box(&scan.root_name),
                black_box(&goal),
            ))
        });
    }
}

// =====================================================================
// Rendering
// =====================================================================

mod render {
    use super::*;

    fn path_set(scan: &ScanResult) -> FxHashSet<String> {
        let depths = depths_of(scan);
        let goal = deepest_node(scan, depths);
        match astar(&scan.adj, &scan.root_name, &goal, depths) {
            Some((nodes, _)) => nodes.into_iter().collect(),
            None => FxHashSet::default(),
        }
    }

    /// Unicode + color rendering of a ~1000-node tree into an in-memory buffer.
    #[divan::bench]
    fn print_tree_unicode(bencher: Bencher) {
        let scan = large_scan();
        let depths = depths_of(scan);
        let path = path_set(scan);
        bencher
            .with_inputs(|| Vec::<u8>::with_capacity(256 * 1024))
            .bench_values(|mut out| {
                print_tree(
                    &mut out,
                    &scan.adj,
                    &scan.meta,
                    &scan.root_name,
                    depths,
                    false,
                    false,
                    &path,
                )
                .expect("render");
                out
            });
    }

    /// ASCII, no-color rendering — the pipe-friendly output path.
    #[divan::bench]
    fn print_tree_ascii_plain(bencher: Bencher) {
        let scan = large_scan();
        let depths = depths_of(scan);
        let path = path_set(scan);
        bencher
            .with_inputs(|| Vec::<u8>::with_capacity(256 * 1024))
            .bench_values(|mut out| {
                print_tree(
                    &mut out,
                    &scan.adj,
                    &scan.meta,
                    &scan.root_name,
                    depths,
                    true,
                    true,
                    &path,
                )
                .expect("render");
                out
            });
    }

    /// Graphviz DOT generation for a small tree (writes to a temp file).
    #[divan::bench]
    fn generate_dot_small(bencher: Bencher) {
        let scan = small_scan();
        let depths = depths_of(scan);
        let goal = deepest_node(scan, depths);
        let path = astar(&scan.adj, &scan.root_name, &goal, depths)
            .map(|(nodes, _)| nodes)
            .unwrap_or_default();
        let out_path = std::env::temp_dir()
            .join(format!("atree_bench_{}.dot", std::process::id()))
            .to_string_lossy()
            .to_string();
        bencher.bench(|| {
            generate_dot(
                &scan.adj,
                &scan.meta,
                depths,
                &path,
                &scan.root_name,
                &out_path,
            )
            .expect("dot")
        });
    }
}

// =====================================================================
// JSON output
// =====================================================================

mod json {
    use super::*;

    /// Assemble the `JsonReport` (FxHashMap → sorted BTreeMap conversions).
    #[divan::bench]
    fn build_report(bencher: Bencher) {
        let scan = large_scan();
        let depths = depths_of(scan);
        let opts = scan_options(PathBuf::from("/tmp/atree-bench"), true, false);
        bencher.bench(|| {
            black_box(build_json_report(
                black_box(scan),
                &opts,
                depths,
                None,
                12.34,
            ))
        });
    }

    /// Serialize the report exactly like `atree --json` does.
    #[divan::bench]
    fn serialize_report(bencher: Bencher) {
        let report = json_report_fixture(large_scan());
        bencher
            .with_inputs(|| Vec::<u8>::with_capacity(1024 * 1024))
            .bench_values(|mut out| {
                serde_json::to_writer(&mut out, black_box(&report)).expect("serialize");
                out
            });
    }

    /// Parse a report back, as a downstream consumer of the JSON output would.
    #[divan::bench]
    fn deserialize_report(bencher: Bencher) {
        let json = serde_json::to_string(&json_report_fixture(large_scan())).expect("serialize");
        bencher.bench(|| {
            black_box(serde_json::from_str::<JsonReport>(black_box(&json)).expect("deserialize"))
        });
    }
}

// =====================================================================
// Helpers on the scan hot path
// =====================================================================

mod helpers {
    use super::*;

    /// `sanitize_name` runs once per directory entry during every scan.
    #[divan::bench]
    fn sanitize_name_clean(bencher: Bencher) {
        let names: Vec<String> = (0..256)
            .map(|i| format!("some_source_file_{:03}.rs", i))
            .collect();
        bencher.bench(|| {
            for name in &names {
                black_box(sanitize_name(black_box(name)));
            }
        });
    }

    /// Worst case: names full of control characters that must be replaced.
    #[divan::bench]
    fn sanitize_name_hostile(bencher: Bencher) {
        let names: Vec<String> = (0..256)
            .map(|i| format!("\x1b[2J\x07evil_{:03}\n\t.txt", i))
            .collect();
        bencher.bench(|| {
            for name in &names {
                black_box(sanitize_name(black_box(name)));
            }
        });
    }

    /// Size formatting, called once per rendered node.
    #[divan::bench]
    fn human_size_mixed(bencher: Bencher) {
        let sizes: Vec<u64> = (0..256).map(|i| (i as u64 + 1) * 7919).collect();
        bencher.bench(|| {
            for size in &sizes {
                black_box(human_size(black_box(*size)));
            }
        });
    }
}
