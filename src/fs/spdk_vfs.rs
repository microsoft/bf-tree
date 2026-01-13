// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

use std::{
    alloc::Layout,
    cell::RefCell,
    ffi::{c_char, CStr},
    ops::Deref,
};

use crate::{
    leaf_node::DEFAULT_LEAF_NODE_SIZE,
    sync::atomic::{AtomicBool, AtomicUsize, Ordering},
};

use crate::info;
use spdk_rs::libspdk as spdk;

use super::VfsImpl;

const SECTOR_SIZE: usize = 512;
const SECTOR_PER_BLOCK: usize = DEFAULT_LEAF_NODE_SIZE / SECTOR_SIZE;

#[repr(C)]
pub struct TailQEntry<T> {
    pub tqe_next: *mut T,
    pub(crate) tqe_prev: *mut *mut T,
}

#[repr(C)]
pub struct TailQHead<T> {
    pub tqh_first: *mut T,
    pub(crate) tqh_last: *mut *mut T,
}

impl<T> TailQHead<T> {
    /// # Safety
    /// Self referential struct, can't be moved.
    pub unsafe fn self_referential_init(&mut self) {
        self.tqh_first = std::ptr::null_mut();
        self.tqh_last = &mut self.tqh_first;
    }

    pub fn tailq_is_empty(&self) -> bool {
        self.tqh_first.is_null()
    }
}

#[repr(C)]
pub struct CtrlrEntry {
    pub ctrlr: *mut spdk::spdk_nvme_ctrlr,
    pub(crate) link: TailQEntry<CtrlrEntry>,
    pub(crate) name: [c_char; 1024],
}

#[repr(C)]
pub struct NsEntry {
    pub ctrlr: *mut spdk::spdk_nvme_ctrlr,
    pub ns: *mut spdk::spdk_nvme_ns,
    pub link: TailQEntry<NsEntry>,
    pub qpair: *mut spdk::spdk_nvme_qpair,
}

impl TailQHead<CtrlrEntry> {
    /// # Safety
    /// Dereference raw pointer
    pub unsafe fn insert_tail(&mut self, entry: *mut CtrlrEntry) {
        { &mut *entry }.link.tqe_next = std::ptr::null_mut();
        { &mut *entry }.link.tqe_prev = self.tqh_last;

        *self.tqh_last = entry;
        self.tqh_last = &mut unsafe { &mut *entry }.link.tqe_next;
    }
}

impl TailQHead<NsEntry> {
    /// # Safety
    /// Dereference raw pointer
    pub unsafe fn insert_tail(&mut self, entry: *mut NsEntry) {
        { &mut *entry }.link.tqe_next = std::ptr::null_mut();
        { &mut *entry }.link.tqe_prev = self.tqh_last;

        *self.tqh_last = entry;
        self.tqh_last = &mut unsafe { &mut *entry }.link.tqe_next;
    }
}

unsafe extern "C" fn probe_cb(
    cb_ctx: *mut ::std::os::raw::c_void,
    trid: *const spdk::spdk_nvme_transport_id,
    _opts: *mut spdk::spdk_nvme_ctrlr_opts,
) -> bool {
    let target_device = CStr::from_ptr(cb_ctx as *mut c_char);
    let addr = CStr::from_ptr((*trid).traddr.as_ptr());

    if target_device != addr {
        println!(
            "[probe_cb] Skipping {}, not matching target device {}",
            addr.to_string_lossy(),
            target_device.to_string_lossy()
        );
        return false;
    }

    println!("[probe_cb] Attaching to {}", addr.to_string_lossy(),);
    true
}

unsafe fn register_ns(ctrlr: *mut spdk::spdk_nvme_ctrlr, ns: *mut spdk::spdk_nvme_ns) {
    if !spdk::spdk_nvme_ns_is_active(ns) {
        return;
    }

    let entry = std::alloc::alloc(Layout::new::<NsEntry>()) as *mut NsEntry;
    unsafe { &mut *entry }.ctrlr = ctrlr;
    unsafe { &mut *entry }.ns = ns;

    G_NAMESPACES.insert_tail(entry);

    println!(
        "Registering namespace id: {}, size: {} GB, sector: {} B",
        spdk::spdk_nvme_ns_get_id(ns),
        spdk::spdk_nvme_ns_get_size(ns) / 1024 / 1024 / 1024,
        spdk::spdk_nvme_ns_get_sector_size(ns)
    );
}

unsafe extern "C" fn attach_cb(
    _cb_ctx: *mut ::std::os::raw::c_void,
    trid: *const spdk::spdk_nvme_transport_id,
    ctrlr: *mut spdk::spdk_nvme_ctrlr,
    _opts: *const spdk::spdk_nvme_ctrlr_opts,
) {
    let addr = CStr::from_ptr((*trid).traddr.as_ptr());
    println!("[attach_cb] Attaching to {}", addr.to_string_lossy());

    let _c_data = spdk::spdk_nvme_ctrlr_get_data(ctrlr);

    let entry = std::alloc::alloc(Layout::new::<CtrlrEntry>()) as *mut CtrlrEntry;
    unsafe { &mut *entry }.ctrlr = ctrlr;
    G_CONTROLLER.insert_tail(entry);

    let mut nsid = spdk::spdk_nvme_ctrlr_get_first_active_ns(ctrlr);
    while nsid != 0 {
        let ns = spdk::spdk_nvme_ctrlr_get_ns(ctrlr, nsid);
        if ns.is_null() {
            continue;
        }
        register_ns(ctrlr, ns);
        nsid = spdk::spdk_nvme_ctrlr_get_next_active_ns(ctrlr, nsid);
    }
}

/// # Safety
/// This function is called from C code.
pub unsafe fn completion_is_err(completion: *const spdk::spdk_nvme_cpl) -> bool {
    (*completion).status().sc() != spdk::SPDK_NVME_SC_SUCCESS as u16
        || ((*completion).status().sct() != spdk::SPDK_NVME_SCT_GENERIC as u16)
}

const CORE_MASK_CSTR: &[u8; 6] = b"0xfff\0";

pub fn spdk_init(device_addr: &str) {
    unsafe {
        G_CONTROLLER.self_referential_init();
        G_NAMESPACES.self_referential_init();
    }

    let device_c_str = std::ffi::CString::new(device_addr).unwrap();

    let mut env_opts: spdk::spdk_env_opts = unsafe { std::mem::zeroed() };

    unsafe { spdk::spdk_env_opts_init(&mut env_opts) };
    env_opts.core_mask = CORE_MASK_CSTR.as_ptr() as *const i8;

    println!("env_init_rt: {:?}", env_opts);

    let rt = unsafe { spdk::spdk_env_init(&env_opts) };
    if rt < 0 {
        panic!("Unable to initialize SPDK env");
    }

    let rt = unsafe {
        spdk::spdk_nvme_probe(
            std::ptr::null_mut(),
            device_c_str.as_ptr() as *mut _,
            Some(probe_cb),
            Some(attach_cb),
            None,
        )
    };

    if rt != 0 {
        panic!("probe nvme failed: {}", rt);
    }

    if unsafe { G_CONTROLLER.tailq_is_empty() } {
        panic!("No NVMe controllers found");
    }

    info!("Initialization complete.");
}

pub static mut G_CONTROLLER: TailQHead<CtrlrEntry> = unsafe { std::mem::zeroed() };
pub static mut G_NAMESPACES: TailQHead<NsEntry> = unsafe { std::mem::zeroed() };

unsafe extern "C" fn spdk_completion(
    ctx: *mut ::std::os::raw::c_void,
    completion: *const spdk::spdk_nvme_cpl,
) {
    if completion_is_err(completion) {
        panic!("Read I/O failed");
    }

    let completed = ctx as *const AtomicBool;
    unsafe { &*completed }.store(true, Ordering::SeqCst);
}

struct SpdkQpairInstance {
    qpairs: Vec<RefCell<*mut spdk::spdk_nvme_qpair>>,
}

impl SpdkQpairInstance {
    fn new(ns_entry: *mut NsEntry) -> Self {
        let parallelism: usize = std::thread::available_parallelism().unwrap().into();
        let thread_cnt = 32.max(parallelism);
        let mut paris = Vec::with_capacity(thread_cnt);

        for _i in 0..thread_cnt {
            let qpair = unsafe {
                spdk::spdk_nvme_ctrlr_alloc_io_qpair((*ns_entry).ctrlr, std::ptr::null(), 0)
            };
            if qpair.is_null() {
                panic!("Unable to allocate IO qpair");
            }
            paris.push(RefCell::new(qpair));
        }

        Self { qpairs: paris }
    }

    fn get_current_pair(&self) -> &RefCell<*mut spdk::spdk_nvme_qpair> {
        let v = std::thread::current().id().as_u64().get();
        let idx = v % self.qpairs.len() as u64;
        self.get_pair(idx)
    }

    fn get_pair(&self, thread_id: u64) -> &RefCell<*mut spdk::spdk_nvme_qpair> {
        &self.qpairs[thread_id as usize]
    }
}

impl Drop for SpdkQpairInstance {
    fn drop(&mut self) {
        for qpair in self.qpairs.iter() {
            let qpair = qpair.borrow_mut();
            unsafe {
                spdk::spdk_nvme_ctrlr_free_io_qpair(*qpair);
            }
        }
    }
}

pub(crate) struct SpdkVfs {
    ns_entry: *mut NsEntry,
    qpair: SpdkQpairInstance,
    next_available_offset: AtomicUsize,
}

impl VfsImpl for SpdkVfs {
    fn alloc_offset(&self) -> usize {
        self.next_available_offset
            .fetch_add(DEFAULT_LEAF_NODE_SIZE, Ordering::AcqRel)
    }

    fn dealloc_offset(&self, _offset: usize) {
        // do nothing
    }

    fn open(path: impl AsRef<std::path::Path>) -> Self
    where
        Self: Sized,
    {
        let path = path.as_ref().to_str().unwrap();
        spdk_init(path);

        let ns_entry = unsafe { G_NAMESPACES.tqh_first };
        if ns_entry.is_null() {
            panic!("The first namespace is null!");
        }

        if unsafe { spdk::spdk_nvme_ns_get_csi((*ns_entry).ns) } == spdk::SPDK_NVME_CSI_ZNS {
            panic!("ZNS not supported");
        }

        {
            // I don't know why SPDK's core affinity won't work, so we have to set it here manually.
            // Note that we can't use std::thread::available_parallelism() here,
            // because it will return the current cpu limit by cgroup, which is 1.
            let num_cpu: usize = 32;
            let mut cpuset: libc::cpu_set_t = unsafe { std::mem::zeroed() };
            unsafe {
                libc::CPU_ZERO(&mut cpuset);
            }
            for cpu in 0..num_cpu {
                unsafe {
                    libc::CPU_SET(cpu, &mut cpuset);
                }
            }
            // Get the current process ID.
            let pid = std::process::id() as libc::pid_t;
            // Apply the CPU set to the current process, allowing it to run on all CPUs.
            let result = unsafe {
                libc::sched_setaffinity(pid, std::mem::size_of::<libc::cpu_set_t>(), &cpuset)
            };

            if result != 0 {
                eprintln!("Failed to set CPU affinity.");
            }
        }

        let qpair = SpdkQpairInstance::new(ns_entry);

        Self {
            ns_entry,
            qpair,
            next_available_offset: AtomicUsize::new(DEFAULT_LEAF_NODE_SIZE),
        }
    }

    fn read(&self, offset: usize, buf: &mut [u8]) {
        let lba = offset / SECTOR_SIZE;
        let completed = AtomicBool::new(false);
        let qpair = self.qpair.get_current_pair().borrow_mut();
        let rt = unsafe {
            spdk::spdk_nvme_ns_cmd_read(
                { &*self.ns_entry }.ns,
                *qpair.deref(),
                buf.as_mut_ptr() as *mut _,
                lba as u64,
                SECTOR_PER_BLOCK as u32,
                Some(spdk_completion),
                &completed as *const AtomicBool as *mut _,
                0,
            )
        };

        if rt != 0 {
            panic!("Read I/O failed");
        }

        while !completed.load(Ordering::SeqCst) {
            unsafe { spdk::spdk_nvme_qpair_process_completions(*qpair.deref(), 0) };
        }
    }

    fn write(&self, offset: usize, buf: &[u8]) {
        let lba = offset / SECTOR_SIZE;
        let completed = AtomicBool::new(false);

        let qpair = self.qpair.get_current_pair().borrow_mut();
        let rt = unsafe {
            spdk::spdk_nvme_ns_cmd_write(
                { &*self.ns_entry }.ns,
                *qpair.deref(),
                buf.as_ptr() as *mut _,
                lba as u64,
                SECTOR_PER_BLOCK as u32,
                Some(spdk_completion),
                &completed as *const AtomicBool as *mut _,
                0,
            )
        };

        if rt != 0 {
            panic!("Write I/O failed");
        }

        while !completed.load(Ordering::SeqCst) {
            unsafe { spdk::spdk_nvme_qpair_process_completions(*qpair.deref(), 0) };
        }
    }
}

use crossbeam_queue::ArrayQueue;

#[derive(Debug)]
pub(super) struct SpdkAllocGuard {
    ptr: *mut u8,
}

unsafe impl Send for SpdkAllocGuard {}

impl SpdkAllocGuard {
    fn alloc(layout: std::alloc::Layout) -> Self {
        let ptr = unsafe {
            spdk_rs::libspdk::spdk_malloc(
                layout.size() as u64,
                layout.align() as u64,
                std::ptr::null_mut(),
                spdk_rs::libspdk::SPDK_ENV_SOCKET_ID_ANY,
                spdk_rs::libspdk::SPDK_MALLOC_DMA,
            )
        };
        if ptr.is_null() {
            panic!("Unable to allocate memory");
        }
        Self {
            ptr: ptr as *mut u8,
        }
    }

    pub(super) fn from_ptr(ptr: *mut u8) -> Self {
        Self { ptr }
    }

    pub(super) fn into_ptr(self) -> *mut u8 {
        let ptr = self.ptr;
        std::mem::forget(self);
        ptr
    }
}

impl Drop for SpdkAllocGuard {
    fn drop(&mut self) {
        unsafe {
            spdk_rs::libspdk::spdk_free(self.ptr as *mut _);
        }
    }
}

pub(super) fn spdk_alloc_queue() -> &'static ArrayQueue<SpdkAllocGuard> {
    use std::sync::OnceLock;

    static ALLOC_QUEUE: OnceLock<ArrayQueue<SpdkAllocGuard>> = OnceLock::new();

    ALLOC_QUEUE.get_or_init(|| {
        let queue = ArrayQueue::new(128);
        for _ in 0..128 {
            let ptr = SpdkAllocGuard::alloc(
                std::alloc::Layout::from_size_align(DEFAULT_LEAF_NODE_SIZE, DEFAULT_LEAF_NODE_SIZE)
                    .unwrap(),
            );
            queue.push(ptr).unwrap();
        }
        queue
    })
}
