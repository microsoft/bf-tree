// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

mod freelist;
mod metrics;

use std::{cell::UnsafeCell, marker::PhantomData, sync::TryLockError};

use crate::{
    circular_buffer::freelist::FreeList,
    sync::{
        atomic::{AtomicU8, AtomicUsize, Ordering},
        Mutex, MutexGuard,
    },
    utils::Backoff,
};

use std::convert::Into;

pub use self::metrics::CircularBufferMetrics;

pub(crate) const CB_ALLOC_META_SIZE: usize = std::mem::size_of::<AllocMeta>();

const BUFFER_ALIGNMENT: usize = 4096;

#[repr(u8)]
#[derive(Debug, PartialEq, Eq)]
enum MetaState {
    NotReady = 0,
    Ready = 1,
    Tombstone = 2,
    BeginTombStone = 3,
    FreeListed = 4,
    Evicted = 5, // the head address can only pass this memory if it is evicted.
}

impl From<MetaState> for u8 {
    fn from(state: MetaState) -> u8 {
        state as u8
    }
}

/// Life cycle of a piece of memory in the circular buffer:
/// 1. NOT_READY (allocated)
/// 2. READY (used by caller, i.e., the `CircularBufferPtr` is dropped)
/// 3. BEGIN_TOMBSTONE (before deallocating/evicting the memory, it is essentially a x-lock, whoever wins gets to deallocate/evict)
/// 4. TOMBSTONE (not accessible to any thread, free to be reused)
///
/// You can only be in one state at any given moment.
struct MetaRawState {
    state: AtomicU8,
}

impl MetaRawState {
    fn new_not_ready() -> Self {
        Self {
            state: AtomicU8::new(MetaState::NotReady.into()),
        }
    }
    fn new_tombstoned() -> Self {
        Self {
            state: AtomicU8::new(MetaState::Tombstone.into()),
        }
    }

    /// Will panic if the old state is not `META_STATE_NOT_READY`
    fn to_ready(&self) {
        match self.state.compare_exchange(
            MetaState::NotReady.into(),
            MetaState::Ready.into(),
            Ordering::AcqRel,
            Ordering::Relaxed,
        ) {
            Ok(_) => {}
            Err(v) => {
                panic!(
                    "Meta state incorrect, expected {:?}, actual {}",
                    MetaState::NotReady,
                    v
                );
            }
        }
    }

    fn try_begin_tombstone(&self) -> bool {
        self.state
            .compare_exchange(
                MetaState::Ready.into(),
                MetaState::BeginTombStone.into(),
                Ordering::AcqRel,
                Ordering::Relaxed,
            )
            .is_ok()
    }

    fn is_tombstoned(&self) -> bool {
        self.load() == <MetaState as Into<u8>>::into(MetaState::Tombstone)
    }

    fn is_evicted(&self) -> bool {
        self.load() == <MetaState as Into<u8>>::into(MetaState::Evicted)
    }

    fn is_freelisted(&self) -> bool {
        self.load() == <MetaState as Into<u8>>::into(MetaState::FreeListed)
    }

    fn load(&self) -> u8 {
        self.state.load(Ordering::Acquire)
    }

    fn state(&self) -> MetaState {
        let v = self.load();
        unsafe { std::mem::transmute(v) }
    }

    fn revert_to_ready(&self) {
        match self.state.compare_exchange(
            MetaState::BeginTombStone.into(),
            MetaState::Ready.into(),
            Ordering::AcqRel,
            Ordering::Relaxed,
        ) {
            Ok(_) => {}
            Err(v) => {
                panic!(
                    "Meta state incorrect, expected {:?}, actual {}",
                    MetaState::BeginTombStone,
                    v
                );
            }
        }
    }

    fn free_list_to_tombstone(&self) {
        match self.state.compare_exchange(
            MetaState::FreeListed.into(),
            MetaState::Tombstone.into(),
            Ordering::AcqRel,
            Ordering::Relaxed,
        ) {
            Ok(_) => {}
            Err(v) => {
                panic!(
                    "Meta state incorrect, expected {:?}, actual {}",
                    MetaState::FreeListed,
                    v
                );
            }
        }
    }

    fn to_freelist(&self) {
        match self.state.compare_exchange(
            MetaState::BeginTombStone.into(),
            MetaState::FreeListed.into(),
            Ordering::AcqRel,
            Ordering::Relaxed,
        ) {
            Ok(_) => {}
            Err(v) => {
                panic!(
                    "Meta state incorrect, expected {:?}, actual {}",
                    MetaState::Ready,
                    v
                );
            }
        }
    }

    /// Previous state must be `META_STATE_BEGIN_TOMBSTONE`
    fn to_tombstone(&self) {
        match self.state.compare_exchange(
            MetaState::BeginTombStone.into(),
            MetaState::Tombstone.into(),
            Ordering::AcqRel,
            Ordering::Relaxed,
        ) {
            Ok(_) => {}
            Err(v) => {
                panic!(
                    "Meta state incorrect, expected {:?}, actual {}",
                    MetaState::BeginTombStone,
                    v
                );
            }
        }
    }

    fn tombstone_to_evicted(&self) {
        match self.state.compare_exchange(
            MetaState::Tombstone.into(),
            MetaState::Evicted.into(),
            Ordering::AcqRel,
            Ordering::Relaxed,
        ) {
            Ok(_) => {}
            Err(v) => {
                panic!(
                    "Meta state incorrect, expected {:?}, actual {}",
                    MetaState::Tombstone,
                    v
                );
            }
        }
    }
}

#[cfg(all(feature = "shuttle", test))]
#[repr(C, align(256))]
struct AllocMeta {
    /// the allocated size, not including the meta itself
    pub(crate) size: u32,

    states: MetaRawState,
}

#[cfg(not(all(feature = "shuttle", test)))]
#[repr(C, align(8))]
struct AllocMeta {
    /// the allocated size, not including the meta itself
    pub(crate) size: u32,

    states: MetaRawState,
}

impl AllocMeta {
    fn new(size: u32, tombstone: bool) -> Self {
        #[cfg(not(feature = "shuttle"))]
        debug_assert_eq!(std::mem::size_of::<AllocMeta>(), 8);

        let states = if tombstone {
            MetaRawState::new_tombstoned()
        } else {
            MetaRawState::new_not_ready()
        };

        Self { size, states }
    }

    fn data_ptr(&self) -> *mut u8 {
        unsafe { (self as *const Self as *mut u8).add(std::mem::size_of::<Self>()) }
    }

    fn state(&self) -> MetaState {
        self.states.state()
    }
}

fn align_up(addr: usize, align: usize) -> usize {
    (addr + align - 1) & !(align - 1)
}

/// The guard returned by [CircularBuffer::alloc].
/// While this guard is being hold, the allocated memory is not allowed to be evicted.
/// This means that you may block the evicting thread if you hold this guard for too long.
///
/// The lock will be released once the guard is dropped.
pub struct CircularBufferPtr<'a> {
    ptr: *mut u8,
    _pt: PhantomData<&'a ()>,
}

impl CircularBufferPtr<'_> {
    fn new(ptr: *mut u8) -> Self {
        Self {
            ptr,
            _pt: PhantomData,
        }
    }

    /// Get the actual pointer to the allocated memory.
    pub fn as_ptr(&self) -> *mut u8 {
        self.ptr
    }
}

impl Drop for CircularBufferPtr<'_> {
    fn drop(&mut self) {
        // set can be evicted to true
        let meta = CircularBuffer::get_meta_from_data_ptr(self.ptr);
        meta.states.to_ready();
    }
}

/// This is a opaque handle that you can use to deallocate the ptr.
/// You need to acquire this handle by [CircularBuffer::acquire_exclusive_dealloc_handle] before [CircularBuffer::dealloc] any memory.
///
#[derive(Debug)]
pub struct TombstoneHandle {
    pub(crate) ptr: *mut u8,
}

impl TombstoneHandle {
    fn into_ptr(self) -> *mut u8 {
        let ptr = self.ptr;
        std::mem::forget(self);
        ptr
    }

    pub fn as_ptr(&self) -> *mut u8 {
        self.ptr
    }
}

impl Drop for TombstoneHandle {
    fn drop(&mut self) {
        let meta = CircularBuffer::get_meta_from_data_ptr(self.ptr);
        meta.states.revert_to_ready();
    }
}

#[derive(Debug)]
pub enum CircularBufferError {
    Full,
    EmptyAlloc,
    WouldBlock,
}

impl std::fmt::Display for CircularBufferError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CircularBufferError::Full => write!(f, "CircularBuffer is full"),
            CircularBufferError::EmptyAlloc => write!(f, "Empty allocation"),
            CircularBufferError::WouldBlock => write!(f, "Would block"),
        }
    }
}

impl std::error::Error for CircularBufferError {}

#[derive(Debug)]
struct States {
    head_addr: AtomicUsize,
    evicting_addr: usize,
    tail_addr: usize,
}

impl States {
    fn new() -> Self {
        Self {
            head_addr: AtomicUsize::new(0),
            evicting_addr: 0,
            tail_addr: 0,
        }
    }

    fn head_addr(&self) -> usize {
        self.head_addr.load(Ordering::Relaxed)
    }

    fn tail_addr(&self) -> usize {
        self.tail_addr
    }
}

/// The circular buffer inspired by FASTER's ring buffer.
/// It acts mostly like a variable length buffer pool, except that evicting entires are handled by the callback.
///
///
/// Getting this to be correct is quite challenging, especially that we need to support concurrent allocation/deallocation/eviction,
/// and we don't want a big lock on everything.
#[derive(Debug)]
pub struct CircularBuffer {
    states: UnsafeCell<States>,
    capacity: usize,
    data_ptr: *mut u8,
    lock: Mutex<()>,

    free_list: FreeList,

    /// if true, when dropping the circular buffer, we will check and panic if there is any elements that are not marked as tombstone.
    check_tombstone_on_drop: bool,

    copy_on_access_threshold: usize,
}

impl Drop for CircularBuffer {
    fn drop(&mut self) {
        if self.check_tombstone_on_drop {
            let iter = self.iter().unwrap();
            for meta in iter {
                assert!(meta.states.is_tombstoned() || meta.states.is_freelisted());
            }
        }

        let layout = std::alloc::Layout::from_size_align(self.capacity, BUFFER_ALIGNMENT).unwrap();
        unsafe { std::alloc::dealloc(self.data_ptr, layout) };
    }
}

impl CircularBuffer {
    /// Create a new circular buffer with the given capacity, the capacity has to be a power of two and large enough to hold at least one leaf page.
    ///
    /// TODO: I don't like the fact that we require users to set cache capacity to be power of two just because it is easy to do modulo.
    /// We should actually investigate how much performance is actually gained by requiring power of two.
    ///
    /// TODO: I don't think we should ever expose the `copy_on_access_percent` to user, it is an internal implementation detail.
    ///
    ///
    /// ```
    /// use bf_tree::circular_buffer::CircularBuffer;
    /// let buffer = CircularBuffer::new(4096 * 2, 0.1, 64, 1952, 4096, 32, None, false);
    /// ```
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        capacity: usize,
        copy_on_access_percent: f64,
        min_record_size: usize,
        max_record_size: usize,
        leaf_page_size: usize,
        max_fence_len: usize,
        pre_alloc_ptr: Option<*mut u8>,
        cache_only: bool,
    ) -> Self {
        assert!(capacity.is_power_of_two());
        // It needs to accomodate at least one full page
        assert!(capacity >= leaf_page_size + std::mem::size_of::<AllocMeta>());

        let layout = std::alloc::Layout::from_size_align(capacity, BUFFER_ALIGNMENT).unwrap();
        let ptr = match pre_alloc_ptr {
            Some(p) => {
                assert_eq!(layout.size(), capacity);
                p
            }
            None => unsafe { std::alloc::alloc(layout) },
        };

        let copy_on_access_threshold = (capacity as f64 * (1.0 - copy_on_access_percent)) as usize;

        Self {
            states: UnsafeCell::new(States::new()),
            capacity,
            free_list: FreeList::new(
                min_record_size,
                max_record_size,
                leaf_page_size,
                max_fence_len,
                cache_only,
            ),
            data_ptr: ptr,
            lock: Mutex::new(()),
            check_tombstone_on_drop: true,
            copy_on_access_threshold,
        }
    }

    /// Returns the metrics of CircularBuffer.
    /// Note that this is a very slow, exclusive operation,
    /// it essentially stops all other operations,
    /// so use it with caution.
    ///
    /// You should only use it for debugging and testing.
    pub fn get_metrics(&self) -> CircularBufferMetrics {
        let (lock, states) = self.lock_states();

        let mut metrics = CircularBufferMetrics::new(self.capacity, states);

        let iter = AllocatedIter {
            _lock: lock,
            buffer: self,
            head_addr: states.head_addr(),
            tail_addr: states.tail_addr(),
        };

        let mut tombstone_size = 0;

        for meta in iter {
            match meta.state() {
                MetaState::Ready => metrics.ready_cnt += 1,
                MetaState::NotReady => metrics.not_ready_cnt += 1,
                MetaState::Tombstone => {
                    metrics.tombstone_cnt += 1;
                    tombstone_size += meta.size as usize;
                }
                MetaState::BeginTombStone => metrics.begin_tombstone_cnt += 1,
                MetaState::FreeListed => metrics.free_listed_cnt += 1,
                MetaState::Evicted => metrics.evicted_cnt += 1,
            }
            metrics.allocated_cnt += 1;
            let alloc_size = meta.size as usize;
            metrics
                .size_cnt
                .entry(alloc_size)
                .and_modify(|v| *v += 1)
                .or_insert(1);
        }
        metrics.tombstone_size = tombstone_size;
        metrics
    }

    #[allow(clippy::mut_from_ref)]
    fn try_get_states(&self) -> Result<(MutexGuard<'_, ()>, &mut States), CircularBufferError> {
        let lock = match self.lock.try_lock() {
            Ok(v) => v,
            Err(TryLockError::Poisoned(_)) => {
                panic!("Poisoned lock")
            }
            Err(TryLockError::WouldBlock) => return Err(CircularBufferError::WouldBlock),
        };

        let states = unsafe { &mut *self.states.get() };
        Ok((lock, states))
    }

    #[allow(clippy::mut_from_ref)]
    fn lock_states(&self) -> (MutexGuard<'_, ()>, &mut States) {
        (self.lock.lock().unwrap(), unsafe {
            &mut *self.states.get()
        })
    }

    /// Allocate a piece of memory from the circular buffer, returns a guard that will panic if not used.
    ///
    /// Ignores alignment, always align to 8
    /// Returns None if we have no free space, which caller needs to call [CircularBuffer::evict_one] or [CircularBuffer::evict_n].
    ///
    /// ```
    /// use bf_tree::circular_buffer::CircularBuffer;
    /// let mut buffer = CircularBuffer::new(4096 * 2, 0.1, 64, 1952, 4096, 32, None, false);
    ///
    /// let allocated = buffer.alloc(128);
    /// let ptr = allocated.unwrap().as_ptr();
    ///
    /// let v = unsafe { buffer.acquire_exclusive_dealloc_handle(ptr).unwrap() };
    /// buffer.dealloc(v); // dealloc is mandatory before buffer being dropped.
    /// ```
    #[cfg_attr(feature = "tracing", tracing::instrument)]
    pub fn alloc(&self, size: usize) -> Result<CircularBufferPtr<'_>, CircularBufferError> {
        if size == 0 {
            return Err(CircularBufferError::EmptyAlloc);
        }

        // The allocated space has to be greater than or equal to smallest mini-page size
        assert!(size >= self.free_list.size_classes[self.free_list.size_classes.len() - 1]);

        let (lock_guard, states) = self.lock_states();

        while let Some(ptr) = self.free_list.remove(size) {
            let raw_ptr: *mut u8 = ptr.as_ptr();

            let old_meta = CircularBuffer::get_meta_from_data_ptr(raw_ptr);

            if self.ptr_is_copy_on_access(raw_ptr) {
                // here we might fail because ptr might be also evicted by evict_n,
                // nevertheless, someone will tombstone it, so we are fine.
                old_meta.states.free_list_to_tombstone();
                // retry
                continue;
            }

            assert!(old_meta.size as usize >= size);

            // we need to ensure while we are allocating this, no one else is evicting it.
            match old_meta.states.state.compare_exchange_weak(
                MetaState::FreeListed.into(),
                MetaState::NotReady.into(),
                Ordering::AcqRel,
                Ordering::Relaxed,
            ) {
                Ok(_) => {
                    return Ok(CircularBufferPtr::new(raw_ptr));
                }
                Err(_) => {
                    continue;
                }
            };
        }

        let logical_remaining = self.capacity - (states.tail_addr() - states.head_addr()); // Total amount of un-used memory
        let physical_remaining = self.capacity - (states.tail_addr & (self.capacity - 1)); // Total amount of contiguous memory starting from the tail

        let aligned_size = align_up(size, CB_ALLOC_META_SIZE);
        let required = aligned_size + std::mem::size_of::<AllocMeta>();

        if logical_remaining < required {
            return Err(CircularBufferError::Full);
        }

        if physical_remaining < required {
            // fill the remaining physical space with a tombstone meta.
            assert!(physical_remaining >= CB_ALLOC_META_SIZE);
            let physical_addr = self.logical_to_physical(states.tail_addr);
            let meta = AllocMeta::new((physical_remaining - CB_ALLOC_META_SIZE) as u32, true);
            unsafe {
                physical_addr.cast::<AllocMeta>().write(meta);
            }
            states.tail_addr += physical_remaining;
            std::mem::drop(lock_guard);
            return self.alloc(size);
        }

        let meta = AllocMeta::new(aligned_size as u32, false);

        unsafe {
            let physical_addr = self.logical_to_physical(states.tail_addr);
            physical_addr.cast::<AllocMeta>().write(meta);
        }
        let return_addr = states.tail_addr + std::mem::size_of::<AllocMeta>();
        states.tail_addr += required;

        let ptr = CircularBufferPtr::new(self.logical_to_physical(return_addr));
        Ok(ptr)
    }

    fn logical_to_physical(&self, addr: usize) -> *mut u8 {
        let offset = addr & (self.capacity - 1);
        unsafe { self.data_ptr.add(offset) }
    }

    fn debug_check_ptr_is_from_me(&self, ptr: *mut u8) {
        let offset = ptr as usize - self.data_ptr as usize;
        debug_assert!(offset <= self.capacity);
    }

    /// Returns whether the pointer is inside copy-on-access region.
    /// Useful to detect if the ptr is about to be evicted.
    ///
    /// If a ptr is close to head, it is copy on access.
    /// If a ptr is close to tail, it is inplace updatable.
    pub fn ptr_is_copy_on_access(&self, ptr: *mut u8) -> bool {
        let distance = self.distance_to_tail(ptr);
        distance >= self.copy_on_access_threshold
    }

    fn distance_to_tail(&self, ptr: *mut u8) -> usize {
        let ptr_usize = ptr as usize;
        let tail_ptr = self.logical_to_physical(self.get_fuzzy_tail_addr());
        let tail_usize = tail_ptr as usize;

        if tail_usize >= ptr_usize {
            tail_usize - ptr_usize
        } else {
            self.capacity - (ptr_usize - tail_usize)
        }
    }

    fn get_fuzzy_tail_addr(&self) -> usize {
        unsafe { &*self.states.get() }.tail_addr()
    }

    /// This is used to sanity check that
    /// the given address has already been tombstoned, if not, return false.
    ///
    /// # Safety
    /// The addr must be allocated by this buffer.
    #[allow(dead_code)]
    pub(crate) unsafe fn addr_is_tombstoned(addr: *mut u8) -> bool {
        let meta = CircularBuffer::get_meta_from_data_ptr(addr);

        meta.states.is_tombstoned()
    }

    /// Deallocates the given address.
    /// Deallocate is mandatory before the buffer being dropped.
    ///
    /// It panics if the ptr is already dealloced, so double free is not allowed.
    ///
    /// ```
    /// use bf_tree::circular_buffer::CircularBuffer;
    /// let mut buffer = CircularBuffer::new(4096 * 2, 0.1, 64, 1952, 4096, 32, None, false);
    ///
    /// let allocated = buffer.alloc(128);
    /// let ptr = allocated.unwrap().as_ptr();
    ///
    /// let v = unsafe{ buffer.acquire_exclusive_dealloc_handle(ptr).unwrap() };
    /// buffer.dealloc(v); // dealloc is mandatory before buffer being dropped.
    /// ```
    ///
    #[cfg_attr(feature = "tracing", tracing::instrument)]
    pub fn dealloc(&self, ptr: TombstoneHandle) {
        self.dealloc_inner(ptr, true);
    }

    fn dealloc_inner(&self, ptr: TombstoneHandle, add_to_freelist: bool) {
        self.debug_check_ptr_is_from_me(ptr.as_ptr());
        let ptr = ptr.into_ptr();
        let meta = CircularBuffer::get_meta_from_data_ptr(ptr);

        if !add_to_freelist || self.ptr_is_copy_on_access(ptr) {
            meta.states.to_tombstone();
            return;
        }

        // we don't want to pollute dealloc with a Result<>, so if there's contention, we don't add it to free list.
        match self.free_list.try_add(ptr, meta.size as usize) {
            Ok(_lock) => {
                meta.states.to_freelist();
            }
            Err(_) => {
                meta.states.to_tombstone();
            }
        }
    }

    /// Check if the ptr is accessible.
    ///
    /// # Safety
    /// The ptr must be allocated by this buffer.
    pub unsafe fn check_ptr_is_ready(ptr: *mut u8) {
        let meta = CircularBuffer::get_meta_from_data_ptr(ptr);

        assert!(meta.states.state() == MetaState::Ready);
    }

    /// Set the ptr's state to be tombstoning, no future access is allowed, no concurrent tombstoning is allowed.
    /// This is the required call before you can deallocate the ptr.
    ///
    /// Returns the handle that you can use to deallocate the ptr.
    ///     Or Err if contention happened.
    ///
    /// This call is necessary because two concurrent threads can deallocating the same ptr at the same time:
    ///    1. thread A deallocates the ptr normally,
    ///    2. thread B deallocates the ptr by calling evict_n
    ///
    /// This causes contention and unnecessary complexity to handle the race.
    /// It is possible for user to coordinate the two threads, but I feel like it is better to handle it in the library.
    /// I'm not 100% sure this is the best way to do it. If you are reading this, it's a good time to revisit the design.
    ///
    /// Some other thoughts:
    /// This is essentially a x-lock, whoever wins gets to deallocate/evict.
    /// Why not directly expose a locking API?
    ///     While it is possible, I don't like it because we will have too many places to lock.
    ///     In a complex system like bf-tree, the more place to lock, the higher mental burden to the maintainer.
    ///     I want bf-tree to be simple to maintain.
    ///
    /// The next question is: if we don't want so many places to lock, why do we have to lock here?
    /// Why not directly expose the raw bare minimal API, and let users to coordinate the locking?
    /// Readers of this comment should think carefully and consider it as a refactoring opportunity.
    ///
    /// The very high level question here is: what is the safest and efficient interface of a circular buffer that serves our use case?
    ///
    /// # Safety
    /// The ptr must be allocated by this buffer.
    pub unsafe fn acquire_exclusive_dealloc_handle(
        &self,
        ptr: *mut u8,
    ) -> Result<TombstoneHandle, CircularBufferError> {
        self.debug_check_ptr_is_from_me(ptr);

        let meta = CircularBuffer::get_meta_from_data_ptr(ptr);

        if meta.states.try_begin_tombstone() {
            Ok(TombstoneHandle { ptr })
        } else {
            Err(CircularBufferError::WouldBlock)
        }
    }

    /// Returns an iterator that allows you to iterate over the allocated items in the buffer, from head to tail.
    /// Useful for sanity check.
    fn iter(&self) -> Result<AllocatedIter<'_>, CircularBufferError> {
        let (lock, states) = self.try_get_states()?;
        Ok(AllocatedIter {
            _lock: lock,
            buffer: self,
            head_addr: states.head_addr(),
            tail_addr: states.tail_addr(),
        })
    }

    /// Evict n items from the buffer, calling callback on each item,
    /// and returning (elements that is evicted from callback, the number of bytes the head advanced).
    /// This is necessary when the buffer is full, i.e., failed to allocate a new item.
    ///
    ///
    /// The call back is called on each item.
    /// The input handle gives excluesive access to the item, i.e., no other thread can deallocate/evict it.
    /// If you failed to evict the item, return Err, and eviction will release the handle and restart the eviction again.
    ///
    ///
    /// ```
    /// use bf_tree::circular_buffer::CircularBuffer;
    /// let mut buffer = CircularBuffer::new(1024 * 2, 0.1, 64, 256, 1024, 32, None, true);
    ///
    /// for _i in 0..7 {
    ///     let alloc = buffer.alloc(256).unwrap();
    ///     unsafe { *alloc.as_ptr() = 42 };
    ///     drop(alloc);
    /// }
    ///
    /// let not_allocated = buffer.alloc(400);
    /// assert!(not_allocated.is_err());
    /// drop(not_allocated);
    ///
    /// buffer.evict_n(
    ///     usize::MAX,
    ///     |h| {
    ///         let ptr = h.as_ptr();
    ///         assert_eq!(unsafe { *ptr }, 42);
    ///         Ok(h)
    ///     },
    /// );
    ///
    /// let allocated = buffer.alloc(400).unwrap();
    /// let ptr = allocated.as_ptr();
    /// drop(allocated);
    /// let v = unsafe { buffer.acquire_exclusive_dealloc_handle(ptr).unwrap() };
    /// buffer.dealloc(v);
    /// ```
    pub fn evict_n<T>(&self, n: usize, mut callback: T) -> Result<u32, CircularBufferError>
    where
        T: FnMut(TombstoneHandle) -> Result<TombstoneHandle, TombstoneHandle>,
    {
        let mut cur_n = 0;
        let mut cur_evicted = 0;
        while cur_n < n {
            let evicted = self.evict_one(&mut callback);
            match evicted {
                None => return Ok(cur_evicted),
                Some(v) => {
                    cur_evicted += v;
                    cur_n += 1;
                }
            }
        }
        Ok(cur_evicted)
    }

    fn get_meta(&self, logical_address: usize) -> &AllocMeta {
        let ptr = self.logical_to_physical(logical_address);
        self.debug_check_ptr_is_from_me(ptr);
        let meta_ptr = ptr.cast::<AllocMeta>();
        unsafe { &*meta_ptr }
    }

    fn get_meta_from_data_ptr<'a>(data_ptr: *mut u8) -> &'a AllocMeta {
        debug_assert_eq!(data_ptr as usize % 8, 0);
        let meta_ptr = unsafe { data_ptr.sub(CB_ALLOC_META_SIZE) } as *mut AllocMeta;
        unsafe { &*meta_ptr }
    }

    #[cfg_attr(feature = "tracing", tracing::instrument)]
    fn try_bump_head_address_to_evicting_addr(
        &self,
        states: &mut States,
    ) -> Result<u32, CircularBufferError> {
        let mut head_addr = states.head_addr();
        let old_addr = head_addr;
        let evicting_addr = states.evicting_addr;
        while head_addr < evicting_addr {
            let meta = self.get_meta(head_addr);
            if !meta.states.is_evicted() {
                #[cfg(all(feature = "shuttle", test))]
                {
                    shuttle::thread::yield_now();
                }
                return Err(CircularBufferError::WouldBlock);
            }

            let to_add = meta.size as usize + CB_ALLOC_META_SIZE;
            states.head_addr.fetch_add(to_add, Ordering::Relaxed);
            head_addr += to_add;
        }
        Ok((head_addr - old_addr) as u32)
    }

    /// Drain the buffer, calling the callback on each item.
    pub fn drain<T>(&self, mut callback: T)
    where
        T: FnMut(TombstoneHandle) -> Result<TombstoneHandle, TombstoneHandle>,
    {
        loop {
            let evicted = self.evict_one(&mut callback);
            if evicted.is_none() {
                break;
            }
        }
        let backoff = Backoff::new();
        let (_lock, states) = self.lock_states();
        loop {
            if self.try_bump_head_address_to_evicting_addr(states).is_ok() {
                assert_eq!(states.evicting_addr, states.head_addr());
                assert_eq!(states.head_addr(), states.tail_addr());
                return;
            } else {
                backoff.snooze();
            }
        }
    }

    /// Evict one element from the buffer, it never fails.
    /// Returns the number of bytes the head advanced.
    /// Return None if the buffer is empty.
    ///
    /// This is a complex function, the design goal is to not holding a lock while waiting for IO.
    /// This is two step process:
    /// (1) take the lock and make the reservation: bump the evicting address
    /// (2) evict the data, potentially long running IO call.
    /// (3) finish the reservation: bump the head address to the evicting address
    pub fn evict_one<T>(&self, callback: &mut T) -> Option<u32>
    where
        T: FnMut(TombstoneHandle) -> Result<TombstoneHandle, TombstoneHandle>,
    {
        let (start_addr, end_addr) = {
            let (lock, states) = self.lock_states();

            let evicting_addr = states.evicting_addr;

            if evicting_addr == states.tail_addr() {
                // we are behind schedule here, should we call bump head address?
                #[cfg(all(feature = "shuttle", test))]
                {
                    shuttle::thread::yield_now();
                }
                return None;
            }

            let evicting_meta = self.get_meta(evicting_addr);
            let size = evicting_meta.size as usize;

            let advance = size + CB_ALLOC_META_SIZE;

            states.evicting_addr += advance;
            drop(lock);
            (evicting_addr, evicting_addr + advance)
        };

        let meta = self.get_meta(start_addr);
        let data_ptr = meta.data_ptr();

        let backoff = Backoff::new();

        // evict the data using the callback, IO long running call.
        loop {
            let h = unsafe { self.acquire_exclusive_dealloc_handle(data_ptr) };
            match h {
                Ok(v) => {
                    match callback(v) {
                        Ok(h) => {
                            self.dealloc_inner(h, false);
                            meta.states.tombstone_to_evicted();
                            break;
                        }
                        Err(h) => {
                            drop(h);
                            backoff.spin();
                        }
                    };
                }
                Err(_) => {
                    let state = meta.states.state();

                    if state == MetaState::NotReady {
                        // do nothing and wait for the ptr to be ready.
                    } else {
                        if state == MetaState::Tombstone {
                            meta.states.tombstone_to_evicted();
                            break;
                        }
                        if state == MetaState::FreeListed {
                            let found =
                                self.free_list.find_and_remove(data_ptr, meta.size as usize);
                            if found {
                                meta.states.free_list_to_tombstone();
                                meta.states.tombstone_to_evicted();
                                break;
                            }
                        }
                    }
                    backoff.spin();
                }
            }
        }

        let (_lock, states) = self.lock_states();
        _ = self.try_bump_head_address_to_evicting_addr(states);
        Some((end_addr - start_addr) as u32)
    }
}

/// An iterator that allows you to iterate over the allocated items in the buffer, from head to tail.
/// This iterator holds an lock on the entire buffer, which prevents concurrent allocation/eviction.
/// Use it wisely.
///
/// The only proper use case I can think of is to sanity check every allocated item in the buffer.
struct AllocatedIter<'a> {
    _lock: MutexGuard<'a, ()>,
    buffer: &'a CircularBuffer,
    head_addr: usize,
    tail_addr: usize,
}

impl<'a> Iterator for AllocatedIter<'a> {
    type Item = &'a AllocMeta;

    fn next(&mut self) -> Option<Self::Item> {
        if self.head_addr == self.tail_addr {
            return None;
        }

        let meta = self.buffer.get_meta(self.head_addr);

        let size = meta.size as usize;
        let advance = size + CB_ALLOC_META_SIZE;

        self.head_addr += advance;

        Some(meta)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{BfTree, Config};
    use rstest::rstest;

    #[rstest]
    #[case(64, 1952, 4096)] // 1 leaf page = 1 disk page
    #[case(3072, 3072, 8192)] // 1 leaf page = 1 disk page, uniform record size
    #[case(64, 2048, 16384)] // 1 leaf page = 4 disk page
    fn test_circular_buffer_initialization(
        #[case] min_record_size: usize,
        #[case] max_record_size: usize,
        #[case] leaf_page_size: usize,
    ) {
        let capacity = leaf_page_size * 2; // Use a valid power of two and greater than 1024
        let buffer = CircularBuffer::new(
            capacity,
            0.1,
            min_record_size,
            max_record_size,
            leaf_page_size,
            32,
            None,
            false,
        );

        let (_lock, states) = buffer.try_get_states().unwrap();
        assert_eq!(states.head_addr(), 0);
        assert_eq!(states.tail_addr(), 0);
        assert!(!buffer.data_ptr.is_null());
        assert_eq!(buffer.capacity, capacity);
    }

    #[rstest]
    #[case(64, 1952, 4096, false)] // 1 leaf page = 1 disk page
    #[case(3072, 3072, 8192, false)] // 1 leaf page = 1 disk page, uniform record size
    #[case(64, 2048, 16384, true)] // 1 leaf page = 4 disk page
    fn test_circular_buffer_alloc_and_dealloc(
        #[case] min_record_size: usize,
        #[case] max_record_size: usize,
        #[case] leaf_page_size: usize,
        #[case] pre_allocated_buffer: bool,
    ) {
        let buffer_ptr = if pre_allocated_buffer {
            let layout =
                std::alloc::Layout::from_size_align(leaf_page_size * 2, BUFFER_ALIGNMENT).unwrap();
            let ptr = unsafe { std::alloc::alloc(layout) };
            Some(ptr)
        } else {
            None
        };

        let buffer = CircularBuffer::new(
            leaf_page_size * 2,
            0.1,
            min_record_size,
            max_record_size,
            leaf_page_size,
            32,
            buffer_ptr,
            false,
        );

        // Allocate a page of the smallest/largest mini-page
        let mini_page_size = vec![
            buffer.free_list.size_classes[0],
            buffer.free_list.size_classes[buffer.free_list.size_classes.len() - 1],
        ];

        for i in 0..mini_page_size.len() {
            let size = mini_page_size[i]; // this size cannot be smaller than mini-page size
            let alloc_ptr = buffer.alloc(size).expect("Allocation failed").ptr;
            assert!(!alloc_ptr.is_null());

            unsafe {
                let p = buffer.acquire_exclusive_dealloc_handle(alloc_ptr).unwrap();
                buffer.dealloc(p);
            }

            // Check tombstone
            let meta = CircularBuffer::get_meta_from_data_ptr(alloc_ptr);
            assert!(meta.states.is_tombstoned() || meta.states.is_freelisted());
        }
    }

    #[rstest]
    #[case(32, 1952, 4096)] // 1 leaf page = 1 disk page
    #[case(3072, 3072, 8192)] // 1 leaf page = 1 disk page, uniform record size
    #[case(64, 2048, 16384)] // 1 leaf page = 4 disk page
    fn test_circular_buffer_evict_n(
        #[case] min_record_size: usize,
        #[case] max_record_size: usize,
        #[case] leaf_page_size: usize,
    ) {
        let buffer = CircularBuffer::new(
            leaf_page_size * 2,
            0.1,
            min_record_size,
            max_record_size,
            leaf_page_size,
            32,
            None,
            false,
        );
        let size = buffer.free_list.size_classes[0]; // Smallest mini-page size

        // Allocate and then evict
        let _ = buffer.alloc(size).expect("Allocation failed");
        let bytes_advanced = buffer.evict_n(1, |h| Ok(h)).unwrap() as usize;

        assert_eq!(
            bytes_advanced,
            align_up(size, CB_ALLOC_META_SIZE) + CB_ALLOC_META_SIZE
        );
    }

    #[test]
    fn test_circular_buffer_evict_more_than_present() {
        let buffer = CircularBuffer::new(4096 * 2, 0.1, 64, 1952, 4096, 64, None, true);

        // Evict more items than are in the buffer
        let bytes_advanced = buffer.evict_n(10, |h| Ok(h)).unwrap();
        assert_eq!(bytes_advanced, 0);
    }

    #[test]
    fn test_align_up_function() {
        let addr = 123;
        let align = 8;
        let aligned_addr = align_up(addr, align);

        assert_eq!(aligned_addr % align, 0);
    }

    #[test]
    fn alloc_and_evict() {
        let buffer = CircularBuffer::new(4096 * 2, 0.1, 64, 1952, 4096, 32, None, true);

        // Fill up the circular buffer
        for _i in 0..3 {
            let alloc = buffer.alloc(2048).unwrap();
            unsafe { *alloc.as_ptr() = 42 };
            drop(alloc);
        }

        // New allcation fails
        let not_allocated = buffer.alloc(2048);
        assert!(matches!(not_allocated, Err(CircularBufferError::Full)));
        drop(not_allocated);

        // Evict everything in the circular buffer
        buffer
            .evict_n(usize::MAX, |h| {
                assert_eq!(unsafe { *(h.as_ptr()) }, 42);
                Ok(h)
            })
            .unwrap();

        // Allocation succeeds
        let allocated = buffer.alloc(2048).unwrap();
        let ptr = allocated.as_ptr();
        drop(allocated);
        unsafe {
            let p = buffer.acquire_exclusive_dealloc_handle(ptr).unwrap();
            buffer.dealloc(p);
        }
    }

    #[test]
    fn idential_mini_page_classes() {
        // Create a regular bf-tree and check the BfTree's mini-page classes are identical to its CB's mini-page classes
        let mut config = Config::default();
        config.cb_max_record_size(1928);

        let mut tree = BfTree::with_config(config.clone(), None).unwrap();

        let a = tree.mini_page_size_classes.clone();
        let mut b = tree.storage.circular_buffer.free_list.size_classes.clone();
        b.reverse();
        assert_eq!(a, b);
        drop(tree);

        config.cache_only = true;
        tree = BfTree::with_config(config.clone(), None).unwrap();
        let c = tree.mini_page_size_classes.clone();
        let mut d = tree.storage.circular_buffer.free_list.size_classes.clone();
        d.reverse();
        assert_eq!(c, d);
        assert_eq!(a, c);

        drop(tree);
    }
}
