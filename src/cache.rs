//! Small bounded, thread-safe memo cache shared by the translator and the
//! embedding encoders.
//!
//! Eviction is FIFO (oldest insertion goes first) — reads do not refresh an
//! entry's position. Entries whose key exceeds `max_key_bytes` are never
//! stored, so a client sending very large inputs cannot pin unbounded memory
//! in the cache.
//!
//! Uses a `std::sync::Mutex` because every critical section is a short,
//! non-async map operation; that makes the cache usable from both async code
//! and `spawn_blocking` workers.

use std::collections::{HashMap, VecDeque};
use std::hash::Hash;
use std::sync::Mutex;

pub(crate) struct BoundedCache<K, V> {
    inner: Mutex<Inner<K, V>>,
    capacity: usize,
    max_key_bytes: usize,
}

struct Inner<K, V> {
    map: HashMap<K, V>,
    order: VecDeque<K>,
}

/// Size estimate for a cache key, used for the `max_key_bytes` guard.
pub(crate) trait KeySize {
    fn key_size(&self) -> usize;
}

impl KeySize for String {
    fn key_size(&self) -> usize {
        self.len()
    }
}

impl<K: Eq + Hash + Clone + KeySize, V: Clone> BoundedCache<K, V> {
    pub(crate) fn new(capacity: usize, max_key_bytes: usize) -> Self {
        Self {
            inner: Mutex::new(Inner { map: HashMap::new(), order: VecDeque::new() }),
            capacity,
            max_key_bytes,
        }
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, Inner<K, V>> {
        // A panic while holding this lock cannot leave the map logically
        // inconsistent in a way that matters for a memo cache, so recover.
        self.inner.lock().unwrap_or_else(|e| e.into_inner())
    }

    pub(crate) fn get(&self, key: &K) -> Option<V> {
        self.lock().map.get(key).cloned()
    }

    /// Insert `value` under `key`. No-op if the key is already present, the
    /// key is over the size guard, or the capacity is zero.
    pub(crate) fn insert(&self, key: K, value: V) {
        if self.capacity == 0 || key.key_size() > self.max_key_bytes {
            return;
        }
        let mut inner = self.lock();
        if inner.map.contains_key(&key) {
            return;
        }
        while inner.map.len() >= self.capacity {
            match inner.order.pop_front() {
                Some(old) => {
                    inner.map.remove(&old);
                }
                None => break,
            }
        }
        inner.order.push_back(key.clone());
        inner.map.insert(key, value);
    }

    #[cfg(test)]
    pub(crate) fn len(&self) -> usize {
        self.lock().map.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn evicts_oldest_when_full() {
        let c: BoundedCache<String, u32> = BoundedCache::new(2, 1024);
        c.insert("a".into(), 1);
        c.insert("b".into(), 2);
        c.insert("c".into(), 3);
        assert_eq!(c.len(), 2);
        assert_eq!(c.get(&"a".into()), None);
        assert_eq!(c.get(&"c".into()), Some(3));
    }

    #[test]
    fn skips_oversized_keys() {
        let c: BoundedCache<String, u32> = BoundedCache::new(4, 3);
        c.insert("abcd".into(), 1);
        assert_eq!(c.len(), 0);
        c.insert("abc".into(), 1);
        assert_eq!(c.len(), 1);
    }

    #[test]
    fn duplicate_insert_keeps_first_value() {
        let c: BoundedCache<String, u32> = BoundedCache::new(4, 1024);
        c.insert("a".into(), 1);
        c.insert("a".into(), 2);
        assert_eq!(c.get(&"a".into()), Some(1));
        assert_eq!(c.len(), 1);
    }
}
