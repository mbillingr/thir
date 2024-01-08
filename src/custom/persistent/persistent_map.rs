use crate::custom::persistent;
use crate::custom::persistent::hamt::Hamt;
use std::borrow::Borrow;
use std::hash::Hash;

#[derive(Debug, PartialEq)]
pub struct PersistentMap<K, T>(Hamt<K, T>);

impl<K: Hash + Eq, T> PersistentMap<K, T> {
    #[inline]
    pub fn new() -> Self {
        PersistentMap(Hamt::new(0, vec![]))
    }

    #[inline]
    pub fn get<Q: ?Sized>(&self, key: &Q) -> Option<&T>
    where
        K: Borrow<Q>,
        Q: Hash + Eq,
    {
        self.get_pair(key).map(|(_, v)| v)
    }

    #[inline]
    pub fn contains_key<Q: ?Sized>(&self, key: &Q) -> bool
    where
        K: Borrow<Q>,
        Q: Hash + Eq,
    {
        self.get_pair(key).is_some()
    }

    #[inline]
    pub fn get_pair<Q: ?Sized>(&self, key: &Q) -> Option<(&K, &T)>
    where
        K: Borrow<Q>,
        Q: Hash + Eq,
    {
        let k = persistent::hash(&key);
        self.0.get(key, k).map(|rc| (&rc.0, &rc.1))
    }

    #[inline]
    pub fn insert(&self, key: K, val: T) -> Self {
        let k = persistent::hash(&key);
        let hamt = self.0.insert(key, val, k);
        PersistentMap(hamt)
    }

    #[inline]
    pub fn remove<Q: ?Sized>(&self, key: &Q) -> Option<Self>
    where
        K: Borrow<Q>,
        Q: Hash + Eq,
    {
        let k = persistent::hash(&key);
        self.0.remove_from_root(key, k).map(Self)
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// Merge two maps.
    /// Returns a new map that contains all entries from two maps.
    /// If a key is present in both maps, the value is taken from `other`.
    #[inline]
    pub fn merge(&self, other: &Self) -> Self {
        PersistentMap(self.0.merge(&other.0))
    }

    /// Set intersection on map keys.
    /// That is, returns a new map that contains all keys that are also in another map.
    #[inline]
    pub fn intersect<U>(&self, other: &PersistentMap<K, U>) -> Self {
        PersistentMap(self.0.intersect(&other.0))
    }

    /// Set difference on map keys.
    /// That is, returns a new map that contains all keys that are not contained in another map.
    #[inline]
    pub fn difference<U>(&self, other: &PersistentMap<K, U>) -> Self {
        PersistentMap(self.0.difference(&other.0))
    }

    /// Symmetric set difference on map keys.
    /// That is, returns a new map that contains all keys that are in either map but not in both.
    #[inline]
    pub fn symmetric_difference(&self, other: &Self) -> Self {
        PersistentMap(self.0.symmetric_difference(&other.0))
    }
}

impl<K, T> Clone for PersistentMap<K, T> {
    fn clone(&self) -> Self {
        PersistentMap(self.0.clone())
    }
}
