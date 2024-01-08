use crate::custom::persistent::persistent_map::PersistentMap;
use std::borrow::Borrow;
use std::hash::Hash;

#[derive(Debug, PartialEq)]
pub struct PersistentSet<T>(PersistentMap<T, ()>);

impl<T: Hash + Eq> PersistentSet<T> {
    #[inline]
    pub fn new() -> Self {
        PersistentSet(PersistentMap::new())
    }

    #[inline]
    pub fn contains<Q: ?Sized>(&self, key: &Q) -> bool
    where
        T: Borrow<Q>,
        Q: Hash + Eq,
    {
        self.0.get_pair(key).is_some()
    }

    #[inline]
    pub fn get<Q: ?Sized>(&self, key: &Q) -> Option<&T>
    where
        T: Borrow<Q>,
        Q: Hash + Eq,
    {
        self.0.get_pair(key).map(|(k, _)| k)
    }

    #[inline]
    pub fn insert(&self, key: T) -> Self {
        PersistentSet(self.0.insert(key, ()))
    }

    #[inline]
    pub fn remove<Q: ?Sized>(&self, key: &Q) -> Option<Self>
    where
        T: Borrow<Q>,
        Q: Hash + Eq,
    {
        self.0.remove(key).map(Self)
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.0.len()
    }

    #[inline]
    pub fn union(&self, other: &Self) -> Self {
        PersistentSet(self.0.merge(&other.0))
    }

    #[inline]
    pub fn intersection(&self, other: &Self) -> Self {
        PersistentSet(self.0.intersect(&other.0))
    }

    #[inline]
    pub fn difference(&self, other: &Self) -> Self {
        PersistentSet(self.0.difference(&other.0))
    }

    #[inline]
    pub fn symmetric_difference(&self, other: &Self) -> Self {
        PersistentSet(self.0.symmetric_difference(&other.0))
    }

    /// An iterator visiting all key/value pairs in arbitrary order.
    #[inline]
    pub fn iter(&self) -> impl Iterator<Item = &T> {
        self.0.keys()
    }
}

impl<T> From<PersistentMap<T, ()>> for PersistentSet<T> {
    fn from(map: PersistentMap<T, ()>) -> Self {
        PersistentSet(map)
    }
}

impl<T> From<PersistentSet<T>> for PersistentMap<T, ()> {
    fn from(set: PersistentSet<T>) -> Self {
        set.0
    }
}

impl<T: Eq + Hash> FromIterator<T> for PersistentSet<T> {
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        PersistentSet(iter.into_iter().map(|k| (k, ())).collect())
    }
}
