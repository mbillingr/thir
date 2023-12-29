/*!
Substitutions associate type variables with types.
!*/

use crate::eq_intersect;
use crate::types::{Type, Tyvar};
use std::rc::Rc;

/// A substitution that associates type variables with types.
#[derive(Debug)]
pub struct Subst(SubstImpl);

/// Implementation of substitution as association list
#[derive(Clone, Debug)]
enum SubstImpl {
    Empty,
    Assoc(Rc<(Tyvar, Type, Self)>),
}

impl Subst {
    pub fn null_subst() -> Self {
        Subst(SubstImpl::empty())
    }

    pub fn single(u: Tyvar, t: Type) -> Self {
        Subst(SubstImpl::empty().assoc(u, t))
    }

    pub fn from_rev_iter(it: impl IntoIterator<Item = (Tyvar, Type)>) -> Self {
        let mut out = SubstImpl::empty();

        for (u, t) in it {
            out = out.assoc(u, t);
        }

        Subst(out)
    }

    pub fn lookup(&self, u: &Tyvar) -> Option<&Type> {
        self.0.lookup(u)
    }

    pub fn apply<U, T: Types<U> + ?Sized>(&self, this: &T) -> U {
        this.apply_subst(self)
    }

    pub fn keys(&self) -> Vec<Tyvar> {
        let mut out = vec![];

        let mut cursor = &self.0;
        while let SubstImpl::Assoc(ass) = cursor {
            let (u, t, nxt) = &**ass;
            cursor = nxt;
            out.push(u.clone());
        }

        out
    }

    /// @@ operator
    pub fn compose(&self, other: &Self) -> Self {
        return Subst(other.0.append_map(self.0.clone(), |t| self.apply(t)));
    }

    pub fn merge(&self, other: &Self) -> crate::Result<Self> {
        for v in eq_intersect(self.keys(), other.keys()) {
            if self.apply(&Type::TVar(v.clone())) != other.apply(&Type::TVar(v)) {
                Err("merge fails")?
            }
        }

        Ok(Subst(self.0.append(other.0.clone())))
    }
}

impl SubstImpl {
    fn empty() -> Self {
        Self::Empty
    }

    fn assoc(self, v: Tyvar, t: Type) -> Self {
        Self::Assoc(Rc::new((v, t, self)))
    }

    pub fn lookup(&self, u: &Tyvar) -> Option<&Type> {
        match self {
            Self::Empty => None,
            Self::Assoc(ass) if &ass.0 == u => Some(&ass.1),
            Self::Assoc(ass) => ass.2.lookup(u),
        }
    }

    pub fn append(&self, other: Self) -> Self {
        self.append_map(other, Clone::clone)
    }

    pub fn append_map(&self, other: Self, f: impl Fn(&Type) -> Type) -> Self {
        let mut out = other;

        let mut cursor = self;
        while let SubstImpl::Assoc(ass) = cursor {
            let (u, t, nxt) = &**ass;
            cursor = nxt;
            out = out.assoc(u.clone(), f(t));
        }

        out
    }
}

/// Interface for applying substitutions to types and other things
pub trait Types<T: ?Sized = Self> {
    /// apply substitution
    fn apply_subst(&self, s: &Subst) -> T;

    /// get type vars
    fn tv(&self) -> Vec<Tyvar>;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kinds::Kind;

    #[test]
    fn lookup_in_subst() {
        let foo = Tyvar("foo".into(), Kind::Star);
        let bar = Tyvar("bar".into(), Kind::Star);

        let s = Subst::single(foo.clone(), Type::t_int());

        assert_eq!(s.lookup(&foo), Some(&Type::t_int()));
        assert_eq!(s.lookup(&bar), None);
    }

    #[test]
    fn from_rev_iter_last_takes_precedence() {
        let foo = Tyvar("foo".into(), Kind::Star);
        let s = Subst::from_rev_iter(vec![
            (foo.clone(), Type::t_int()),
            (foo.clone(), Type::t_char()),
        ]);
        assert_eq!(s.lookup(&foo), Some(&Type::t_char()));
    }

    #[test]
    fn composition() {
        let foo = Tyvar("foo".into(), Kind::Star);
        let tfoo = Type::TVar(foo.clone());
        let s1 = Subst::single(foo.clone(), Type::t_int());
        let bar = Tyvar("bar".into(), Kind::Star);
        let tbar = Type::TVar(bar.clone());
        let s2 = Subst::single(bar.clone(), tfoo);

        let s = Subst::compose(&s1, &s2);
        let app_s = s.apply(&tbar);

        let app_ref = s1.apply(&s2.apply(&tbar));

        assert_eq!(app_ref, Type::t_int());
        assert_eq!(app_s, app_ref);
        assert_eq!(app_s, Type::t_int());
    }

    #[test]
    fn composition_allows_access_to_vars_from_both() {
        let foo = Tyvar("foo".into(), Kind::Star);
        let s1 = Subst::single(foo.clone(), Type::t_int());
        let bar = Tyvar("bar".into(), Kind::Star);
        let s2 = Subst::single(bar.clone(), Type::t_char());

        let s = Subst::compose(&s1, &s2);

        assert_eq!(s.lookup(&foo), Some(&Type::t_int()));
        assert_eq!(s.lookup(&bar), Some(&Type::t_char()));
    }

    #[test]
    fn composition_second_takes_precedence() {
        let foo = Tyvar("foo".into(), Kind::Star);
        let s1 = Subst::single(foo.clone(), Type::t_int());
        let s2 = Subst::single(foo.clone(), Type::t_char());

        let s = Subst::compose(&s1, &s2);

        assert_eq!(s.lookup(&foo), Some(&Type::t_char()));
    }

    #[test]
    fn merge_allows_access_to_vars_from_both() {
        let foo = Tyvar("foo".into(), Kind::Star);
        let s1 = Subst::single(foo.clone(), Type::t_int());
        let bar = Tyvar("bar".into(), Kind::Star);
        let s2 = Subst::single(bar.clone(), Type::t_char());

        let s = Subst::compose(&s1, &s2);

        assert_eq!(s.lookup(&foo), Some(&Type::t_int()));
        assert_eq!(s.lookup(&bar), Some(&Type::t_char()));
    }

    #[test]
    fn merge_substitions_must_agree() {
        let foo = Tyvar("foo".into(), Kind::Star);
        let s1 = Subst::single(foo.clone(), Type::t_int());
        let s2 = Subst::single(foo.clone(), Type::t_char());

        let s = Subst::merge(&s1, &s2);

        assert!(s.is_err());

        let foo = Tyvar("foo".into(), Kind::Star);
        let s1 = Subst::single(foo.clone(), Type::t_int());
        let s2 = Subst::single(foo.clone(), Type::t_int());

        let s = Subst::merge(&s1, &s2);

        assert!(s.is_ok());
    }
}
