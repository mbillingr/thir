/*!
Types
!*/
use crate::kinds::{HasKind, Kind};
use crate::{Id, Int};
use std::fmt::{Debug, Formatter};
use std::rc::Rc;

/// The type of a value
#[derive(Clone, PartialEq)]
pub enum Type {
    /// A type variable
    TVar(Tyvar),

    /// A type constant
    TCon(Tycon),

    /// A type application (applying a type of kind `k1` to a type of
    /// kind `k1 -> k2` results in a type of kind `k2`.)
    TApp(Rc<(Self, Self)>),

    /// A generic (quantified) type variable
    TGen(Int),
}

impl Debug for Type {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Type::TVar(tv) => write!(f, "{}", tv.0),
            Type::TCon(tc) => write!(f, "{}", tc.0),
            Type::TApp(rc) => {
                write!(f, "(")?;
                dbg_tapp(&rc, f)?;
                write!(f, ")")
            }
            Type::TGen(k) => write!(f, "'{k}"),
        }
    }
}

fn dbg_tapp((c, a): &(Type, Type), f: &mut Formatter<'_>) -> std::fmt::Result {
    match c {
        Type::TApp(rc) => dbg_tapp(rc, f),
        _ => write!(f, "{:?}", c),
    }?;
    write!(f, " {:?}", a)
}

/// A type variable
#[derive(Clone, Debug, PartialEq)]
pub struct Tyvar(pub Id, pub Kind);

/// A type constant
#[derive(Clone, Debug, PartialEq)]
pub struct Tycon(pub Id, pub Kind);

impl Type {
    pub fn tvar(name: Id, kind: Kind) -> Type {
        Type::TVar(Tyvar(name, kind))
    }

    /// construct a type application (convenience method)
    pub fn tapp(a: Type, b: Type) -> Type {
        Type::TApp(Rc::new((a, b)))
    }

    pub fn get_head_name(&self) -> Option<Id> {
        match self {
            Type::TVar(tv) => Some(tv.0),
            Type::TCon(tc) => Some(tc.0),
            Type::TApp(app) => app.0.get_head_name(),
            Type::TGen(_) => None,
        }
    }
}

impl HasKind for Tyvar {
    fn kind(&self) -> crate::Result<&Kind> {
        Ok(&self.1)
    }
}

impl HasKind for Tycon {
    fn kind(&self) -> crate::Result<&Kind> {
        Ok(&self.1)
    }
}

impl HasKind for Type {
    fn kind(&self) -> crate::Result<&Kind> {
        match self {
            Type::TCon(tc) => tc.kind(),
            Type::TVar(u) => u.kind(),
            Type::TApp(app) => match app.0.kind()? {
                Kind::Kfun(kf) => Ok(&kf.1),
                _ => Err("Invalid Kind in TApp")?,
            },
            Type::TGen(_) => panic!("Don't know what to do :(   (maybe ignore somehow?)"),
        }
    }
}
