/*!
Types
!*/

use crate::type_checker::kinds::{HasKind, Kind};
use crate::type_checker::GenId;
use crate::type_checker::Id;
use std::fmt::{Debug, Formatter};
use std::rc::Rc;

/// The type of a value
#[derive(Clone, Eq, Hash, PartialEq)]
pub enum Type {
    /// A type variable
    TVar(Tyvar),

    /// A type constant
    TCon(Tycon),

    /// A type application (applying a type of kind `k1` to a type of
    /// kind `k1 -> k2` results in a type of kind `k2`.)
    TApp(Rc<(Self, Self)>),

    /// A generic (quantified) type variable
    TGen(GenId),
}

impl Debug for Type {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Type::TVar(tv) => write!(f, "{}", tv.0),
            Type::TCon(tc) => write!(f, "{}", tc.0),
            Type::TApp(rc) => {
                write!(f, "(")?;
                write_tapp(&rc.0, &rc.1, f)?;
                write!(f, ")")
            }
            Type::TGen(k) => write!(f, "'{k}"),
        }
    }
}

fn write_tapp(rator: &Type, rand: &Type, f: &mut Formatter<'_>) -> std::fmt::Result {
    match rator {
        Type::TApp(rc) => match &rc.0 {
            Type::TVar(Tyvar(i, _)) | Type::TCon(Tycon(i, _)) if i == "->" => {
                write!(f, "{:?} -> {:?}", rc.1, rand)
            }
            _ => {
                write_tapp(&rc.0, &rc.1, f)?;
                write!(f, " {:?}", rand)
            }
        },
        _ => write!(f, "{:?} {:?}", rator, rand),
    }
}

/// A type variable
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct Tyvar(pub Id, pub Kind);

/// A type constant
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct Tycon(pub Id, pub Kind);

impl Type {
    /// construct a type application (convenience method)
    pub fn tapp(a: Type, b: Type) -> Type {
        Type::TApp(Rc::new((a, b)))
    }

    /// assuming this is a function type, return all types components (argument and return types)
    pub fn fn_types(&self) -> (Vec<&Type>, &Type) {
        let mut args = vec![];
        let mut t = self;
        while let Some((arg, ret)) = t.as_fn() {
            args.push(arg);
            t = ret;
        }
        (args, t)
    }

    /// get argument and return types, if this is a function type
    pub fn as_fn(&self) -> Option<(&Type, &Type)> {
        match self {
            Type::TApp(app1) => match &app1.0 {
                Type::TApp(app2) => match &app2.0 {
                    Type::TCon(Tycon(op, _)) if op == "->" => Some((&app2.1, &app1.1)),
                    _ => None,
                },
                _ => None,
            },
            _ => None,
        }
    }

    pub fn contains(&self, component: &Self) -> bool {
        match self {
            Type::TVar(_) => self == component,
            Type::TCon(_) => self == component,
            Type::TApp(app) => app.0.contains(component) || app.1.contains(component),
            Type::TGen(_) => self == component,
        }
    }

    pub fn find_first_arg_with_genvar(&self, k: GenId) -> Option<usize> {
        let (args, _) = self.fn_types();
        for (i, arg) in args.iter().enumerate() {
            if arg.contains(&Type::TGen(k)) {
                return Some(i);
            }
        }
        None
    }

    /// compare types for equality, considering generics equal to all types
    pub fn soft_eq(&self, ty: &Self) -> bool {
        match (self, ty) {
            (Type::TApp(app1), Type::TApp(app2)) => {
                app1.0.soft_eq(&app2.0) && app1.1.soft_eq(&app2.1)
            }
            (Type::TGen(_), _) => true,
            (_, Type::TGen(_)) => true,
            _ => self == ty,
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
            Type::TGen(_) => Err("unknown kind")?,
        }
    }
}
