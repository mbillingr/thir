//! Define types specific to the language.

use crate::type_checker::kinds::Kind;
use crate::type_checker::types::{Tycon, Type};

impl Type {
    /// construct the unit type
    pub fn t_unit() -> Self {
        Type::TCon(Tycon("()".into(), Kind::Star))
    }

    /// construct the character type
    pub fn t_char() -> Self {
        Type::TCon(Tycon("Char".into(), Kind::Star))
    }

    /// construct the string type
    pub fn t_string() -> Self {
        Type::TCon(Tycon("String".into(), Kind::Star))
    }

    /// construct the boolean type
    pub fn t_bool() -> Self {
        Type::TCon(Tycon("Bool".into(), Kind::Star))
    }

    /// construct the int type
    pub fn t_int() -> Self {
        Type::TCon(Tycon("Int".into(), Kind::Star))
    }

    /// construct the floating point type
    pub fn t_float() -> Self {
        Type::TCon(Tycon("Double".into(), Kind::Star))
    }

    /// construct the list type constructor
    pub fn t_list() -> Self {
        Type::TCon(Tycon("[]".into(), Kind::kfun(Kind::Star, Kind::Star)))
    }

    /// construct the list type constructor
    pub fn t_array() -> Self {
        Type::TCon(Tycon("Array".into(), Kind::kfun(Kind::Star, Kind::Star)))
    }

    /// construct the dict type constructor
    pub fn t_dict() -> Self {
        Type::TCon(Tycon(
            "Dict".into(),
            Kind::kfun(Kind::Star, Kind::kfun(Kind::Star, Kind::Star)),
        ))
    }

    /// construct the function type constructor
    pub fn t_arrow() -> Self {
        Type::TCon(Tycon(
            "->".into(),
            Kind::kfun(Kind::Star, Kind::kfun(Kind::Star, Kind::Star)),
        ))
    }

    /// construct a function type
    pub fn func(a: Type, b: Type) -> Type {
        Type::tapp(Type::tapp(Type::t_arrow(), a), b)
    }

    /// construct a list type
    pub fn list(t: Type) -> Type {
        Type::tapp(Type::t_list(), t)
    }

    /*/// construct a dict type
    pub fn dict(k: Type, v: Type) -> Type {
        Type::tapp(Type::tapp(Type::t_dict(), k), v)
    }*/
}
