use std::rc::Rc;
use crate::kinds::Kind;
use crate::types::{Tycon, Type};

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

    /// construct the int type
    pub fn t_int() -> Self {
        Type::TCon(Tycon("Int".into(), Kind::Star))
    }

    /// construct the integer type
    pub fn t_integer() -> Self {
        Type::TCon(Tycon("Integer".into(), Kind::Star))
    }

    /// construct the float type
    pub fn t_float() -> Self {
        Type::TCon(Tycon("Float".into(), Kind::Star))
    }

    /// construct the double type
    pub fn t_double() -> Self {
        Type::TCon(Tycon("Double".into(), Kind::Star))
    }

    /// construct the list type constructor
    pub fn t_list() -> Self {
        Type::TCon(Tycon("[]".into(), Kind::kfun(Kind::Star, Kind::Star)))
    }

    /// construct the function type constructor
    pub fn t_arrow() -> Self {
        Type::TCon(Tycon(
            "(->)".into(),
            Kind::kfun(Kind::Star, Kind::kfun(Kind::Star, Kind::Star)),
        ))
    }

    /// construct the 2-tuple type constructor
    pub fn t_tuple2() -> Self {
        Type::TCon(Tycon(
            "(,)".into(),
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

    /// construct a pair type
    pub fn pair(a: Type, b: Type) -> Type {
        Type::tapp(Type::tapp(Type::t_tuple2(), a), b)
    }
}