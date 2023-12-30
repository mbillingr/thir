use crate::classes::EnvTransformer;
use crate::kinds::Kind;
use crate::predicates::Pred::IsIn;
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
            "->".into(),
            Kind::kfun(Kind::Star, Kind::kfun(Kind::Star, Kind::Star)),
        ))
    }

    /// construct the 2-tuple type constructor
    pub fn t_tuple2() -> Self {
        Type::TCon(Tycon(
            ",".into(),
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

pub fn add_core_classes() -> EnvTransformer {
    use EnvTransformer as ET;
    ET::add_class("Eq".into(), vec![])
        .compose(ET::add_class("Ord".into(), vec!["Eq".into()]))
        .compose(ET::add_class("Show".into(), vec![]))
        .compose(ET::add_class("Read".into(), vec![]))
        .compose(ET::add_class("Bounded".into(), vec![]))
        .compose(ET::add_class("Enum".into(), vec![]))
        .compose(ET::add_class("Functor".into(), vec![]))
        .compose(ET::add_class("Monad".into(), vec![]))
}

pub fn add_num_classes() -> EnvTransformer {
    use EnvTransformer as ET;
    let et = ET::add_class("Num".into(), vec!["Eq".into(), "Show".into()])
        .compose(ET::add_class(
            "Real".into(),
            vec!["Num".into(), "Ord".into()],
        ))
        .compose(ET::add_class("Fractional".into(), vec!["Num".into()]))
        .compose(ET::add_class(
            "Integral".into(),
            vec!["Real".into(), "Enum".into()],
        ))
        .compose(ET::add_class(
            "RealFrac".into(),
            vec!["Real".into(), "Fractional".into()],
        ))
        .compose(ET::add_class("Floating".into(), vec!["Fractional".into()]))
        .compose(ET::add_class(
            "RealFloat".into(),
            vec!["RealFrac".into(), "Floating".into()],
        ));

    et.compose(ET::add_inst(vec![], IsIn("Num".into(), Type::t_int())))
        .compose(ET::add_inst(vec![], IsIn("Num".into(), Type::t_double())))
}
