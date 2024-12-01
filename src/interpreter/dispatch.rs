use crate::interpreter::Value;
use crate::type_checker::qualified::Qual;
use crate::type_checker::scheme::Scheme;
use crate::type_checker::types::{Tycon, Type, Tyvar};

pub fn scheme_matches_type(Scheme::Forall(_, Qual(_, ty)): &Scheme, t: &Type) -> bool {
    type_matches_type(ty, t)
}

pub fn type_matches_type(ty: &Type, t: &Type) -> bool {
    ty == t
}

pub fn scheme_matches(Scheme::Forall(_, Qual(_, ty)): &Scheme, args: &[Value]) -> bool {
    let (param_tys, ret_ty) = ty.fn_types();
    if param_tys.len() != args.len() {
        return false;
    }
    for (t, v) in param_tys.into_iter().zip(args.iter()) {
        if !type_matches(t, v) {
            return false;
        }
    }

    true
}

pub fn type_matches(t: &Type, v: &Value) -> bool {
    v.is_a(t)
}
