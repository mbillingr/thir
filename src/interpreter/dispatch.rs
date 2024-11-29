use crate::interpreter::Value;
use crate::type_checker::qualified::Qual;
use crate::type_checker::scheme::Scheme;
use crate::type_checker::types::{Tycon, Type, Tyvar};

pub fn scheme_matches_type(Scheme::Forall(_, Qual(_, ty)): &Scheme, t: &Type) -> bool {
    ty == t
}

pub fn scheme_matches(
    Scheme::Forall(_, Qual(_, ty)): &Scheme,
    args: &[Value],
    ret: Option<Type>,
) -> bool {
    let (param_tys, ret_ty) = ty.fn_types();
    if param_tys.len() != args.len() {
        return false;
    }
    for (t, v) in param_tys.into_iter().zip(args.iter()) {
        match t {
            Type::TVar(Tyvar(name, _)) => {
                if !v.is_a(name) {
                    return false;
                }
            }
            Type::TCon(Tycon(name, _)) => {
                if !v.is_a(name) {
                    return false;
                }
            }
            Type::TApp(_) => {
                if v.as_constructor().is_none() {
                    return false;
                }
                todo!("{t:?} =?= {v:?}")
            }
            Type::TGen(_) => {}
        }
    }

    if let Some(ret) = ret {
        return &ret == ret_ty;
    }

    true
}
