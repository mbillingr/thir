use crate::substitutions::{Subst, Types};
use crate::types::{Type, Tyvar};

pub fn mgu(a: &Type, b: &Type) -> crate::Result<Subst> {
    use Type::*;
    match (a, b) {
        (TApp(app1), TApp(app2)) => {
            let s1 = mgu(&app1.0, &app2.0)?;
            let s2 = mgu(&s1.apply(&app1.1), &s1.apply(&app2.1))?;
            Ok(s2.compose(&s1))
        }

        (TVar(u), t) => var_bind(u, t),

        (t, TVar(u)) => var_bind(u, t),

        (TCon(tc1), TCon(tc2)) if tc1 == tc2 => Ok(Subst::null_subst()),

        _ => Err(format!("types do not unify: {a:?}, {b:?}"))?,
    }
}

fn var_bind(u: &Tyvar, t: &Type) -> crate::Result<Subst> {
    if let Type::TVar(v) = t {
        if u == v {
            return Ok(Subst::null_subst());
        }
    }

    if t.tv().contains(u) {
        Err("occurs check failed")?
    }

    Ok(Subst::single(u.clone(), t.clone()))
}

pub fn matches(a: &Type, b: &Type) -> crate::Result<Subst> {
    use Type::*;
    match (a, b) {
        (TApp(app1), TApp(app2)) => {
            let sl = matches(&app1.0, &app2.0)?;
            let sr = matches(&app1.1, &app2.1)?;
            sl.merge(&sr)
        }

        (TVar(u), t) => Ok(Subst::single(u.clone(), t.clone())),

        (TCon(tc1), TCon(tc2)) if tc1 == tc2 => Ok(Subst::null_subst()),

        _ => Err("types do not match")?,
    }
}
