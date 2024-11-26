use crate::kinds::Kind;
use crate::lists::List;
use crate::predicates::Pred;
use crate::{ast, predicates, qualified, scheme, specific_inference as si, types, Id};
use std::collections::HashMap;
use std::rc::Rc;

pub type TEnv = HashMap<Id, types::Type>;

pub fn build_program(bgs: Vec<ast::BindGroup>, tyenv: &TEnv) -> si::Program {
    si::Program(
        bgs.into_iter()
            .map(|bg| build_bindgroup(bg, tyenv))
            .collect(),
    )
}

pub fn build_bindgroup(ast::BindGroup(bg): ast::BindGroup, tyenv: &TEnv) -> si::BindGroup {
    let mut expls = vec![];
    let mut impls = vec![];

    for b in bg {
        match b {
            ast::Bind::Explicit(expl) => expls.push(build_expl(expl, tyenv)),
            ast::Bind::Implicit(impl_) => impls.push(vec![build_impl(impl_)]),
            ast::Bind::Mutual(mut_) => {
                impls.push(mut_.into_iter().map(|impl_| build_impl(impl_)).collect())
            }
        }
    }

    si::BindGroup(expls, impls)
}

pub fn build_impl(ast::Impl(id, alts): ast::Impl) -> si::Impl {
    let alts = build_alts(alts, &HashMap::new());
    si::Impl(id, alts)
}

pub fn build_expl(ast::Expl(id, sc, alts): ast::Expl, tyenv: &TEnv) -> si::Expl {
    let sc = build_scheme(sc, tyenv);
    let alts = build_alts(alts, tyenv);
    si::Expl(id, sc, alts)
}

pub fn build_alts(alts: Vec<ast::Alt>, tyenv: &TEnv) -> Vec<si::Alt> {
    alts.into_iter().map(|alt| build_alt(alt, tyenv)).collect()
}

pub fn build_alt(ast::Alt(pats, expr): ast::Alt, tyenv: &TEnv) -> si::Alt {
    let pats = pats.into_iter().map(|pat| build_pat(pat, tyenv)).collect();
    let expr = build_expr(expr, tyenv);
    si::Alt(pats, expr)
}

pub fn build_pat(pat: ast::Pat, tyenv: &TEnv) -> si::Pat {
    match pat {
        ast::Pat::PVar(id) => si::Pat::PVar(id),
        ast::Pat::PWildcard => si::Pat::PWildcard,
        ast::Pat::PAs(id, pat) => si::Pat::PAs(id, Rc::new(build_pat(*pat, tyenv))),
        ast::Pat::PLit(lit) => si::Pat::PLit(lit),
        ast::Pat::PNpk(id, n) => si::Pat::PNpk(id, n),
    }
}

pub fn build_expr(expr: ast::Expr, tyenv: &TEnv) -> si::Expr {
    match expr {
        ast::Expr::Var(id) => si::Expr::Var(id),
        ast::Expr::Lit(lit) => si::Expr::Lit(lit),
        ast::Expr::App(e1, e2) => {
            let e1 = Rc::new(build_expr(*e1, tyenv));
            let e2 = Rc::new(build_expr(*e2, tyenv));
            si::Expr::App(e1, e2)
        }
        ast::Expr::Let(bg, e) => {
            let bg = build_bindgroup(bg, tyenv);
            let e = Rc::new(build_expr(*e, tyenv));
            si::Expr::Let(bg, e)
        }
    }
}

pub fn build_type(ty: ast::Type, tyenv: &HashMap<Id, types::Type>) -> types::Type {
    match ty {
        ast::Type::Named(name) => {
            if let Some(ty) = tyenv.get(&name) {
                ty.clone()
            } else {
                panic!("unknown type name: {}", name)
            }
        }
        ast::Type::Apply(t1, t2) => {
            let t1 = build_type(*t1, tyenv);
            let t2 = build_type(*t2, tyenv);
            types::Type::tapp(t1, t2)
        }
    }
}

pub fn build_scheme(sc: ast::Scheme, tyenv: &HashMap<Id, types::Type>) -> scheme::Scheme {
    let mut tyenv = tyenv.clone();
    let (kinds, preds) = build_typeargs(sc.genvars, &mut tyenv);

    let ty = build_type(sc.ty, &tyenv);

    let qual_ty = qualified::Qual(preds, ty);
    scheme::Scheme::Forall(kinds, qual_ty)
}

pub fn build_typeargs(
    genvars: Vec<(Id, Kind, Vec<Id>)>,
    tyenv: &mut HashMap<Id, types::Type>,
) -> (List<Kind>, Vec<Pred>) {
    let mut kinds = List::Nil;
    let mut idx = 0;
    let mut preds = vec![];

    for (name, kind, constraints) in genvars {
        kinds = kinds.cons(kind);
        tyenv.insert(name, types::Type::TGen(idx));

        for c in constraints {
            let pred = predicates::Pred::IsIn(c, types::Type::TGen(idx));
            preds.push(pred);
        }

        idx += 1;
    }

    (kinds, preds)
}
