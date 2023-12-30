// This is "Typying Haskell in Rust", based on "Typing Haskell in Haskell":
// https://web.cecs.pdx.edu/~mpj/thih/thih.pdf?_gl=1*1kpcq97*_ga*MTIwMTgwNTIxMS4xNzAyMzAzNTg2*_ga_G56YW5RFXN*MTcwMjMwMzU4NS4xLjAuMTcwMjMwMzU4NS4wLjAuMA..

mod assumptions;
mod classes;
mod instantiate;
mod kinds;
mod lists;
mod predicates;
mod qualified;
mod scheme;
mod specific_inference;
mod specifics;
mod substitutions;
mod type_inference;
mod types;
mod unification;

use crate::assumptions::Assump;
use crate::classes::{ClassEnv, EnvTransformer};
use crate::kinds::{HasKind, Kind};
use crate::lists::{eq_diff, List};
use crate::predicates::{match_pred, mgu_pred};
use crate::qualified::Qual;
use crate::scheme::Scheme;
use crate::specific_inference::{
    ti_program, Alt, BindGroup, Expl, Expr, Impl, Literal, Pat, Program,
};
use crate::specifics::{add_core_classes, add_num_classes};
use crate::substitutions::{Subst, Types};
use crate::type_inference::TI;
use crate::types::{Type, Tyvar};
use crate::unification::{matches, mgu};
use predicates::{Pred, Pred::IsIn};
use std::fmt::{Debug, Formatter};
use std::iter::once;
use std::rc::Rc;

type Result<T> = std::result::Result<T, String>;

fn main() {
    let ce = ClassEnv::default();
    let ce = add_core_classes().apply(&ce).unwrap();
    let ce = add_num_classes().apply(&ce).unwrap();

    let prog = Program(vec![BindGroup(
        vec![
            /*Expl(
                "foo".into(),
                Scheme::Forall(List::Nil, Qual(vec![], Type::t_int())),
                vec![Alt(vec![], Expr::Var("bar".into()))],
            ),*/
            /*Expl(
                "ident".into(),
                Scheme::Forall(
                    list![Kind::Star],
                    Qual(vec![], Type::func(Type::TGen(0), Type::TGen(0))),
                ),
                vec![Alt(vec![Pat::PVar("x".into())], Expr::Var("x".into()))],
            ),*/
            /*Expl(
                "a-const".into(),
                Scheme::Forall(list![], Qual(vec![], Type::t_int())),
                vec![Alt(vec![], Expr::Lit(Literal::Int(42)).into())],
            ),*/
        ],
        vec![vec![
            // todo: why do the implicits (in particular, the constant) result in generic types?
            Impl(
                "a-const".into(),
                vec![Alt(vec![], Expr::Lit(Literal::Int(42)).into())],
            ),
            /*Impl(
                "bar".into(),
                vec![Alt(
                    vec![],
                    Expr::App(
                        Expr::Var("ident".into()).into(),
                        Expr::Lit(Literal::Int(42)).into(),
                    ),
                )],
            ),
            Impl(
                "baz".into(),
                vec![Alt(
                    vec![],
                    Expr::App(
                        Expr::Var("ident".into()).into(),
                        Expr::Var("ident".into()).into(),
                    ),
                )],
            ),*/
        ]],
    )]);

    let r = ti_program(&ce, vec![], &prog);
    println!("{r:#?}")
}

type Int = usize;
type Id = String;

fn enum_id(n: Int) -> Id {
    format!("v{n}")
}

fn find<'a>(i: &Id, ass: impl IntoIterator<Item = &'a Assump>) -> Result<&'a Scheme> {
    for a in ass {
        if &a.i == i {
            return Ok(&a.sc);
        }
    }
    Err(format!("unbound identifier: {i}"))
}

struct Ambiguity(Tyvar, Vec<Pred>);

fn ambiguities(ce: &ClassEnv, vs: Vec<Tyvar>, ps: &[Pred]) -> Vec<Ambiguity> {
    let mut out = vec![];
    for v in eq_diff(ps.tv(), vs) {
        let ps_ = ps.iter().filter(|p| p.tv().contains(&v)).cloned().collect();
        out.push(Ambiguity(v, ps_))
    }
    out
}

const NUM_CLASSES: [&str; 7] = [
    "Num",
    "Integral",
    "Floating",
    "Fractional",
    "Real",
    "RealFloat",
    "RealFrac",
];
const STD_CLASSES: [&str; 17] = [
    "Eq",
    "Ord",
    "Show",
    "Read",
    "Bounded",
    "Enum",
    "Ix",
    "Functor",
    "Monad",
    "MonadPlus",
    "Num",
    "Integral",
    "Floating",
    "Fractional",
    "Real",
    "RealFloat",
    "RealFrac",
];

fn candidates(ce: &ClassEnv, Ambiguity(v, qs): &Ambiguity) -> Vec<Type> {
    let is_ = || qs.iter().map(|Pred::IsIn(i, t)| i);
    let ts_: Vec<_> = qs.iter().map(|Pred::IsIn(i, t)| t).collect();

    if !ts_
        .into_iter()
        .all(|t| if let Type::TVar(u) = t { u == v } else { false })
    {
        return vec![];
    }

    if !is_().any(|i| NUM_CLASSES.contains(&i.as_str())) {
        return vec![];
    }

    if !is_().all(|i| STD_CLASSES.contains(&i.as_str())) {
        return vec![];
    }

    let mut out = vec![];
    for t_ in ce.defaults() {
        if is_()
            .map(|i| Pred::IsIn(i.clone(), t_.clone()))
            .all(|p| ce.entail(&[], &p))
        {
            out.push(t_.clone());
        }
    }

    out
}

fn with_defaults<T>(
    f: impl Fn(Vec<Ambiguity>, Vec<Type>) -> T,
    ce: &ClassEnv,
    vs: Vec<Tyvar>,
    ps: &[Pred],
) -> Result<T> {
    let vps = ambiguities(ce, vs, ps);
    let tss = vps.iter().map(|vp| candidates(ce, vp));

    let mut heads = Vec::with_capacity(vps.len());
    for ts in tss {
        heads.push(
            ts.into_iter()
                .next()
                .ok_or_else(|| "cannot resolve ambiguity")?,
        )
    }

    Ok(f(vps, heads))
}

fn defaulted_preds(ce: &ClassEnv, vs: Vec<Tyvar>, ps: &[Pred]) -> Result<Vec<Pred>> {
    with_defaults(
        |vps, ts| vps.into_iter().map(|Ambiguity(_, p)| p).flatten().collect(),
        ce,
        vs,
        ps,
    )
}

fn default_subst(ce: &ClassEnv, vs: Vec<Tyvar>, ps: &[Pred]) -> Result<Subst> {
    with_defaults(
        |vps, ts| Subst::from_rev_iter(vps.into_iter().map(|Ambiguity(v, _)| v).zip(ts).rev()),
        ce,
        vs,
        ps,
    )
}
