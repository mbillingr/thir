// This is "Typying Haskell in Rust", based on "Typing Haskell in Haskell":
// https://web.cecs.pdx.edu/~mpj/thih/thih.pdf?_gl=1*1kpcq97*_ga*MTIwMTgwNTIxMS4xNzAyMzAzNTg2*_ga_G56YW5RFXN*MTcwMjMwMzU4NS4xLjAuMTcwMjMwMzU4NS4wLjAuMA..

mod ambiguity;
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

use crate::ambiguity::Ambiguity;
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
