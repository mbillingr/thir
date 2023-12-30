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

use crate::classes::ClassEnv;
use crate::kinds::Kind;
use crate::predicates::Pred::IsIn;
use crate::qualified::Qual;
use crate::scheme::Scheme;
use crate::specific_inference::{
    ti_program, Alt, BindGroup, Expl, Expr, Impl, Literal, Pat, Program,
};
use crate::specifics::{add_core_classes, add_num_classes};
use crate::types::Type;

type Result<T> = std::result::Result<T, String>;

fn main() {
    let ce = ClassEnv::default();
    let ce = add_core_classes().apply(&ce).unwrap();
    let ce = add_num_classes().apply(&ce).unwrap();

    let prog = Program(vec![BindGroup(
        vec![Expl(
            "foo".into(),
            Scheme::Forall(
                list![Kind::Star, Kind::Star],
                Qual(
                    vec![IsIn(
                        "Mix".into(),
                        vec![Type::TGen(0), Type::TGen(1), Type::TGen(1)],
                    )],
                    Type::func(Type::TGen(0), Type::func(Type::TGen(1), Type::TGen(1))),
                ),
            ),
            vec![Alt(
                vec![Pat::PVar("a".into()), Pat::PVar("b".into())],
                Expr::Var("b".into()),
            )],
        )],
        vec![],
    )]);

    let r = ti_program(&ce, vec![], &prog);
    println!("{r:#?}")
}

type Int = usize;
type Id = String;
