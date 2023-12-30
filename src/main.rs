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

use crate::assumptions::Assump;
use crate::classes::ClassEnv;
use crate::kinds::Kind;
use crate::predicates::Pred;
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

    // todo: these should be populated by class declarations
    //       actually, they should be accessed using Expr::Const
    let initial_assumptions = vec![Assump {
        i: "show".into(),
        sc: Scheme::Forall(
            list![Kind::Star],
            Qual(
                vec![Pred::IsIn("Show".into(), Type::TGen(0))],
                Type::func(Type::TGen(0), Type::t_string()),
            ),
        ),
    }];

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
                "ignore-arg".into(),
                Scheme::Forall(
                    list![Kind::Star],
                    Qual(vec![], Type::func(Type::TGen(0), Type::t_int())),
                ),
                vec![Alt(vec![Pat::PWildcard], Expr::Lit(Literal::Int(0)))],
            ),
            Expl(
                "fst".into(),
                Scheme::Forall(
                    list![Kind::Star, Kind::Star],
                    Qual(
                        vec![],
                        Type::func(Type::TGen(0), Type::func(Type::TGen(1), Type::TGen(0))),
                    ),
                ),
                vec![Alt(
                    vec![Pat::PVar("x".into()), Pat::PWildcard],
                    Expr::Var("x".into()),
                )],
            ),
            Expl(
                "snd".into(),
                Scheme::Forall(
                    list![Kind::Star, Kind::Star],
                    Qual(
                        vec![],
                        Type::func(Type::TGen(0), Type::func(Type::TGen(1), Type::TGen(1))),
                    ),
                ),
                vec![Alt(
                    vec![Pat::PWildcard, Pat::PVar("x".into())],
                    Expr::Var("x".into()),
                )],
            ),*/
            /*Expl(
                "a-const".into(),
                Scheme::Forall(list![], Qual(vec![], Type::t_int())),
                vec![Alt(vec![], Expr::Lit(Literal::Int(42)).into())],
            ),*/
            Expl(
                "show-int".into(),
                Scheme::Forall(
                    list![],
                    Qual(vec![], Type::func(Type::t_int(), Type::t_string())),
                ),
                vec![Alt(vec![], Expr::Var("show".into()))],
            ),
            Expl(
                "str42".into(),
                Scheme::Forall(list![], Qual(vec![], Type::t_string())),
                vec![Alt(
                    vec![],
                    Expr::App(
                        Expr::Var("show".into()).into(),
                        Expr::Lit(Literal::Int(42)).into(),
                    ),
                )],
            ),
        ],
        vec![/*vec![
            /*Impl(
                "a-const".into(),
                vec![Alt(vec![], Expr::Lit(Literal::Int(42)).into())],
            ),*/
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
        ]*/],
    )]);

    let r = ti_program(&ce, initial_assumptions, &prog);
    println!("{r:#?}")
}

type Int = usize;
type Id = String;
