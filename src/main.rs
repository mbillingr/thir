// This is "Typying Haskell in Rust", based on "Typing Haskell in Haskell":
// https://web.cecs.pdx.edu/~mpj/thih/thih.pdf?_gl=1*1kpcq97*_ga*MTIwMTgwNTIxMS4xNzAyMzAzNTg2*_ga_G56YW5RFXN*MTcwMjMwMzU4NS4xLjAuMTcwMjMwMzU4NS4wLjAuMA..

mod ambiguity;
mod assumptions;
mod classes;
mod custom_stuff;
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
use crate::custom_stuff::{ti_module, MethodImpl, Module};
use crate::kinds::Kind;
use crate::predicates::Pred;
use crate::qualified::Qual;
use crate::scheme::Scheme;
use crate::specific_inference::{ti_program, Alt, BindGroup, Expl, Expr, Literal, Pat, Program};
use crate::specifics::{add_core_classes, add_num_classes};
use crate::types::Type;

type Result<T> = std::result::Result<T, String>;

fn main() {
    let ce = ClassEnv::default();
    let ce = add_core_classes().apply(&ce).unwrap();
    let ce = add_num_classes().apply(&ce).unwrap();

    let ce = EnvTransformer::add_class("ShowTy".into(), vec![])
        .apply(&ce)
        .unwrap();

    let ce = EnvTransformer::add_inst(vec![], Pred::IsIn("ShowTy".into(), Type::t_int()))
        .apply(&ce)
        .unwrap();

    let ce = EnvTransformer::add_inst(vec![], Pred::IsIn("ShowTy".into(), Type::t_char()))
        .apply(&ce)
        .unwrap();

    let initial_assumptions = vec![
        Assump {
            i: "show".into(),
            sc: Scheme::Forall(
                list![Kind::Star],
                Qual(
                    vec![Pred::IsIn("Show".into(), Type::TGen(0))],
                    Type::func(Type::TGen(0), Type::t_string()),
                ),
            ),
        },
        // should be added by the ShowTy class definition
        Assump {
            i: "show-ty".into(),
            sc: Scheme::Forall(
                list![Kind::Star],
                Qual(
                    vec![Pred::IsIn("ShowTy".into(), Type::TGen(0))],
                    Type::func(Type::TGen(0), Type::t_string()),
                ),
            ),
        },
    ];

    let module = Module {
        impls: vec![
            // impl ShowTy for Int: show-ty
            MethodImpl {
                method: "show-ty".into(),
                ty: Type::t_int(),
                alts: vec![Alt(
                    vec![Pat::PVar("x".into())],
                    //Expr::Lit(Literal::Str("Int".into())),  // correct version
                    Expr::App(
                        // infinitely recursive version
                        Expr::Var("show-ty".into()).into(),
                        Expr::Var("x".into()).into(),
                    ),
                )],
            },
        ],
        free_bindings: BindGroup(
            vec![Expl(
                "int-ty-str".into(),
                Scheme::Forall(list![], Qual(vec![], Type::t_string())),
                vec![Alt(
                    vec![],
                    Expr::App(
                        Expr::Var("show-ty".into()).into(),
                        Expr::Annotate(Type::t_int(), Expr::Lit(Literal::Int(0)).into()).into(),
                    ),
                )],
            )],
            vec![],
        ),
    };
    let r = ti_module(&ce, &initial_assumptions, &module);
    println!("{r:#?}");
    return;

    let prog = Program(vec![BindGroup(
        vec![
            /*Expl(
                "show-ty".into(),
                Scheme::Forall(
                    list![],
                    Qual(
                        vec![Pred::IsIn("ShowTy".into(), Type::t_int())],
                        Type::func(Type::t_int(), Type::t_string()),
                    ),
                ),
                vec![Alt(
                    vec![Pat::PWildcard],
                    Expr::Lit(Literal::Str("Int".into())),
                )],
            ),*/
            Expl(
                "int-ty-str".into(),
                Scheme::Forall(list![], Qual(vec![], Type::t_string())),
                vec![Alt(
                    vec![],
                    Expr::App(
                        Expr::Var("show-ty".into()).into(),
                        Expr::Annotate(Type::t_int(), Expr::Lit(Literal::Int(0)).into()).into(),
                    ),
                )],
            ),
            Expl(
                "char-ty-str".into(),
                Scheme::Forall(list![], Qual(vec![], Type::t_string())),
                vec![Alt(
                    vec![],
                    Expr::App(
                        Expr::Var("show-ty".into()).into(),
                        Expr::Lit(Literal::Char('x')).into(),
                    ),
                )],
            ),
        ],
        vec![],
    )]);

    let r = ti_program(&ce, initial_assumptions, &prog);
    println!("{r:#?}")
}

type Int = usize;
type Id = String;
