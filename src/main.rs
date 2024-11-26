// This is "Typying Haskell in Rust", based on "Typing Haskell in Haskell":
// https://web.cecs.pdx.edu/~mpj/thih/thih.pdf?_gl=1*1kpcq97*_ga*MTIwMTgwNTIxMS4xNzAyMzAzNTg2*_ga_G56YW5RFXN*MTcwMjMwMzU4NS4xLjAuMTcwMjMwMzU4NS4wLjAuMA..

mod ambiguity;
mod assumptions;
mod ast;
mod ast_to_typeck;
mod classes;
mod instantiate;
mod kinds;
mod lists;
mod parser_utils;
mod predicates;
mod qualified;
mod scheme;
mod specific_inference;
mod specifics;
mod substitutions;
mod type_inference;
mod types;
mod unification;

lalrpop_mod!(grammar);

use crate::assumptions::Assump;
use crate::ast_to_typeck::{build_program, build_scheme, build_type};
use crate::classes::{ClassEnv, EnvTransformer};
use crate::kinds::Kind;
use crate::predicates::Pred;
use crate::qualified::Qual;
use crate::scheme::Scheme;
use crate::specific_inference::{ti_program, Alt, BindGroup, Expl, Expr, Literal, Program};
use crate::specifics::{add_core_classes, add_num_classes};
use crate::types::{Tycon, Type, Tyvar};
use lalrpop_util::lalrpop_mod;
use std::collections::HashMap;
use std::io::BufRead;

type Result<T> = std::result::Result<T, String>;

fn main() {
    let ce = ClassEnv::default();
    let ce = add_core_classes().apply(&ce).unwrap();
    let ce = add_num_classes().apply(&ce).unwrap();
    let mut ce = ce;

    let mut tenv = HashMap::new();
    tenv.insert("->".into(), Type::t_arrow());
    tenv.insert("Int".into(), Type::t_int());
    tenv.insert("String".into(), Type::t_string());
    tenv.insert("[]".into(), Type::t_list());

    // todo: these should be populated by class declarations
    //       actually, they should be accessed using Expr::Const
    let mut global_assumptions = vec![
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
        Assump {
            i: "abc".into(),
            sc: Scheme::Forall(list![], Qual(vec![], Type::list(Type::t_int()))),
        },
    ];

    for line in std::io::stdin().lock().lines() {
        let line = line.unwrap();
        let top = grammar::ToplevelParser::new().parse(&line);
        println!("{:?}", top);

        match top.unwrap() {
            ast::TopLevel::DefClass(dc) => {
                let et = EnvTransformer::add_class(dc.name.clone(), dc.super_classes);
                ce = et.apply(&ce).unwrap();

                let mut local_tenv = tenv.clone();
                local_tenv.insert(dc.varname.clone(), Type::TGen(0));
                for (i, mut sc) in dc.methods {
                    sc.genvars
                        .insert(0, (dc.varname.clone(), Kind::Star, vec![dc.name.clone()]));
                    let sc = build_scheme(sc, &local_tenv);
                    global_assumptions.push(Assump { i, sc });
                }
                println!("{:#?}", global_assumptions);
            }

            ast::TopLevel::ImplClass(ic) => {
                let ty = tenv.get(&ic.ty).expect("unknown type").clone();
                let et = EnvTransformer::add_inst(vec![], Pred::IsIn(ic.cls, ty));
                ce = et.apply(&ce).unwrap();
                // todo: check method definitions
            }

            ast::TopLevel::DataType(dt) => {
                let dty = Type::TCon(Tycon(dt.typename.clone(), Kind::Star));
                tenv.insert(dt.typename.clone(), dty.clone());
                for (i, params) in dt.constructors {
                    let mut ty = dty.clone();
                    for p in params.into_iter().rev() {
                        ty = Type::func(build_type(p, &tenv), ty);
                    }
                    global_assumptions.push(Assump {
                        i,
                        sc: Scheme::Forall(list![], Qual(vec![], ty)),
                    });
                }
            }

            ast::TopLevel::BindGroup(bg) => {
                let prog = build_program(vec![bg], &tenv);
                let r = ti_program(&ce, global_assumptions.clone(), &prog);
                println!("{r:#?}");
                if let Ok(ass) = r {
                    global_assumptions.extend(ass)
                }
            }
        }
    }

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

    let r = ti_program(&ce, global_assumptions, &prog);
    println!("{r:#?}")
}

type Int = usize;
type Id = String;
