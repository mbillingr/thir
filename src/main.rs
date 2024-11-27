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
use crate::ast::{DataType, DefClass, ImplClass};
use crate::ast_to_typeck::{
    build_alts, build_program, build_scheme, build_type, build_typeargs, TEnv,
};
use crate::classes::{ClassEnv, EnvTransformer};
use crate::kinds::Kind;
use crate::predicates::Pred;
use crate::qualified::Qual;
use crate::scheme::Scheme;
use crate::specific_inference::{ti_program, BindGroup, Expl, Program};
use crate::specifics::{add_core_classes, add_num_classes};
use crate::types::{Tycon, Type};
use lalrpop_util::lalrpop_mod;
use std::collections::HashMap;
use std::io::BufRead;

type Result<T> = std::result::Result<T, String>;

fn main() {
    let mut ctx = GlobalContext::new();

    for line in std::io::stdin().lock().lines() {
        let line = line.unwrap();
        let top = grammar::ToplevelParser::new().parse(&line);
        println!("{:?}", top);

        let top = top.unwrap();
        ctx.exec_toplevel(top);
    }
}

struct GlobalContext {
    class_env: ClassEnv,

    // could store these directly inside each class, but this is easier for now.
    // also, i don't think i want `ast::` types inside the "thih" core.
    methods: HashMap<Id, HashMap<Id, (Id, ast::Scheme)>>,

    type_env: TEnv,

    assumptions: Vec<Assump>,
}

impl GlobalContext {
    pub fn new() -> GlobalContext {
        let ce = ClassEnv::default();
        let ce = add_core_classes().apply(&ce).unwrap();
        let ce = add_num_classes().apply(&ce).unwrap();

        let methods = Default::default();

        let mut tenv = HashMap::new();
        tenv.insert("->".into(), Type::t_arrow());
        tenv.insert("Int".into(), Type::t_int());
        tenv.insert("Double".into(), Type::t_double());
        tenv.insert("String".into(), Type::t_string());
        tenv.insert("[]".into(), Type::t_list());

        let assumptions = vec![
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

        GlobalContext {
            class_env: ce,
            methods,
            type_env: tenv,
            assumptions,
        }
    }

    fn exec_toplevel(&mut self, top: ast::TopLevel) {
        match top {
            ast::TopLevel::DefClass(dc) => self.define_class(dc),
            ast::TopLevel::ImplClass(ic) => self.implement_class(ic),
            ast::TopLevel::DataType(dt) => self.define_datatype(dt),
            ast::TopLevel::BindGroup(bg) => self.define_globals(bg),
        }
    }

    fn define_class(&mut self, dc: DefClass) {
        let et = EnvTransformer::add_class(dc.name.clone(), dc.super_classes);
        self.class_env = et.apply(&self.class_env).unwrap();

        let mut local_tenv = self.type_env.clone();
        local_tenv.insert(dc.varname.clone(), Type::TGen(0));
        for (i, mut sc) in dc.methods {
            self.methods
                .entry(dc.name.clone())
                .or_insert(HashMap::new())
                .insert(i.clone(), (dc.varname.clone(), sc.clone()));

            // insert the "self" type as the first generic
            sc.genvars
                .insert(0, (dc.varname.clone(), Kind::Star, vec![dc.name.clone()]));
            let sc = build_scheme(sc, &local_tenv);
            self.assumptions.push(Assump { i, sc });
        }
        println!("{:#?}", self.assumptions);
    }

    fn implement_class(&mut self, ic: ImplClass) {
        let mut required_methods = self.methods.get(&ic.cls).cloned().unwrap_or(HashMap::new());

        let ty = self.type_env.get(&ic.ty).expect("unknown type").clone();
        let et = EnvTransformer::add_inst(vec![], Pred::IsIn(ic.cls, ty.clone()));
        self.class_env = et.apply(&self.class_env).unwrap();

        let mut scenv = self.type_env.clone();

        let mut expls = vec![];
        for mi in ic.methods {
            let name = mi.0;
            let (var, sc) = required_methods.remove(&name).expect("unexpected method");

            scenv.insert(var, ty.clone()); // actually, var is the same for every method

            let alts = build_alts(mi.1, &self.type_env);

            expls.push(Expl(name, build_scheme(sc, &scenv), alts));
        }

        let r = ti_program(
            &self.class_env,
            self.assumptions.clone(),
            &Program(vec![BindGroup(expls, vec![])]),
        );
        println!("{r:#?}");

        if !required_methods.is_empty() {
            panic!("missing method impls: {:?}", required_methods);
        }
    }

    fn define_datatype(&mut self, dt: DataType) {
        let type_arity = dt.genvars.len();
        let kind = Kind::ty_constructor(type_arity);
        let dty = Type::TCon(Tycon(dt.typename.clone(), kind));
        self.type_env.insert(dt.typename.clone(), dty.clone());

        let mut method_tenv = self.type_env.clone();
        let (kinds, preds) = build_typeargs(dt.genvars, &mut method_tenv);

        for (i, params) in dt.constructors {
            let args: Vec<_> = params
                .into_iter()
                .map(|p| build_type(p, &method_tenv))
                .collect();

            // apply the type constructor
            let mut tc_args = vec![Type::Unknown; type_arity];
            for a in args.iter() {
                match a {
                    Type::TGen(k) => tc_args[*k] = a.clone(),
                    _ => todo!("{:?}", a),
                }
            }
            let mut ty = dty.clone();
            for a in tc_args {
                ty = Type::tapp(ty, a)
            }

            // constructor arguments
            for a in args.into_iter().rev() {
                ty = Type::func(a, ty);
            }

            self.assumptions.push(Assump {
                i,
                sc: Scheme::Forall(kinds.clone(), Qual(preds.clone(), ty)),
            });
        }
    }

    fn define_globals(&mut self, bg: ast::BindGroup) {
        let prog = build_program(vec![bg], &self.type_env);
        let r = ti_program(&self.class_env, self.assumptions.clone(), &prog);
        println!("{r:#?}");
        if let Ok(ass) = r {
            self.assumptions.extend(ass)
        }
    }
}

type Int = usize;
type Id = String;
