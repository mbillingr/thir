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
use crate::ast_to_typeck::TEnv;
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
        match rep(&mut ctx, line.unwrap()) {
            Ok(()) => println!("OK"),
            Err(err) => println!("Error: {}", err),
        }
    }
}

fn rep(ctx: &mut GlobalContext, line: String) -> Result<()> {
    let top = grammar::ToplevelParser::new()
        .parse(&line)
        .map_err(|e| e.to_string())?;
    println!("AST: {:?}", top);

    ctx.exec_toplevel(top)
}

pub struct GlobalContext {
    class_env: ClassEnv,

    /// could store these directly inside each class, but this is easier for now.
    /// also, i don't think i want `ast::` types inside the "thih" core.
    methods: HashMap<Id, HashMap<Id, (Id, ast::Scheme)>>,

    type_env: TEnv,

    /// all bindings
    assumptions: Vec<Assump>,
    /// only data constructor bindings
    constructors: Vec<Assump>,

    /// identifiers whose type has been explicitly declared but not yet defined
    free_decls: HashMap<Id, ast::Decl>,
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

        let constructors = vec![];

        let free_decls = HashMap::new();

        GlobalContext {
            class_env: ce,
            methods,
            type_env: tenv,
            assumptions,
            constructors,
            free_decls,
        }
    }

    fn exec_toplevel(&mut self, top: ast::TopLevel) -> Result<()> {
        match top {
            ast::TopLevel::DefClass(dc) => self.define_class(dc),
            ast::TopLevel::ImplClass(ic) => self.implement_class(ic),
            ast::TopLevel::DataType(dt) => self.define_datatype(dt),
            ast::TopLevel::BindGroup(bg) => self.define_globals(bg),
        }
    }

    fn define_class(&mut self, dc: DefClass) -> Result<()> {
        let et = EnvTransformer::add_class(dc.name.clone(), dc.super_classes);
        self.class_env = et.apply(&self.class_env)?;

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
            let sc = self.build_scheme(sc);
            self.assumptions.push(Assump { i, sc });
        }
        println!("{:#?}", self.assumptions);
        Ok(())
    }

    fn implement_class(&mut self, ic: ImplClass) -> Result<()> {
        let mut required_methods = self.methods.get(&ic.cls).cloned().unwrap_or(HashMap::new());

        let ty = self
            .type_env
            .get(&ic.ty)
            .ok_or_else(|| format!("unknown type: {}", ic.ty))?
            .clone();
        let et = EnvTransformer::add_inst(vec![], Pred::IsIn(ic.cls, ty.clone()));
        let class_env = et.apply(&self.class_env)?;

        let mut scenv = self.type_env.clone();

        let mut expls = vec![];
        for mi in ic.methods {
            let name = mi.0;
            let (var, sc) = required_methods
                .remove(&name)
                .ok_or_else(|| format!("unexpected method: {name}"))?;

            scenv.insert(var, ty.clone()); // actually, var is the same for every method

            let alts = self.build_alts(mi.1);

            expls.push(Expl(name, self.build_scheme(sc), alts));
        }

        let r = ti_program(
            &class_env,
            self.assumptions.clone(),
            &Program(vec![BindGroup(expls, vec![])]),
        )?;
        println!("{r:#?}");

        if !required_methods.is_empty() {
            return Err(format!("missing method impls: {:?}", required_methods));
        }

        self.class_env = class_env;

        Ok(())
    }

    fn define_datatype(&mut self, dt: DataType) -> Result<()> {
        let type_arity = dt.genvars.len();
        let kind = Kind::ty_constructor(type_arity);
        let dty = Type::TCon(Tycon(dt.typename.clone(), kind));
        self.type_env.insert(dt.typename.clone(), dty.clone());

        let mut method_tenv = self.type_env.clone();
        let (kinds, preds) = self.build_typeargs(dt.genvars, &mut method_tenv);

        let backup = std::mem::replace(&mut self.type_env, method_tenv);

        for (i, params) in dt.constructors {
            let args: Vec<_> = params.into_iter().map(|p| self.build_type(p)).collect();

            // apply the type-constructor
            let mut ty = dty.clone();
            let tc_args = (0..type_arity).map(|k| Type::TGen(k));
            for a in tc_args {
                ty = Type::tapp(ty, a)
            }

            // constructor-function arguments
            for a in args.into_iter().rev() {
                ty = Type::func(a, ty);
            }

            let assump = Assump {
                i,
                sc: Scheme::Forall(kinds.clone(), Qual(preds.clone(), ty)),
            };
            self.assumptions.push(assump.clone());
            self.constructors.push(assump);
        }

        self.type_env = backup;

        println!("{:#?}", self.assumptions);

        Ok(())
    }

    fn define_globals(&mut self, bg: ast::BindGroup) -> Result<()> {
        let prog = self.build_program(vec![bg]);
        let r = ti_program(&self.class_env, self.assumptions.clone(), &prog)?;
        println!("{r:#?}");
        self.assumptions.extend(r);
        Ok(())
    }
}

type Int = usize;
type Id = String;
