// This is "Typying Haskell in Rust", based on "Typing Haskell in Haskell":
// https://web.cecs.pdx.edu/~mpj/thih/thih.pdf?_gl=1*1kpcq97*_ga*MTIwMTgwNTIxMS4xNzAyMzAzNTg2*_ga_G56YW5RFXN*MTcwMjMwMzU4NS4xLjAuMTcwMjMwMzU4NS4wLjAuMA..

mod ambiguity;
mod assumptions;
mod ast;
mod ast_to_typeck;
mod classes;
mod instantiate;
mod interpreter;
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
use crate::ast::{DataType, DefClass, Expr, ImplClass};
use crate::ast_to_typeck::TEnv;
use crate::classes::{ClassEnv, EnvTransformer};
use crate::kinds::Kind;
use crate::predicates::Pred;
use crate::qualified::Qual;
use crate::scheme::Scheme;
use crate::specific_inference::{ti_expr, ti_program, BindGroup, Expl, Program};
use crate::specifics::{add_core_classes, add_num_classes};
use crate::type_inference::TI;
use crate::types::{Tycon, Type};
use lalrpop_util::lalrpop_mod;
use std::collections::HashMap;
use std::io::{BufRead, Write};

type Result<T> = std::result::Result<T, String>;

fn main() {
    let mut ctx = GlobalContext::new();
    ctx.init();

    print!("> ");
    std::io::stdout().flush().unwrap();

    for line in std::io::stdin().lock().lines() {
        match rep(&mut ctx, line.unwrap()) {
            Ok(res) => print!("{}", res.to_string()),
            Err(err) => println!("Error: {}", err),
        }

        print!("> ");
        std::io::stdout().flush().unwrap();
    }
}

fn rep(ctx: &mut GlobalContext, line: String) -> Result<Box<dyn ToString>> {
    let cmd = grammar::ReplParser::new()
        .parse(&line)
        .map_err(|e| e.to_string())?;

    ctx.handle_cmd(cmd)
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

    value_env: interpreter::Env,
}

impl GlobalContext {
    pub fn new() -> GlobalContext {
        let ce = ClassEnv::default();
        let ce = add_core_classes().apply(&ce).unwrap();
        let ce = add_num_classes().apply(&ce).unwrap();

        let methods = HashMap::default();

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

        let value_env = HashMap::new();

        let constructors = vec![];

        let free_decls = HashMap::new();

        GlobalContext {
            class_env: ce,
            methods,
            type_env: tenv,
            assumptions,
            constructors,
            free_decls,
            value_env,
        }
    }

    fn init(&mut self) {
        {
            // Add a type class and primitives for arithmetic subtraction
            self.class_env = EnvTransformer::add_class("Sub".into(), vec![])
                .apply(&self.class_env)
                .unwrap();

            let sub_scm = self.build_scheme(
                grammar::SchemeParser::new()
                    .parse("forall a => a -> a -> a")
                    .unwrap(),
            );
            self.assumptions.push(Assump {
                i: "sub".into(),
                sc: sub_scm,
            });

            let sub_int_scm = grammar::SchemeParser::new()
                .parse("Int -> Int -> Int")
                .unwrap();
            let sub_flt_scm = grammar::SchemeParser::new()
                .parse("Double -> Double -> Double")
                .unwrap();

            let mut sub_mth = HashMap::new();
            sub_mth.insert("sub".into(), ("Int".into(), sub_int_scm.clone()));
            sub_mth.insert("sub".into(), ("Double".into(), sub_flt_scm.clone()));
            self.methods.insert("Sub".into(), sub_mth);

            let sub_mth = interpreter::Value::method();
            sub_mth.add_impl(
                self.build_scheme(sub_int_scm),
                interpreter::Value::primitive("i-i", 2, |args| {
                    let a = args[0].as_int();
                    let b = args[1].as_int();
                    interpreter::Value::I64(a - b)
                }),
            );
            sub_mth.add_impl(
                self.build_scheme(sub_flt_scm),
                interpreter::Value::primitive("f-f", 2, |args| {
                    let a = args[0].as_float();
                    let b = args[1].as_float();
                    interpreter::Value::F64(a - b)
                }),
            );

            self.value_env.insert("sub".into(), sub_mth);
        }

        {
            // Add a type class and primitives for zero constants
            self.class_env = EnvTransformer::add_class("Zero".into(), vec![])
                .apply(&self.class_env)
                .unwrap();

            let zero_scm =
                self.build_scheme(grammar::SchemeParser::new().parse("forall a => a").unwrap());
            self.assumptions.push(Assump {
                i: "zero".into(),
                sc: zero_scm,
            });

            let zero_int_scm = grammar::SchemeParser::new().parse("Int").unwrap();
            let zero_flt_scm = grammar::SchemeParser::new().parse("Double").unwrap();

            let mut zero_mth = HashMap::new();
            zero_mth.insert("zero".into(), ("Int".into(), zero_int_scm.clone()));
            zero_mth.insert("zero".into(), ("Double".into(), zero_flt_scm.clone()));
            self.methods.insert("zero".into(), zero_mth);

            let zero_mth = interpreter::Value::method();
            zero_mth.add_impl(self.build_scheme(zero_int_scm), interpreter::Value::I64(0));
            zero_mth.add_impl(
                self.build_scheme(zero_flt_scm),
                interpreter::Value::F64(0.0),
            );
            self.value_env.insert("zero".into(), zero_mth);
        }
    }

    fn handle_cmd(&mut self, cmd: ast::ReplCmd) -> Result<Box<dyn ToString>> {
        match cmd {
            ast::ReplCmd::ShowTypeEnv => Ok(Box::new(format!("{:#?}\n", self.type_env))),
            ast::ReplCmd::ShowAssumptions => Ok(Box::new(format!("{:#?}\n", self.assumptions))),
            ast::ReplCmd::ShowValueEnv => Ok(Box::new(format!("{:#?}\n", self.value_env))),
            ast::ReplCmd::TopLevel(top) => self
                .exec_toplevel(top)
                .map(|_| -> Box<dyn ToString> { Box::new("") }),
            ast::ReplCmd::EvalExpr(expr) => self.eval_expr(expr),
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
            self.assumptions.push(Assump { i: i.clone(), sc });

            self.value_env.insert(i, interpreter::Value::method());
        }
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
            let sc_ = self.with_tyenv(scenv.clone(), |ctx| ctx.build_scheme(sc));

            let alts = self.build_alts(mi.1);

            expls.push(Expl(name, sc_, alts));
        }

        let mut prog = Program(vec![BindGroup(expls, vec![])]);
        let (_, ti) = ti_program(&class_env, self.assumptions.clone(), &prog)?;

        if !required_methods.is_empty() {
            return Err(format!("missing method impls: {:?}", required_methods));
        }

        self.class_env = class_env;

        let ctx = interpreter::Context::new(ti);
        for Expl(name, sc, alts) in prog.0.pop().unwrap().0 {
            let val = ctx.eval_alts(&alts, &self.value_env);
            self.value_env.get(&name).unwrap().add_impl(sc, val);
        }

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
                i: i.clone(),
                sc: Scheme::Forall(kinds.clone(), Qual(preds.clone(), ty)),
            };
            self.assumptions.push(assump.clone());
            self.constructors.push(assump);

            self.value_env.insert(
                i.clone(),
                interpreter::Value::constructor(dt.typename.clone(), i),
            );
        }

        self.type_env = backup;
        Ok(())
    }

    fn define_globals(&mut self, bg: ast::BindGroup) -> Result<()> {
        let prog = self.build_program(vec![bg]);
        let (r, ti) = ti_program(&self.class_env, self.assumptions.clone(), &prog)?;
        self.assumptions.extend(r);

        interpreter::Context::new(ti).exec_program(&prog, &mut self.value_env);

        Ok(())
    }

    fn eval_expr(&mut self, expr: Expr) -> Result<Box<dyn ToString>> {
        let expr = self.build_expr(expr);

        let mut ti = TI::new();
        let (ps, t) = ti_expr(&mut ti, &self.class_env, &self.assumptions, &expr)?;

        let s = ti.get_subst();
        let rs = self.class_env.reduce(&s.apply(&ps))?;

        let t_ = s.apply(&t);

        let value = interpreter::Context::new(ti).eval_expr(&expr, &self.value_env);

        Ok(Box::new(format!("{:?}, where {:?}\n{}\n", t_, rs, value)))
    }
}

type Int = usize;
type Id = String;
