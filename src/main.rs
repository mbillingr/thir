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

use crate::assumptions::{find, Assump};
use crate::ast::{DataType, DefClass, ImplClass};
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
use std::env;
use std::fs;
use std::io::Write;

type Result<T> = std::result::Result<T, String>;

fn main() -> Result<()> {
    let args: Vec<String> = env::args().collect();
    if args.len() != 2 {
        return Err(format!("Usage: {} <file_path>", args[0]));
    }

    let file_path = &args[1];
    let file_content = fs::read_to_string(file_path).map_err(|e| e.to_string())?;

    let program = grammar::ProgramParser::new()
        .parse(&file_content)
        .map_err(|e| e.to_string())?;

    let mut ctx = GlobalContext::new();
    ctx.init();

    for top in program {
        ctx.exec_toplevel(top)?;
    }

    ctx.eval_expr(ast::Expr::app(
        ast::Expr::Var("main".into()),
        ast::Expr::Lit(ast::Literal::Unit),
    ))?;

    Ok(())
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
        tenv.insert("()".into(), Type::t_unit());
        tenv.insert("->".into(), Type::t_arrow());
        tenv.insert("Int".into(), Type::t_int());
        tenv.insert("Double".into(), Type::t_double());
        tenv.insert("String".into(), Type::t_string());
        tenv.insert("[]".into(), Type::t_list());

        let assumptions = vec![Assump {
            i: "show".into(),
            sc: Scheme::Forall(
                list![Kind::Star],
                Qual(
                    vec![Pred::IsIn("Show".into(), Type::TGen(0))],
                    Type::func(Type::TGen(0), Type::t_string()),
                ),
            ),
        }];

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
            // Add a primitive function for printing strings
            let puts_scm =
                self.build_scheme(grammar::SchemeParser::new().parse("String -> ()").unwrap());

            self.assumptions.push(Assump {
                i: "puts".into(),
                sc: puts_scm,
            });

            self.value_env.insert(
                "puts".into(),
                interpreter::Value::primitive("puts", 1, |args| {
                    let s = args[0].as_string();
                    print!("{}", s);
                    interpreter::Value::Unit
                }),
            );

            // Add a primitive function for reading strings
            let gets_scm =
                self.build_scheme(grammar::SchemeParser::new().parse("() -> String").unwrap());

            self.assumptions.push(Assump {
                i: "gets".into(),
                sc: gets_scm,
            });

            self.value_env.insert(
                "gets".into(),
                interpreter::Value::primitive("gets", 1, |_| {
                    std::io::stdout().flush().unwrap();
                    let mut s = String::new();
                    std::io::stdin().read_line(&mut s).unwrap();
                    interpreter::Value::String(s.trim_end_matches('\n').into())
                }),
            );

            // Add a type class and primitives for arithmetic subtraction
            self.class_env = EnvTransformer::add_class("Sub".into(), vec![])
                .compose(EnvTransformer::add_inst(
                    vec![],
                    Pred::IsIn("Sub".into(), Type::t_int()),
                ))
                .compose(EnvTransformer::add_inst(
                    vec![],
                    Pred::IsIn("Sub".into(), Type::t_double()),
                ))
                .apply(&self.class_env)
                .unwrap();

            let sub_scm = self.build_scheme(
                grammar::SchemeParser::new()
                    .parse("forall (a : Sub) => a -> a -> a")
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
                .compose(EnvTransformer::add_inst(
                    vec![],
                    Pred::IsIn("Zero".into(), Type::t_int()),
                ))
                .compose(EnvTransformer::add_inst(
                    vec![],
                    Pred::IsIn("Zero".into(), Type::t_double()),
                ))
                .apply(&self.class_env)
                .unwrap();

            let zero_scm = self.build_scheme(
                grammar::SchemeParser::new()
                    .parse("forall (a : Zero) => a")
                    .unwrap(),
            );
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

        let mut local_tenv = self.type_env.clone();
        local_tenv.insert(dc.varname.clone(), Type::TGen(0));
        let mut assumptions = vec![];
        for (i, mut sc) in dc.methods {
            self.methods
                .entry(dc.name.clone())
                .or_insert(HashMap::new())
                .insert(i.clone(), (dc.varname.clone(), sc.clone()));

            // insert the "self" type as the first generic
            sc.genvars
                .insert(0, (dc.varname.clone(), Kind::Star, vec![dc.name.clone()]));
            let sc = self.build_scheme(sc);

            if sc.is_constant() {
                return Err("all interface type variables must appear in method arguments".into());
            }

            if find(&i, &self.assumptions).is_ok() {
                return Err(format!("name {i} already used"));
            }

            assumptions.push(Assump { i: i.clone(), sc });
        }

        self.class_env = et.apply(&self.class_env)?;
        for a in assumptions {
            self.value_env
                .insert(a.i.clone(), interpreter::Value::method());
            self.assumptions.push(a);
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
        if self.type_env.contains_key(&dt.typename) {
            return Err(format!("type {} already defined", dt.typename));
        }
        let type_arity = dt.genvars.len();
        let kind = Kind::ty_constructor(type_arity);
        let dty = Type::TCon(Tycon(dt.typename.clone(), kind));
        self.type_env.insert(dt.typename.clone(), dty.clone());

        let mut method_tenv = self.type_env.clone();
        let (kinds, preds) = self.build_typeargs(dt.genvars, &mut method_tenv);

        let backup = std::mem::replace(&mut self.type_env, method_tenv);

        for (i, params) in dt.constructors {
            if find(&i, &self.assumptions).is_ok() {
                return Err(format!("name {i} already used"));
            }

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

        for a in &r {
            if find(&a.i, &self.assumptions).is_ok() {
                return Err(format!("name {} already used", a.i));
            }
        }

        self.assumptions.extend(r);

        interpreter::Context::new(ti).exec_program(&prog, &mut self.value_env);

        Ok(())
    }

    fn eval_expr(&mut self, expr: ast::Expr) -> Result<Box<dyn ToString>> {
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
