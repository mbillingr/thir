use crate::frontend::ast_to_typeck::TEnv;
use crate::frontend::type_inference::{ti_expr, ti_program, BindGroup, Expl, Program};
use crate::frontend::{ast, grammar};
use crate::interpreter;
use crate::type_checker::assumptions::{find, Assump};
use crate::type_checker::classes::{ClassEnv, EnvTransformer};
use crate::type_checker::kinds::Kind;
use crate::type_checker::predicates::Pred;
use crate::type_checker::qualified::Qual;
use crate::type_checker::scheme::Scheme;
use crate::type_checker::type_inference::TI;
use crate::type_checker::types::{Tycon, Type};
use crate::type_checker::Id;
use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::Path;

pub struct Runner {
    pub class_env: ClassEnv,

    /// could store these directly inside each class, but this is easier for now.
    /// also, i don't think i want `ast::` types inside the "thih" core.
    pub methods: HashMap<Id, (Id, HashMap<Id, ast::Scheme>)>,

    pub type_env: TEnv,

    /// all bindings
    pub assumptions: Vec<Assump>,
    /// only data constructor bindings
    pub constructors: Vec<Assump>,

    /// identifiers whose type has been explicitly declared but not yet defined
    pub free_decls: HashMap<Id, ast::Decl>,

    pub value_env: interpreter::Env,
}

macro_rules! super_classes {
    () => {""};
    ($($cls:literal),*) => {
        concat!(": ", $($cls)*)
    };
}

macro_rules! define_class {
    ($ctx:expr, $cls:literal, $($sup:literal)*, $(, $op:literal,  $sig:literal)*) => {
        $ctx.define_class(
            grammar::DefClassParser::new()
                .parse(concat!(
                    "interface ",
                    $cls,
                    " a ", super_classes!($($sup)*), " { ",
                    $("(", $op, ") : ", $sig, ";",)*
                    " }"
                ))
                .unwrap(),
        )
        .unwrap();
    }
}

macro_rules! type_from_value {
    (Bool, $x:expr) => {
        interpreter::Value::as_bool($x)
    };

    (Int, $x:expr) => {
        &*interpreter::Value::as_int($x)
    };

    (Float, $x:expr) => {
        interpreter::Value::as_float($x)
    };

    (String, $x:expr) => {
        interpreter::Value::as_string($x)
    };
}

macro_rules! type_to_value {
    (Bool) => {
        interpreter::Value::Bool
    };

    (Int) => {
        interpreter::Value::int
    };

    (Float) => {
        interpreter::Value::F64
    };

    (String) => {
        interpreter::Value::String
    };
}

macro_rules! define_arithmetic_impl {
    ($ctx:expr, $cls:literal, $ty:tt $(, $op:literal, $rustop:tt)*) => {
        $ctx.primitive_class_impl(
            $cls,
            stringify!($ty),
            vec![$((
                $op,
                concat!(stringify!($ty), " -> ", stringify!($ty), " -> ", stringify!($ty)),
                interpreter::Value::primitive(concat!(stringify!($ty), $op, stringify!($ty)), 2, |args| {
                    let a = type_from_value!($ty, &args[0]);
                    let b = type_from_value!($ty, &args[1]);
                    type_to_value!($ty)(a $rustop b)
                }),
            )),*],
        );
    }
}

macro_rules! define_arithmetic_operator {
    ($ctx:expr, $cls:literal <: $($sup:literal)* $(, $op:literal, $rustop:tt)*) => {
        define_class!($ctx, $cls, $($sup)*, $(, $op, "a -> a -> a")*);
        define_arithmetic_impl!($ctx, $cls, Int $(, $op, $rustop)*);
        define_arithmetic_impl!($ctx, $cls, Float $(, $op, $rustop)*);
    };

    ($ctx:expr, $cls:literal, $($rest:tt)*) => {
        define_arithmetic_operator!($ctx, $cls <:, $($rest)*)
    };
}

macro_rules! define_comparison_impl {
    ($ctx:expr, $cls:literal, $ty:tt $(, $op:literal, $rustop:tt)*) => {
        $ctx.primitive_class_impl(
            $cls,
            stringify!($ty),
            vec![$((
                $op,
                concat!(stringify!($ty), " -> ", stringify!($ty), " -> Bool"),
                interpreter::Value::primitive(concat!(stringify!($ty), $op, stringify!($ty)), 2, |args| {
                    let a = type_from_value!($ty, &args[0]);
                    let b = type_from_value!($ty, &args[1]);
                    type_to_value!(Bool)(a $rustop b)
                }),
            )),*],
        );
    }
}

macro_rules! define_comparison_operator {
    ($ctx:expr, $cls:literal <: $($sup:literal)* $(, $op:literal, $rustop:tt)*) => {
        define_class!($ctx, $cls, $($sup)*, $(, $op, "a -> a -> Bool")*);
        define_comparison_impl!($ctx, $cls, Bool $(, $op, $rustop)*);
        define_comparison_impl!($ctx, $cls, Int $(, $op, $rustop)*);
        define_comparison_impl!($ctx, $cls, Float $(, $op, $rustop)*);
        define_comparison_impl!($ctx, $cls, String $(, $op, $rustop)*);
    };

    ($ctx:expr, $cls:literal, $($rest:tt)*) => {
        define_comparison_operator!($ctx, $cls <:, $($rest)*)
    };
}

impl Runner {
    pub fn new() -> Runner {
        let ce = ClassEnv::default();
        //let ce = add_core_classes().apply(&ce).unwrap();
        //let ce = add_num_classes().apply(&ce).unwrap();

        let methods = HashMap::default();

        let mut tenv = HashMap::new();
        tenv.insert("()".into(), Type::t_unit());
        tenv.insert("->".into(), Type::t_arrow());
        tenv.insert("Bool".into(), Type::t_bool());
        tenv.insert("Int".into(), Type::t_int());
        tenv.insert("Float".into(), Type::t_float());
        tenv.insert("String".into(), Type::t_string());
        tenv.insert("Dict".into(), Type::t_dict());
        tenv.insert(
            "Hasher".into(),
            Type::TCon(Tycon("Hasher".into(), Kind::Star)),
        );

        let assumptions = vec![];

        let value_env = interpreter::Env::default();

        let constructors = vec![];

        let free_decls = HashMap::new();

        Runner {
            class_env: ce,
            methods,
            type_env: tenv,
            assumptions,
            constructors,
            free_decls,
            value_env,
        }
    }

    pub fn init(&mut self) {
        {
            // A List type
            self.define_datatype(
                grammar::DataTypeParser::new()
                    .parse("data [] a = Nil | (::) a [a];")
                    .unwrap(),
            )
            .unwrap();

            // Add a primitive function for debug printing
            self.define_primitive("dbg", "forall a => a -> ()", |args| {
                println!("{:?}", args);
                interpreter::Value::Unit
            });

            // Add a primitive function for printing strings
            self.define_primitive("puts", "String -> ()", |args| {
                let s = args[0].as_string();
                print!("{}", s);
                interpreter::Value::Unit
            });

            // Add a primitive function for reading strings
            self.define_primitive("gets", "() -> String", |_| {
                std::io::stdout().flush().unwrap();
                let mut s = String::new();
                std::io::stdin().read_line(&mut s).unwrap();
                interpreter::Value::String(s.into())
            });

            // Add a primitive function for trimming strings
            self.define_primitive("trim", "String -> String", |args| {
                let s = args[0].as_string().trim().to_string();
                interpreter::Value::String(s.into())
            });

            // Regex string search
            self.define_primitive("str-find-all", "String -> String -> [String]", |args| {
                let pattern = args[0].as_string();
                let haystack = args[1].as_string();

                let re = regex::Regex::new(&pattern).unwrap();

                let parts: Vec<_> = re
                    .find_iter(&haystack)
                    .map(|part| part.as_str().to_string())
                    .map(|part| interpreter::Value::String(part.into()))
                    .collect();

                interpreter::Value::make_list(parts.into_iter())
            });

            // Add a primitive function for converting strings to integers
            self.define_primitive("atoi", "String -> Int", |args| {
                let s = args[0].as_string();
                let i = s.parse::<num::BigInt>().unwrap();
                interpreter::Value::int(i)
            });

            // Add type class for converting values to string
            self.define_class(
                grammar::DefClassParser::new()
                    .parse("interface Show a { show : a -> String; }")
                    .unwrap(),
            )
            .unwrap();

            // Implement Show for all primitives
            let show_fn = interpreter::Value::primitive("show", 1, |args| {
                interpreter::Value::String(args[0].to_string().into())
            });
            self.primitive_class_impl(
                "Show",
                "Bool",
                vec![("show", "Bool -> String", show_fn.clone())],
            );
            self.primitive_class_impl(
                "Show",
                "Int",
                vec![("show", "Int -> String", show_fn.clone())],
            );
            self.primitive_class_impl(
                "Show",
                "Float",
                vec![("show", "Float -> String", show_fn.clone())],
            );
            self.primitive_class_impl(
                "Show",
                "String",
                vec![("show", "String -> String", show_fn.clone())],
            );

            // Concatenation class
            self.define_class(
                grammar::DefClassParser::new()
                    .parse("interface Concatenate a { (++) : a -> a -> a; }")
                    .unwrap(),
            )
            .unwrap();
            // Implement Concatenation for certain primitives
            self.primitive_class_impl(
                "Concatenate",
                "String",
                vec![(
                    "++",
                    "String -> String -> String",
                    interpreter::Value::primitive("s++s", 2, |args| {
                        let a = args[0].as_string();
                        let b = args[1].as_string();
                        interpreter::Value::String(format!("{}{}", a, b).into())
                    })
                    .clone(),
                )],
            );

            define_arithmetic_operator!(self, "Add", "+", +);
            define_arithmetic_operator!(self, "Sub", "-", -);
            define_arithmetic_operator!(self, "Mul", "*", *);
            define_arithmetic_operator!(self, "Div", "/", /);

            define_comparison_operator!(self, "Cmp", "==", ==, "!=", !=);
            define_comparison_operator!(self, "Ord" <: "Cmp", "<", <, ">", >, "<=", <=, ">=", >=);

            let eq_fn = self.value_env.get("==").unwrap().clone();
            interpreter::value::GLOBAL_EQ_FN.with_borrow_mut(|global| {
                *global = Some(eq_fn);
            });

            self.define_class(
                grammar::DefClassParser::new()
                    .parse("interface Hashable a : Cmp { hash : a -> Int; }")
                    .unwrap(),
            )
            .unwrap();

            let hash_fn = self.value_env.get("hash").unwrap().clone();
            interpreter::value::GLOBAL_HASH_FN.with_borrow_mut(|global| {
                *global = Some(hash_fn);
            });

            self.primitive_class_impl(
                "Hashable",
                "Int",
                vec![(
                    "hash",
                    "Int -> Hasher -> ()",
                    interpreter::Value::primitive("hash-int", 2, |_| unimplemented!()),
                )],
            );

            self.primitive_class_impl(
                "Hashable",
                "String",
                vec![(
                    "hash",
                    "String -> Hasher -> ()",
                    interpreter::Value::primitive("hash-str", 2, |_| unimplemented!()),
                )],
            );

            self.define_primitive("dict", "forall a b => () -> Dict a b", |_| {
                interpreter::Value::dict()
            });

            self.define_primitive(
                "dict-insert",
                "forall (a : Hashable) b => a -> b -> Dict a b -> Dict a b",
                |args| {
                    let k = args[0].clone();
                    let v = args[1].clone();
                    let dict = args[2].clone();
                    dict.dict_insert(k, v)
                },
            );

            self.define_primitive(
                "dict-get",
                "forall (a : Hashable) b => a -> Dict a b -> b",
                |args| {
                    let k = args[0].clone();
                    let dict = args[1].clone();
                    dict.as_dict().unwrap().get(&k).unwrap().clone()
                },
            );

            self.define_primitive(
                "dict-keys",
                "forall (a : Hashable) b => Dict a b -> [a]",
                |args| {
                    let dict = args[0].clone();
                    interpreter::Value::make_list_reverse(dict.as_dict().unwrap().keys().cloned())
                },
            );

            self.define_primitive(
                "dict-values",
                "forall (a : Hashable) b => Dict a b -> [b]",
                |args| {
                    let dict = args[0].clone();
                    interpreter::Value::make_list_reverse(dict.as_dict().unwrap().values().cloned())
                },
            );
        }
    }

    fn primitive_class_impl(
        &mut self,
        cls_name: &str,
        impl_ty: &str,
        methods: Vec<(&str, &str, interpreter::Value)>,
    ) {
        self.class_env = EnvTransformer::add_inst(
            vec![],
            Pred::IsIn(cls_name.into(), self.type_env[impl_ty].clone()),
        )
        .apply(&self.class_env)
        .unwrap();

        for (mth_name, mth_sig, value) in methods {
            let scm = self.build_scheme(grammar::SchemeParser::new().parse(mth_sig).unwrap());

            let mth = self.value_env.get(mth_name).unwrap();

            let (_, k, _, _) = mth.as_method().expect("expected method");
            let ty = scm.get_nth_arg_ty(k).unwrap();

            mth.add_impl(ty.clone(), value)
        }
    }

    fn define_primitive(
        &mut self,
        name: &'static str,
        ty: &str,
        semantic: fn(&[interpreter::Value]) -> interpreter::Value,
    ) {
        let sc = self.build_scheme(grammar::SchemeParser::new().parse(ty).unwrap());
        let arity = sc.arity();

        self.assumptions.push(Assump { i: name.into(), sc });

        self.value_env.insert(
            name.into(),
            interpreter::Value::primitive(name, arity, semantic),
        );
    }

    /// load and run a file relative to the current file.
    pub fn run_file(&mut self, file_path: &str, current_dir: &Path) -> crate::Result<()> {
        let full_path = current_dir.join(file_path);
        let new_dir = full_path.parent().unwrap();

        let file_content = fs::read_to_string(&full_path).map_err(|e| {
            format!(
                "{} while reading file {}",
                e.to_string(),
                full_path.to_string_lossy()
            )
        })?;
        self.run_str(&file_content, new_dir)?;
        Ok(())
    }

    fn run_str(&mut self, file_content: &str, current_dir: &Path) -> crate::Result<()> {
        let program = grammar::ProgramParser::new()
            .parse(&file_content)
            .map_err(|e| e.to_string())?;

        for top in program {
            self.exec_toplevel(top, current_dir)?;
        }
        Ok(())
    }

    fn exec_toplevel(&mut self, top: ast::TopLevel, current_dir: &Path) -> crate::Result<()> {
        match top {
            ast::TopLevel::Include(path) => self.run_file(&*path, current_dir),
            ast::TopLevel::DefClass(dc) => self.define_class(dc),
            ast::TopLevel::ImplClass(ic) => self.implement_class(ic),
            ast::TopLevel::DataType(dt) => self.define_datatype(dt),
            ast::TopLevel::BindGroup(bg) => self.define_globals(bg),
        }
    }

    fn define_class(&mut self, class: ast::DefClass) -> crate::Result<()> {
        let et = EnvTransformer::add_class(class.name.clone(), class.super_classes);

        let mut local_tenv = self.type_env.clone();
        local_tenv.insert(class.varname.clone(), Type::TGen(0));
        let mut assumptions = vec![];
        for (method_name, mut sc) in class.methods {
            self.methods
                .entry(class.name.clone())
                .or_insert((class.varname.clone(), HashMap::new()))
                .1
                .insert(method_name.clone(), sc.clone());

            // insert the "self" type as the first generic
            sc.genvars.insert(
                0,
                (
                    class.varname.clone(),
                    class.kind.clone(),
                    vec![class.name.clone()],
                ),
            );
            let sc = self.build_scheme(sc);

            if sc.is_constant() {
                return Err("all interface type variables must appear in method arguments".into());
            }

            if find(&method_name, &self.assumptions).is_ok() {
                return Err(format!("name {method_name} already used"));
            }

            assumptions.push(Assump {
                i: method_name.clone(),
                sc,
            });
        }

        self.class_env = et.apply(&self.class_env)?;
        for a in assumptions {
            let dispatchable_arg = a.sc.find_first_arg_with_genvar(0).unwrap();

            self.value_env.insert(
                a.i.clone(),
                interpreter::Value::method(a.i.clone(), dispatchable_arg),
            );
            self.assumptions.push(a);
        }

        Ok(())
    }

    fn implement_class(&mut self, ic: ast::ImplClass) -> crate::Result<()> {
        let mut cls_typeenv = self.type_env.clone();
        let (_, preds) = self.build_typeargs(ic.genvars.clone(), &mut cls_typeenv);

        let backup = std::mem::replace(&mut self.type_env, cls_typeenv);

        let ty = self.build_type(ic.ty);

        let mut scenv = self.type_env.clone();
        scenv.insert("the impl type".into(), ty.clone());

        let mut expls = vec![];

        let mut original_method_names = vec![];
        if let Some((var, mut required_methods)) = self.methods.get(&ic.cls).cloned() {
            for mi in ic.methods {
                let name = mi.0;
                let mut sc = required_methods
                    .remove(&name)
                    .ok_or_else(|| format!("unexpected method: {name}"))?;

                // rename the "self" type to something impossible, in case it's being shadowed
                sc.rename(&var, "the impl type".into());

                sc.genvars = ic.genvars.iter().cloned().chain(sc.genvars).collect();

                let sc_ = self.with_tyenv(scenv.clone(), |ctx| ctx.build_scheme(sc));

                let alts = self.build_alts(mi.1);

                // adding an impossible prefix makes the names unique so they don't shadow their
                // generic definitions during type checking.
                expls.push(Expl(format!("tmp {name}"), sc_, alts));
                original_method_names.push(name);
            }

            if !required_methods.is_empty() {
                return Err(format!("missing method impls: {:?}", required_methods));
            }
        }

        self.type_env = backup;

        let et = EnvTransformer::add_inst(preds, Pred::IsIn(ic.cls, ty));
        let class_env = et.apply(&self.class_env)?;

        let mut prog = Program(vec![BindGroup(expls, vec![])]);
        let (_, ti) = ti_program(&class_env, self.assumptions.clone(), &prog)?;

        self.class_env = class_env;

        let ctx = interpreter::Context::new(ti);
        for (Expl(_, sc, alts), name) in prog
            .0
            .pop()
            .unwrap()
            .0
            .into_iter()
            .zip(original_method_names)
        {
            let val = ctx.eval_alts(&alts, &self.value_env);

            let method = self.value_env.get(&name).unwrap();
            let (_, dispatch_arg, _, _) = method.as_method().unwrap();
            let ty = sc.get_nth_arg_ty(dispatch_arg).unwrap();

            method.add_impl(ty.clone(), val);
        }

        Ok(())
    }

    fn define_datatype(&mut self, dt: ast::DataType) -> crate::Result<()> {
        if self.type_env.contains_key(&dt.typename) {
            return Err(format!("type {} already defined", dt.typename));
        }
        let type_arity = dt.genvars.len();
        let kind = Kind::ty_constructor(type_arity);
        let tcon = Type::TCon(Tycon(dt.typename.clone(), kind));
        self.type_env.insert(dt.typename.clone(), tcon.clone());

        let mut method_tenv = self.type_env.clone();
        let (vs, preds) = self.build_typeargs(dt.genvars, &mut method_tenv);

        let backup = std::mem::replace(&mut self.type_env, method_tenv);

        for (i, params) in dt.constructors {
            if find(&i, &self.assumptions).is_ok() {
                return Err(format!("name {i} already used"));
            }

            let args: Vec<_> = params.into_iter().map(|p| self.build_type(p)).collect();

            // apply the type-constructor
            let mut dty = tcon.clone();
            let tc_args = vs.iter().map(|v| Type::TVar(v.clone()));
            for a in tc_args {
                dty = Type::tapp(dty, a)
            }

            // constructor-function arguments
            let mut ty = dty.clone();
            for a in args.into_iter().rev() {
                ty = Type::func(a, ty);
            }

            let assump = Assump {
                i: i.clone(),
                sc: Scheme::quantify(&vs, &Qual(preds.clone(), ty)),
            };
            self.assumptions.push(assump.clone());
            self.constructors.push(assump);

            self.value_env
                .insert(i.clone(), interpreter::Value::constructor(dty.clone(), i));
        }

        self.type_env = backup;
        Ok(())
    }

    fn define_globals(&mut self, bg: ast::BindGroup) -> crate::Result<()> {
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

    pub fn eval_expr(&mut self, expr: ast::Expr) -> crate::Result<interpreter::Value> {
        let expr = self.build_expr(expr);

        let mut ti = TI::new();
        let _ = ti_expr(&mut ti, &self.class_env, &self.assumptions, &expr)?;

        let value = interpreter::Context::new(ti).eval_expr(&expr, &self.value_env);

        Ok(value)
    }
}
