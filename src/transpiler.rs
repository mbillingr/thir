use crate::frontend::type_inference::Alt;
use crate::frontend::{ast, type_inference};
use crate::type_checker;
use crate::type_checker::kinds::Kind;
use crate::type_checker::qualified::Qual;
use crate::type_checker::scheme::Scheme;
use crate::type_checker::types::{Tycon, Tyvar};
use crate::type_checker::{types, Id};

pub struct Context {
    pub output: String,
}

impl Context {
    pub fn new() -> Self {
        Context {
            output: String::new(),
        }
    }

    pub fn define_datatype(
        &mut self,
        name: &str,
        genvars: &[(Id, Kind, Vec<Id>)],
        constructors: &[(Id, Vec<ast::Type>)],
    ) {
        let gens = genvars
            .iter()
            .map(|(id, _, _)| id.to_string())
            .collect::<Vec<_>>()
            .join(", ");

        self.push_line(&format!("# datatype {name}"));
        let name = self.julify(name);
        self.push_line(&format!("abstract type {name}{{{gens}}} end"));
        for (con, tys) in constructors {
            let variant = self.julify(&con);
            self.push_line(&format!("# variant {con}"));
            self.push_line(&format!("struct {variant}{{{gens}}} <: {name}{{{gens}}}"));
            for (i, ty) in tys.iter().enumerate() {
                self.push_line(&format!("  field_{i}"));
            }
            self.push_line(&format!("end"));
        }
        self.push_line("")
    }

    pub fn implement_class(
        &mut self,
        clsname: &str,
        ty: &types::Type,
        prog: &type_inference::Program,
        ti: &type_checker::TI,
    ) {
        self.push_line(&format!("# implementing {clsname} for {ty:?}"));

        for type_inference::BindGroup(expl, impl_) in &prog.0 {
            for type_inference::Expl(id, sc, alts) in expl {
                self.push_line(&self.member_function(id, sc, alts, ti))
            }

            for type_inference::Impl(id, alts) in impl_.iter().flatten() {
                //self.push_line(&self.function(id, alts, ti))
                todo!()
            }
        }

        //todo!("{clsname}")
    }

    fn member_function(
        &self,
        id: &str,
        sc: &Scheme,
        alts: &Vec<Alt>,
        ti: &type_checker::TI,
    ) -> String {
        let mut result = String::new();
        result.push_line(&format!("# {id} {sc:?}"));

        let n_args = alts.iter().map(|Alt(ps, _)| ps.len()).max().unwrap();

        let Scheme::Forall(kinds, Qual(preds, ty)) = sc;
        let (argtys, rety) = ty.fn_types();

        let args = argtys
            .iter()
            .enumerate()
            .map(|(i, t)| format!("arg_{}::{}", i, self.arg_type(t)))
            .collect::<Vec<_>>()
            .join(", ");

        result.push_str(&format!("function {id}({args}) = ", id = self.julify(id)));

        result.push_str("... TODO ...");

        result
    }

    fn arg_type(&self, t: &types::Type) -> String {
        match t {
            types::Type::TVar(Tyvar(id, _)) => self.julify(id),
            types::Type::TCon(Tycon(id, _)) => self.julify(id),
            types::Type::TApp(app) => {
                if let types::Type::TApp(app2) = &app.0 {
                    if let types::Type::TCon(Tycon(op, _)) = &app2.0 {
                        if op == "->" {
                            let lhs = self.arg_type(&app2.1);
                            let rhs = self.arg_type(&app.1);
                            return format!("{} -> {}", lhs, rhs);
                        }
                    }
                }

                let mut app = app;
                let mut args_rev = vec![&app.1];
                while let types::Type::TApp(a) = &app.0 {
                    args_rev.push(&a.1);
                    app = a;
                }
                let args = args_rev
                    .iter()
                    .rev()
                    .map(|t| self.arg_type(t))
                    .collect::<Vec<_>>()
                    .join(", ");
                format!("{}{{{}}}", self.arg_type(&app.0), args)
            }
            types::Type::TGen(n) => "Any".to_string(), // i'm not sure about that
        }
    }

    fn julify(&self, s: &str) -> String {
        let result: String = s
            .chars()
            .map(|ch| match ch {
                '_' => "_d_".to_string(),
                '[' => "_bro_".to_string(),
                ']' => "_brc_".to_string(),
                ':' => "_cln_".to_string(),
                _ => ch.to_string(),
            })
            .collect();
        result
    }
}

trait PushLine {
    fn push_line(&mut self, line: &str);
}

impl PushLine for String {
    fn push_line(&mut self, line: &str) {
        self.push_str(line);
        self.push('\n');
    }
}

impl PushLine for Context {
    fn push_line(&mut self, line: &str) {
        self.output.push_line(line);
    }
}
