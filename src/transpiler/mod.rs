use crate::frontend::ast;
use crate::frontend::type_inference::Program;
use crate::type_checker::kinds::Kind;
use crate::type_checker::type_inference::TI;
use crate::type_checker::Id;
use std::collections::HashMap;
use std::rc::Rc;

#[derive(Clone)]
pub struct Context {
    pub output: String,
}

impl Context {
    pub fn new() -> Self {
        Context {
            output: String::new(),
        }
    }

    pub fn define_class(
        &mut self,
        name: &str,
        varname: &str,
        kind: &Kind,
        supers: &[Id],
        methods: &[(Id, ast::Scheme)],
    ) {
        match kind {
            Kind::Star => return self.define_simple_class(name, varname, supers, methods),
            Kind::Kfun(rc) => match &**rc {
                (Kind::Star, Kind::Star) => {
                    return self.define_first_kinded_class(name, varname, supers, methods)
                }
                _ => {}
            },
        }
        eprintln!("TODO: {:?}", kind)
    }

    pub fn define_simple_class(
        &mut self,
        name: &str,
        varname: &str,
        supers: &[String],
        methods: &[(Id, ast::Scheme)],
    ) {
        let mut rename_types = HashMap::new();
        rename_types.insert(varname, "Self");

        let name = rustify(name);

        let opt_supers = if supers.is_empty() {
            "".to_string()
        } else {
            format!(": {}", supers.join(" + "))
        };

        let methods = methods
            .iter()
            .map(|(id, scm)| format!("    {};", generic_function_head(id, scm, &rename_types)))
            .collect::<Vec<_>>()
            .join("\n");

        self.output.push_str(&format!(
            "trait {name} {opt_supers} {{
{methods}
}}

"
        ));
    }

    pub fn define_first_kinded_class(
        &mut self,
        name: &str,
        varname: &str,
        supers: &[String],
        methods: &[(Id, ast::Scheme)],
    ) {
        let mut rename_types = HashMap::new();
        rename_types.insert(varname, "Self");

        let name = rustify(name);

        let opt_supers = if supers.is_empty() {
            "".to_string()
        } else {
            format!(": {}", supers.join(" + "))
        };

        let methods = methods
            .iter()
            .map(|(id, scm)| format!("    {};", generic_function_head(id, scm, &rename_types)))
            .collect::<Vec<_>>()
            .join("\n");

        self.output.push_str(&format!(
            "trait {name} {opt_supers} {{
type Output;
{methods}
}}

"
        ));
    }
}

pub fn generic_function_head(
    name: &str,
    scm: &ast::Scheme,
    rename_types: &HashMap<&str, &str>,
) -> String {
    let name = rustify(name);

    let generics: Vec<String> = scm.genvars.iter().map(|(v, k, ps)| v.clone()).collect();
    let generics = if generics.is_empty() {
        "".to_string()
    } else {
        format!("<{}>", generics.join(", "))
    };

    let (atys, rety) = scm.ty.fn_types();
    let rety = translate_type(rety, rename_types);
    let atys: Vec<_> = atys
        .into_iter()
        .map(|t| translate_type(t, rename_types))
        .enumerate()
        .map(|(i, t)| format!("arg_{i}: &{t}"))
        .collect();
    let args = atys.join(", ");

    format!("fn {name}{generics}({args}) -> {rety}")
}

pub fn translate_type(ty: &ast::Type, rename: &HashMap<&str, &str>) -> String {
    match ty {
        _ if ty.as_fn().is_some() => {
            let (atys, rety) = ty.fn_types();
            let rety = translate_type(rety, rename);
            let atys: Vec<_> = atys
                .into_iter()
                .map(|t| translate_type(t, rename))
                .enumerate()
                .map(|(i, t)| format!("arg_{i}: &{t}"))
                .collect();
            let args = atys.join(", ");

            format!("fn({args}) -> {rety}")
        }
        ast::Type::Named(name) => rustify(rename.get(name.as_str()).unwrap_or(&name.as_str())),
        ast::Type::Apply(f, a) => format!(
            "{}<{}>",
            translate_type(f, rename),
            translate_type(a, rename)
        ),
    }
}

fn rustify(s: &str) -> String {
    s.chars()
        .map(|ch| match ch {
            '_' => "_dash_".to_string(),
            '+' => "_plus_".to_string(),
            '-' => "_minus_".to_string(),
            '*' => "_star_".to_string(),
            '/' => "_slash_".to_string(),
            '=' => "_eq_".to_string(),
            '<' => "_lt_".to_string(),
            '>' => "_gt_".to_string(),
            _ => ch.to_string(),
        })
        .collect()
}
