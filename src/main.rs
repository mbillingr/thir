// This is "Typying Haskell in Rust", based on "Typing Haskell in Haskell":
// https://web.cecs.pdx.edu/~mpj/thih/thih.pdf?_gl=1*1kpcq97*_ga*MTIwMTgwNTIxMS4xNzAyMzAzNTg2*_ga_G56YW5RFXN*MTcwMjMwMzU4NS4xLjAuMTcwMjMwMzU4NS4wLjAuMA..

mod frontend;
mod interpreter;
mod type_checker;
mod utils;

use frontend::ast;
use frontend::Runner;
use std::env;
use std::path::{Path, PathBuf};
type Result<T> = std::result::Result<T, String>;

fn main() -> Result<()> {
    let args: Vec<String> = env::args().collect();
    if args.len() != 2 {
        return Err(format!("Usage: {} <file_path>", args[0]));
    }

    let mut ctx = Runner::new();
    ctx.init();

    // try to load the prelude/prelude.ml relative to the current working directory
    match ctx.run_file("prelude.ml", Path::new("prelude")) {
        Err(e) if e.contains("No such file or directory") => {}
        other => other?,
    }

    ctx.run_file(&args[1], &PathBuf::from("."))?;

    ctx.eval_expr(ast::Expr::app(
        ast::Expr::Var("main".into()),
        ast::Expr::Lit(ast::Literal::Unit),
    ))?;

    Ok(())
}
