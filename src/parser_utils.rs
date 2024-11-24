#[macro_export]
macro_rules! swap {
    ($f:path, $a:tt, $b:tt) => {
        $f($b, $a)
    };
}
