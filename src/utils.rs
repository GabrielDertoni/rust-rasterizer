#[macro_export]
macro_rules! unroll_array {
    ($v:ident = $($values:literal),* => $e:expr) => {
        [$({
            let $v = $values;
            $e
        }),*]
    }
}

#[macro_export]
macro_rules! unroll {
    (for $v:ident in [$($values:literal),*] $block:tt) => {
        $(
            let $v = $values;
            $block;
        )*
    }
}
