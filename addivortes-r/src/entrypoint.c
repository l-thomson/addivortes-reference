// Forward routine registration from C to Rust: R CMD check looks for the
// C-named init routine, and the reference through this translation unit
// stops the linker discarding the Rust static library.

void R_init_addivortesr_extendr(void *dll);

void R_init_addivortesr(void *dll) {
    R_init_addivortesr_extendr(dll);
}
