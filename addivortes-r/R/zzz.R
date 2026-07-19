.onLoad <- function(libname, pkgname) {
  # parsnip engine registration is best-effort: when parsnip is installed
  # register now, otherwise arm a hook for if/when it loads. Never allow a
  # registration hiccup to break package load.
  if (requireNamespace("parsnip", quietly = TRUE)) {
    tryCatch(register_parsnip_models(), error = function(e) NULL)
  } else {
    setHook(
      packageEvent("parsnip", "onLoad"),
      function(...) tryCatch(register_parsnip_models(), error = function(e) NULL)
    )
  }
}
