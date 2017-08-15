#' @exportClass mcbdMod
setClass("mcbdMod",
         representation(
           call = "call",
           opt = "list",
           args = "list",
           quad = "numeric",
           devcomp = "list"
         ))