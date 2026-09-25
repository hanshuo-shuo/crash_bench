# Rebuild the descriptive recovery-window table and figure from the frozen
# 273-decision exact-state corpus. Run from the repository root:
# Rscript docs/audits/20260925/group_meeting/make_recovery_window.R

source_file <- "results/iclr27/option_support_audit_787623226de0_20260829T112149Z/decision_labels.csv"
out_dir <- "docs/audits/20260925/group_meeting"
decisions <- read.csv(source_file, stringsAsFactors = FALSE)
stopifnot(nrow(decisions) == 273, length(unique(decisions$source)) == 20)
glass <- decisions[decisions$condition == "glass", ]
stopifnot(nrow(glass) == 101)

horizons <- c(40, 30, 20, 10, 5)
options <- c(Base = "base_outcome", Detour = "detour_outcome",
             Retreat = "retreat_outcome")
outcomes <- c("catastrophe", "safe_noncompletion", "task_success")

counts <- do.call(rbind, lapply(horizons, function(h) {
  block <- glass[glass$horizon == h, ]
  do.call(rbind, lapply(names(options), function(option) {
    tab <- table(factor(block[[options[[option]]]], levels = outcomes))
    data.frame(horizon = h, option = option, decisions = nrow(block),
               sources = length(unique(block$source)),
               accident = unname(tab["catastrophe"]),
               safe_noncompletion = unname(tab["safe_noncompletion"]),
               task_success = unname(tab["task_success"]))
  }))
}))
stopifnot(all(counts$accident + counts$safe_noncompletion +
              counts$task_success == counts$decisions))
write.csv(counts, file.path(out_dir, "recovery_window_counts.csv"), row.names = FALSE)

png(file.path(out_dir, "recovery_window.png"), width = 1600, height = 750,
    res = 150)
par(mfrow = c(1, 2), mar = c(5.5, 5, 3.5, 1.2), oma = c(3, 0, 1, 0),
    family = "sans")
colors <- c("#bf4f4f", "#d7a94e", "#378d69")
for (option in c("Detour", "Retreat")) {
  block <- counts[counts$option == option, ]
  mat <- rbind(block$accident, block$safe_noncompletion,
               block$task_success)
  barplot(mat / matrix(rep(block$decisions, each = 3), nrow = 3),
          col = colors, border = NA, space = 0.42, ylim = c(0, 1),
          names.arg = paste0("T-", block$horizon, "\nn=", block$decisions),
          cex.names = 0.9, las = 1, axes = FALSE,
          ylab = "Share of saved decisions",
          main = paste(option, "from the same saved states"))
  axis(2, at = seq(0, 1, 0.25), labels = paste0(seq(0, 100, 25), "%"), las = 1)
  abline(h = seq(0.25, 0.75, 0.25), col = "#e7e7e7", lwd = 0.8)
  box(bty = "l")
}
mtext("Red: accident    Amber: safe but unfinished    Green: task complete",
      side = 1, outer = TRUE, line = 1.1, cex = 0.9)
dev.off()
