# Excluded full-model timing attempt

These JSON files preserve an attempted direct/candidate comparison, but they
are not benchmark evidence. `ullm-openai.service` was externally started at
20:19:35 JST while the direct run's tail and the entire candidate run were
executing. They also record different `runner_git_head` values because the
shared repository advanced during the attempt.

Do not compare 14.6857303669 tok/s with 14.9593002915 tok/s or report their
ratio. A future measurement must wait for a coordinated isolated R9700
window, rerun both variants from one fixed commit, and retain the required
service restoration record.
