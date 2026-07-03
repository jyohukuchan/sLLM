# Third-Party Notices

## Scope
- This repository is distributed as uLLM (Apache-2.0).
- `reference-src/` contains local research materials only; contents are not redistributed with uLLM.
- If specific source files are imported from third-party projects in the future, add corresponding license/copyright notices at file level and/or component level and keep this file updated.

## Referenced Third-Party Projects
- llama.cpp — License: MIT
- AMD ROCm / ATOM — License: MIT
- AITER — License: MIT
- vLLM — License: Apache-2.0
- SGLang — License: Apache-2.0
- TensorRT-LLM — License: Apache-2.0

## Policy
- For implementation-code reuse, preserve original copyright headers in the copied files, and keep corresponding LICENSE/NOTICE texts reachable in the distribution artifact.
- For design/reference adaptation without code import, include only an index-level record of consulted projects and avoid implying legal clearance.
- Runtime dependencies, model weights, and benchmark datasets are separate licensing surfaces and must be audited before distribution or publication of binaries/artifacts that include them.
- This file is intentionally concise and intended as a distribution-level index. Detailed license texts are kept in upstream repositories.

### Future redistribution rule of thumb
- If third-party code is copied/ported in the future, copyright/license notices should be kept at file or component level, and NOTICE/Third Party notices should continue to be distributed with redistributed binaries/sources.
