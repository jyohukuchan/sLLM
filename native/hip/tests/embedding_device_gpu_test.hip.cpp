#include "decode_control_kernel_internal.hpp"
#include "embedding_kernel_internal.hpp"
#include <cstdio>
#include <cstring>
#include <vector>
#define H(x)                                                                   \
  do {                                                                         \
    auto e = (x);                                                              \
    if (e != hipSuccess) {                                                     \
      std::fprintf(stderr, "%s\n", hipGetErrorString(e));                      \
      return 1;                                                                \
    }                                                                          \
  } while (0)
int main(int argc, char **argv) {
  hipDeviceProp_t props{};
  H(hipGetDeviceProperties(&props, 0));
  if (argc != 2 ||
      std::strstr(props.gcnArchName, argv[1]) != props.gcnArchName) {
    std::fprintf(stderr, "expected exact target argument, observed %s\n",
                 props.gcnArchName);
    return 9;
  }
  const size_t target_len = std::strlen(argv[1]);
  if (props.gcnArchName[target_len] != '\0' &&
      props.gcnArchName[target_len] != ':')
    return 9;
  constexpr unsigned hidden = 259, vocab = 7, rows = 3;
  std::vector<uint16_t> w(hidden * vocab), out(hidden * rows);
  for (unsigned i = 0; i < w.size(); ++i)
    w[i] = static_cast<uint16_t>(i);
  uint16_t *dw, *dout;
  int32_t *di;
  sllm_decode_control::ControlV1 *dc;
  H(hipMalloc(&dw, w.size() * 2));
  H(hipMalloc(&dout, out.size() * 2));
  H(hipMalloc(&di, rows * 4));
  H(hipMalloc(&dc, sizeof(*dc)));
  H(hipMemcpy(dw, w.data(), w.size() * 2, hipMemcpyHostToDevice));
  for (int bad : {-2, -1, 0, 7, 8}) {
    int32_t ids[rows] = {0, 6, bad == -2 ? 3 : bad};
    sllm_decode_control::ControlV1 c{};
    H(hipMemcpy(di, ids, sizeof(ids), hipMemcpyHostToDevice));
    H(hipMemset(dc, 0, sizeof(*dc)));
    H(sllm_embedding_kernel::launch_gather_device(dw, di, dout, rows, hidden,
                                                  vocab, dc, nullptr));
    H(hipMemcpy(out.data(), dout, out.size() * 2, hipMemcpyDeviceToHost));
    H(hipMemcpy(&c, dc, sizeof(c), hipMemcpyDeviceToHost));
    bool invalid = ids[2] < 0 || ids[2] >= int(vocab);
    if (c.status != (invalid ? static_cast<unsigned>(
                                   sllm_decode_control::Status::InvalidToken)
                             : 0) ||
        c.halted != unsigned(invalid))
      return 2;
    for (unsigned r = 0; r < rows; ++r)
      for (unsigned k = 0; k < hidden; ++k)
        if (out[r * hidden + k] !=
            (r == 2 && invalid ? 0 : w[ids[r] * hidden + k]))
          return 3;
  }
  H(hipFree(dc));
  H(hipFree(di));
  H(hipFree(dout));
  H(hipFree(dw));
  puts("PASS device embedding boundaries and bitwise oracle");
}
