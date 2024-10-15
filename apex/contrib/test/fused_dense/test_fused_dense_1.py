import unittest
import os

import torch
from torch.testing._internal import common_utils
from torch.testing._internal.common_device_type import instantiate_device_type_tests

SKIP_TEST = None
try:
    from apex import fused_dense
    from apex import fp8_utils
except ImportError as e:
    SKIP_TEST = e

@unittest.skipIf(SKIP_TEST, f"{SKIP_TEST}")
class FusedDenseTest(common_utils.TestCase):

    def _test_fused_dense(self, dtype, seed=0):

        os.environ["TORCH_ALLOW_TF32_CUBLAS_OVERRIDE"] = "0"
        torch.manual_seed(seed)

        # seq_length = 4 # 512
        # sequences  = 3 # 3
        # hidden_dim = 8 # 1024
        batch_size   = 16
        in_features  = 16 #3
        out_features = 64 #2

        # --------------------------------------------------------------------------------------------------
        #  Setup
        # --------------------------------------------------------------------------------------------------
        ref_inputs = torch.randn(batch_size, in_features, dtype=torch.float32, device=torch.device("cuda")).requires_grad_(True).to(dtype=dtype)
        tst_inputs = ref_inputs.clone().detach().requires_grad_(True)

        # Create dense
        dense = fused_dense.FusedDense(in_features, out_features)
        dense.cuda()
        ref_weights = dense.weight.t()

        # --------------------------------------------------------------------------------------------------
        #  Farward pass
        # --------------------------------------------------------------------------------------------------
        if dtype == torch.float8_e4m3fnuz or dtype == torch.float8_e5m2fnuz:
            #Cast back to float32 so we can scale them into f8 range
            ref_inputs = ref_inputs.to(dtype=torch.float32)
#             ref_inputs_scale  = fp8_utils.tensor_to_scale(ref_inputs,  dtype).float()
#             ref_weights_scale = fp8_utils.tensor_to_scale(ref_weights, dtype).float()
#
#             ref_inputs_fp8  = fp8_utils.to_fp8_saturated(ref_inputs * ref_inputs_scale,  dtype)
#             ref_weights_fp8 = fp8_utils.to_fp8_saturated(ref_weights* ref_weights_scale, dtype)
#
#             y_ref = fp8_utils.addmm_float8_unwrapped(ref_inputs_fp8, ref_inputs_scale,
#                                            ref_weights_fp8, ref_weights_scale,
#                                            output_dtype=dtype, bias=dense.bias.to(torch.half))
        else:
            dense.to(dtype=dtype)
        print(f"ref_inputs.dtype={ref_inputs.dtype} dense.weight.dtype={dense.weight.dtype} dense.bias.dtype={dense.bias.dtype}")
        y_ref = torch.matmul(ref_inputs, ref_weights)+dense.bias

        print(f"test_input.dtype={tst_inputs.dtype} dense.weight.dtype={dense.weight.dtype} dense.bias.dtype={dense.bias.dtype}")
        print (tst_inputs)
        y_tst = dense(tst_inputs)

        print(f"y_ref={y_ref.dtype} y_tst={y_tst.dtype}")
        torch.testing.assert_close(y_ref.to(torch.float32),  y_tst.to(torch.float32),  atol=1e-3, rtol=1e-3, equal_nan=True)

        # --------------------------------------------------------------------------------------------------
        #  Backward pass
        #    dX  = dY ⋅ WT
        #    dW  = XT ⋅ dY and db=sum(dY)
        # --------------------------------------------------------------------------------------------------
        # @PG - gradients need to e5m2
#         dy  = torch.randn_like(y_tst).to(dtype=dtype)
#         y_tst.backward(dy)
#
#         # print("Ref-Input\n",ref_inputs.t())
#         # print("Dy\n",dy)
#         print("y_ref:\n", y_ref)
#         print("y_tst:\n", y_tst)
#
#
#         print("dw_ref Tensor:\n", torch.matmul(ref_inputs.t(), dy))
#         print("dense.weight.grad Tensor:\n", dense.weight.grad)
#
#         print("********************************************************************")
#         print("dx_ref Tensor:\n", torch.matmul(dy, dense.weight.clone()))
#         print("tst_inputs.grad Tensor:\n", tst_inputs.grad)
#
#         print("********************************************************************")
#         print("db_ref Tensor:\n",   dy.sum(0, False))
#         print("dense.bias.grad Tensor:\n",   dense.bias.grad)
#         print("********************************************************************")

        # torch.testing.assert_close(dx_ref, tst_inputs.grad, atol=1e-3, rtol=1e-3, equal_nan=True)
        # torch.testing.assert_close(dw_ref, dense.weight.grad, atol=1e-3, rtol=1e-3, equal_nan=True)
        # torch.testing.assert_close(db_ref, dense.bias.grad, atol=1e-3, rtol=1e-3, equal_nan=True)

    # @common_utils.parametrize("dtype", [torch.half, torch.float, torch.bfloat16, torch.float8_e4m3fn])
    # @common_utils.parametrize("dtype", [torch.half])
    @common_utils.parametrize("dtype", [torch.float8_e4m3fnuz])
    def test_fused_dense(self, dtype):
        self._test_fused_dense(dtype)


instantiate_device_type_tests(FusedDenseTest, globals(), only_for=("cuda",))

if __name__ == "__main__":
    common_utils.run_tests()
