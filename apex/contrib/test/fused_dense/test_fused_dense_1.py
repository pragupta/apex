import unittest
import os

import torch
from torch.testing._internal import common_utils
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from typing import Optional

SKIP_TEST = None
try:
    from apex import fused_dense
except ImportError as e:
    SKIP_TEST = e

EPS = 1e-12
if torch.version.hip:
     e4m3_type = torch.float8_e4m3fnuz
     e5m2_type = torch.float8_e5m2fnuz
     E4M3_MAX_POS = torch.finfo(torch.float8_e4m3fnuz).max
     E5M2_MAX_POS = torch.finfo(torch.float8_e5m2fnuz).max
else:
     e4m3_type = torch.float8_e4m3fn
     e5m2_type = torch.float8_e5m2
     E4M3_MAX_POS = torch.finfo(torch.float8_e4m3fn).max
     E5M2_MAX_POS = torch.finfo(torch.float8_e5m2).max

def amax_to_scale(amax: torch.Tensor, float8_dtype: torch.dtype, orig_dtype: torch.dtype):
    scale = torch.empty_like(amax, dtype=torch.float32)
    if float8_dtype == e4m3_type:
        result = E4M3_MAX_POS / torch.clamp(amax, min=EPS)
    elif float8_dtype == e5m2_type:
        result = E5M2_MAX_POS / torch.clamp(amax, min=EPS)
    else:
        raise ValueError(f"Unsupported float8_dtype: {float8_dtype}")

    if orig_dtype is torch.float16:
        result = torch.clamp(result, max=torch.finfo(torch.float16).max)

    scale.copy_(result)
    return scale

def tensor_to_scale(x: torch.Tensor, fp8_dtype: torch.dtype, dim=None):
    if dim is None:
        amax = torch.max(torch.abs(x))
    else:
        amax = torch.max(torch.abs(x), dim=dim, keepdim=True).values
    return amax_to_scale(amax, fp8_dtype, x.dtype)

def to_fp8_saturated(x: torch.Tensor, fp8_dtype: torch.dtype):
    if fp8_dtype == e4m3_type:
        x = x.clamp(min=-1*E4M3_MAX_POS, max=E4M3_MAX_POS)
    elif fp8_dtype == e5m2_type:
        x = x.clamp(min=-1*E5M2_MAX_POS, max=E5M2_MAX_POS)
    else:
        raise ValueError(f"to_fp8_staurated(): Unsupported fp8_dtype: {fp8_dtype}")

    return x.to(fp8_dtype)

def addmm_float8_unwrapped(
    a_data: torch.Tensor,
    a_scale: torch.Tensor,
    b_data: torch.Tensor,
    b_scale: torch.tensor,
    output_dtype: torch.dtype,
    output_scale: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    a_inverse_scale = a_scale.reciprocal()
    b_inverse_scale = b_scale.reciprocal()
    if output_dtype == torch.float32 and bias is not None:
        # Bias is not supported by _scaled_mm when output is fp32
        output = torch._scaled_mm(
            a_data,
            b_data,
            scale_a=a_inverse_scale,
            scale_b=b_inverse_scale,
            scale_result=output_scale,
            out_dtype=output_dtype,
        )
        output += bias
        return output
    output = torch._scaled_mm(
        a_data,
        b_data,
        bias=bias,
        scale_a=a_inverse_scale,
        scale_b=b_inverse_scale,
        scale_result=output_scale,
        out_dtype=output_dtype,
    )
    return output

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
        ref_inputs = torch.randn(batch_size,in_features, dtype=torch.float32, device=torch.device("cuda")).requires_grad_(True)
        tst_inputs = ref_inputs.clone().detach().requires_grad_(True).to(dtype=dtype)

        # Create dense
        # self.weight = nn.Parameter(torch.randn(in_features, out_features))
        # self.bias   = nn.Parameter(torch.randn(out_features))
        dense = fused_dense.FusedDense(in_features, out_features)
        dense.cuda()
        ref_weights = dense.weight.t()

        # --------------------------------------------------------------------------------------------------
        #  Farward pass
        # --------------------------------------------------------------------------------------------------
        if dtype == torch.float8_e4m3fnuz or dtype == torch.float8_e5m2fnuz:
            ref_inputs_scale  = tensor_to_scale(ref_inputs,  dtype).float()
            ref_weights_scale = tensor_to_scale(ref_weights, dtype).float()

            ref_inputs_fp8  = to_fp8_saturated(ref_inputs * ref_inputs_scale,  dtype)
            ref_weights_fp8 = to_fp8_saturated(ref_weights* ref_weights_scale, dtype)

            y_ref = addmm_float8_unwrapped(ref_inputs_fp8, ref_inputs_scale,
                                           ref_weights_fp8, ref_weights_scale,
                                           output_dtype=dtype, bias=dense.bias.to(torch.half))
#            dense.to(dtype=dtype)
        else:
            ref_inputs = ref_inputs.to(dtype=dtype)
            dense.to(dtype=dtype)
            y_ref = torch.matmul(ref_inputs, ref_weights)+dense.bias

        y_tst = dense(tst_inputs)
        # torch.testing.assert_close(y_ref,  y_tst,  atol=1e-3, rtol=1e-3, equal_nan=True)

        # --------------------------------------------------------------------------------------------------
        #  Backward pass
        #    dX  = dY ⋅ WT
        #    dW  = XT ⋅ dY and db=sum(dY)
        # --------------------------------------------------------------------------------------------------
        dy  = torch.randn_like(y_tst).to(dtype=dtype)
        y_tst.backward(dy)

        # print("Ref-Input\n",ref_inputs.t())
        # print("Dy\n",dy)
        print("y_ref:\n", y_ref)
        print("y_tst:\n", y_tst)


        print("dw_ref Tensor:\n", torch.matmul(ref_inputs.t(), dy))
        print("dense.weight.grad Tensor:\n", dense.weight.grad)

        print("********************************************************************")
        print("dx_ref Tensor:\n", torch.matmul(dy, dense.weight.clone()))
        print("tst_inputs.grad Tensor:\n", tst_inputs.grad)

        print("********************************************************************")
        print("db_ref Tensor:\n",   dy.sum(0, False))
        print("dense.bias.grad Tensor:\n",   dense.bias.grad)
        print("********************************************************************")

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
