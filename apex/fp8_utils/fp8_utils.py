import torch
from typing import Optional

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

