"""Run against the same installed wheel under each supported Torch runtime."""

import torch

import sgl_fa3  # noqa: F401


def test_forward_dispatcher_contract():
    schema = torch.ops.sgl_fa3.fwd.default._schema
    assert len(schema.arguments) == 37
    assert len(schema.returns) == 4
    assert [arg.name for arg in schema.arguments[-3:]] == [
        "sinks",
        "sparse_mask_fine",
        "only_qv",
    ]
    out = schema.arguments[6]
    assert out.name == "out" and out.alias_info.is_write
    assert out.alias_info.after_set == schema.returns[0].alias_info.after_set
    assert torch._C._dispatch_has_kernel_for_dispatch_key("sgl_fa3::fwd", "CUDA")


def test_scheduler_dispatcher_contract():
    schema = torch.ops.sgl_fa3.get_scheduler_metadata.default._schema
    assert len(schema.arguments) == 24
    assert len(schema.returns) == 1
    assert schema.arguments[8].name == "seqused_k"
    assert schema.arguments[-1].default_value == 0
    assert torch._C._dispatch_has_kernel_for_dispatch_key(
        "sgl_fa3::get_scheduler_metadata", "CUDA"
    )
