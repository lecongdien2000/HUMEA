import torch
from pathlib import Path

from memory_utils import (
    backward_through_gradient_bridge,
    checkpoint_call,
    checkpoint_loss,
    gradient_proxy,
)


def test_checkpoint_loss_preserves_value_and_gradients():
    def quadratic_loss(tensor, scale):
        return (tensor.square() * scale).sum()

    direct_input = torch.randn(8, requires_grad=True)
    checkpoint_input = direct_input.detach().clone().requires_grad_(True)

    direct = quadratic_loss(direct_input, 0.25)
    checkpointed = checkpoint_loss(quadratic_loss, checkpoint_input, 0.25)
    direct.backward()
    checkpointed.backward()

    torch.testing.assert_close(checkpointed, direct)
    torch.testing.assert_close(checkpoint_input.grad, direct_input.grad)


def test_checkpoint_call_supports_nested_outputs_and_keyword_arguments():
    def encoder(tensor, *, scale):
        return [tensor * scale, tensor.square()], {"sum": tensor.sum()}

    value = torch.randn(8, requires_grad=True)
    outputs, metadata = checkpoint_call(encoder, value, scale=0.5)
    (outputs[0].sum() + outputs[1].sum() + metadata["sum"]).backward()

    assert value.grad is not None


def test_gradient_bridge_matches_end_to_end_backpropagation():
    direct_input = torch.randn(6, dtype=torch.float64, requires_grad=True)
    bridged_input = direct_input.detach().clone().requires_grad_(True)
    direct_weight = torch.randn(6, dtype=torch.float64, requires_grad=True)
    bridged_weight = direct_weight.detach().clone().requires_grad_(True)

    direct_hidden = direct_input.square()
    direct_output = direct_hidden.sin()
    direct_loss = (direct_hidden * direct_weight).sum() + direct_output.square().sum()
    direct_loss.backward()

    bridged_hidden = bridged_input.square()
    bridged_output = bridged_hidden.sin()
    hidden_proxy = gradient_proxy(bridged_hidden)
    output_proxy = gradient_proxy(bridged_output)
    bridged_loss = (hidden_proxy * bridged_weight).sum() + output_proxy.square().sum()
    bridged_loss.backward()
    backward_through_gradient_bridge(
        [bridged_hidden, bridged_output], [hidden_proxy, output_proxy]
    )

    torch.testing.assert_close(bridged_loss, direct_loss)
    torch.testing.assert_close(bridged_input.grad, direct_input.grad)
    torch.testing.assert_close(bridged_weight.grad, direct_weight.grad)


def test_gradient_bridge_skips_disabled_modalities():
    value = torch.randn(6, requires_grad=True)
    outputs = [value.square(), torch.zeros_like(value)]
    proxies = [gradient_proxy(output) for output in outputs]
    sum(proxy.sum() for proxy in proxies).backward()
    backward_through_gradient_bridge(outputs, proxies)
    torch.testing.assert_close(value.grad, 2 * value)


def test_training_releases_proxy_references_before_estimator_forward():
    import ast

    tree = ast.parse((Path(__file__).parents[1] / "train.py").read_text())
    train = next(node for node in ast.walk(tree)
                 if isinstance(node, ast.FunctionDef) and node.name == "train")
    epoch_loop = next(node for node in train.body if isinstance(node, ast.For))
    estimator_forward = next(node for node in epoch_loop.body
                             if isinstance(node, ast.With))
    deleted = {name.id for node in epoch_loop.body
               if isinstance(node, ast.Delete) and node.lineno < estimator_forward.lineno
               for name in ast.walk(node) if isinstance(name, ast.Name)}
    assert {"gph_loss_emb", "img_loss_emb", "rel_loss_emb", "att_loss_emb",
            "att_text_loss_emb", "rel_text_loss_emb", "joint_loss_emb",
            "in_loss", "loss_joi"} <= deleted


def test_cuda_cache_is_released_immediately_before_backward():
    source = (Path(__file__).parents[1] / "train.py").read_text(encoding="utf-8")
    backward = source.index("sum(loss_all).backward()")
    cache_release = source.rfind("torch.cuda.empty_cache()", 0, backward)

    assert cache_release != -1
    assert source[cache_release:backward].strip() == "torch.cuda.empty_cache()"


def test_training_does_not_checkpoint_the_whole_encoder_before_backward():
    source = (Path(__file__).parents[1] / "train.py").read_text(encoding="utf-8")

    train_method = source.index("def train(self):")
    forward = source.index(") = self.multimodal_encoder(", train_method)
    backward = source.index("sum(loss_all).backward()", forward)

    assert "checkpoint_call(" not in source[train_method:backward]
    assert train_method < forward < backward


def test_mi_networks_are_checkpointed_after_expert_pair_selection():
    source = (Path(__file__).parents[1] / "model.py").read_text(encoding="utf-8")

    mi_forward = source.index("def forward(self, embeddings: dict):")
    pair_selection = source.index("z2 = embeddings[key][idx2]", mi_forward)
    checkpoint = source.index(
        "checkpoint_call(self.estimators[key], z1, z2)", pair_selection
    )

    assert mi_forward < pair_selection < checkpoint


def test_epoch_outputs_are_released_before_checkpoint_evaluation_and_next_epoch():
    source = (Path(__file__).parents[1] / "train.py").read_text(encoding="utf-8")

    estimator_step = source.index("self.mi_optimizer.step()")
    cleanup = source.index("del embeddings, estimator_loss, _", estimator_step)
    evaluation = source.index("if epoch != 0", estimator_step)

    assert estimator_step < cleanup < evaluation


def test_training_releases_loss_graphs_before_encoder_backward():
    source = (Path(__file__).parents[1] / "train.py").read_text(encoding="utf-8")

    loss_backward = source.index("sum(loss_all).backward()")
    release = source.index("del loss_all", loss_backward)
    encoder_backward = source.index("backward_through_gradient_bridge(", release)
    optimizer_step = source.index("self.optimizer.step()", encoder_backward)

    assert loss_backward < release < encoder_backward < optimizer_step
