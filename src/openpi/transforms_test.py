from fontTools.t1Lib import assertType1
import numpy as np
import pytest

import openpi.models.tokenizer as _tokenizer
import openpi.transforms as _transforms
import openpi.training.config as _config
from openpi.policies.leros2_policy import make_leros2_example

from scipy.spatial.transform import Rotation as R


def test_repack_transform():
    transform = _transforms.RepackTransform(
        structure={
            "a": {"b": "b/c"},
            "d": "e/f",
        }
    )
    item = {"b": {"c": 1}, "e": {"f": 2}}
    assert transform(item) == {"a": {"b": 1}, "d": 2}


def test_delta_actions():
    item = {"state": np.array([1, 2, 3]), "actions": np.array([[3, 4, 5], [5, 6, 7]])}

    transform = _transforms.DeltaActions(mask=[False, True])
    transformed = transform(item)

    assert np.all(transformed["state"] == np.array([1, 2, 3]))
    assert np.all(transformed["actions"] == np.array([[3, 2, 5], [5, 4, 7]]))


def test_delta_actions_noop():
    item = {"state": np.array([1, 2, 3]), "actions": np.array([[3, 4, 5], [5, 6, 7]])}

    # No-op when the mask is disabled.
    transform = _transforms.DeltaActions(mask=None)
    assert transform(item) is item

    # No-op when there are no actions in the input.
    del item["actions"]
    transform = _transforms.DeltaActions(mask=[True, False])
    assert transform(item) is item


def test_absolute_actions():
    item = {"state": np.array([1, 2, 3]), "actions": np.array([[3, 4, 5], [5, 6, 7]])}

    transform = _transforms.AbsoluteActions(mask=[False, True])
    transformed = transform(item)

    assert np.all(transformed["state"] == np.array([1, 2, 3]))
    assert np.all(transformed["actions"] == np.array([[3, 6, 5], [5, 8, 7]]))


def test_absolute_actions_noop():
    item = {"state": np.array([1, 2, 3]), "actions": np.array([[3, 4, 5], [5, 6, 7]])}

    # No-op when the mask is disabled.
    transform = _transforms.AbsoluteActions(mask=None)
    assert transform(item) is item

    # No-op when there are no actions in the input.
    del item["actions"]
    transform = _transforms.AbsoluteActions(mask=[True, False])
    assert transform(item) is item


def test_make_bool_mask():
    assert _transforms.make_bool_mask(2, -2, 2) == (True, True, False, False, True, True)
    assert _transforms.make_bool_mask(2, 0, 2) == (True, True, True, True)


def test_tokenize_prompt():
    tokenizer = _tokenizer.PaligemmaTokenizer(max_len=12)
    transform = _transforms.TokenizePrompt(tokenizer)

    data = transform({"prompt": "Hello, world!"})

    tok_prompt, tok_mask = tokenizer.tokenize("Hello, world!")
    assert np.allclose(tok_prompt, data["tokenized_prompt"])
    assert np.allclose(tok_mask, data["tokenized_prompt_mask"])


def test_tokenize_no_prompt():
    transform = _transforms.TokenizePrompt(_tokenizer.PaligemmaTokenizer())

    with pytest.raises(ValueError, match="Prompt is required"):
        transform({})


def test_transform_dict():
    # Rename and remove keys.
    input = {"a": {"b": 1, "c": 2}}
    output = _transforms.transform_dict({"a/b": "a/c", "a/c": None}, input)
    assert output == {"a": {"c": 1}}

    # Raises and error since the renamed key conflicts with an existing key.
    with pytest.raises(ValueError, match="Key 'a/c' already exists in output"):
        _transforms.transform_dict({"a/b": "a/c"}, input)

    # Full match is required and so nothing will be removed.
    input = {"a": {"b": 1, "c": 2}}
    output = _transforms.transform_dict({"a": None}, input)
    assert output == input

    # The regex matches the entire key and so the entire input will be removed.
    input = {"a": {"b": 1, "c": 2}}
    output = _transforms.transform_dict({"a.+": None}, input)
    assert output == {}

    # Replace keys using backreferences. All leaves named 'c' are replaced with 'd'.
    input = {"a": {"b": 1, "c": 1}, "b": {"c": 2}}
    output = _transforms.transform_dict({"(.+)/c": r"\1/d"}, input)
    assert output == {"a": {"b": 1, "d": 1}, "b": {"d": 2}}


def test_extract_prompt_from_task():
    transform = _transforms.PromptFromLeRobotTask({1: "Hello, world!"})

    data = transform({"task_index": 1})
    assert data["prompt"] == "Hello, world!"

    with pytest.raises(ValueError, match="task_index=2 not found in task mapping"):
        transform({"task_index": 2})


def _norm_quat(quat):
    x, y, z, w = quat
    if w < 0:
        x, y, z, w = -x, -y, -z, -w
    return np.array([x, y, z, w])


def test_quat_to_axis_angle():
    q2r = _transforms.QuatToAxisAngle(action_index=0)
    r2q = _transforms.AxisAngleToQuat(action_index=0)
    data = np.array([R.random().as_quat() for _ in range(100)])
    quat = r2q(q2r({"actions": data}))
    assert np.allclose(
        np.array([_norm_quat(q) for q in quat["actions"]]),
        np.array([_norm_quat(q) for q in data]),
    )


def test_quat_to_r6d():
    q2r = _transforms.QuatToR6D(action_index=0)
    r2q = _transforms.R6DToQuat(action_index=0)
    data = np.array([R.random().as_quat() for _ in range(100)])
    quat = r2q(q2r({"actions": data}))
    assert np.allclose(
        np.array([_norm_quat(q) for q in quat["actions"]]),
        np.array([_norm_quat(q) for q in data]),
    )


def test_scale_actions():
    transform = _transforms.ScaleActions(scale=np.array([2, 1, 0]))
    data = np.array(
        [
            [
                [1, 2, 3, 6],
                [4, 5, 6, 7],
            ]
        ]
    )
    scaled = transform({"actions": data})
    assert np.allclose(scaled["actions"], np.array([[[2, 2, 0, 6], [8, 5, 0, 7]]]))


def test_data_transform_e2e():
    input_example = make_leros2_example()
    train_config = _config.get_config("pi05_leros2")
    data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
    input_transform = _transforms.CompositeTransform(data_config.data_transforms.inputs)
    transformed_input = input_transform(input_example)
    assert transformed_input["state"].shape == (7,)  # Remove one state dimension (quat to axis angles)
    assert transformed_input["actions"].shape == (2, 7)  # Remove one action dimension
    assert transformed_input["state"][6] == input_example["state"][7]  # Keep the gripper invariant
    assert transformed_input["actions"][0, 6] == input_example["actions"][0, 7]  # Keep the gripper invariant
    assert transformed_input["actions"][1, 6] == input_example["actions"][1, 7]  # Keep the gripper invariant

    output_transform = _transforms.CompositeTransform(data_config.data_transforms.outputs)
    transformed_output = output_transform(transformed_input)
    assert transformed_output["actions"].shape == (2, 8)  # Add one action dimension
    assert transformed_output["actions"][0, 7] == input_example["actions"][0, 7]  # Keep the gripper invariant
    assert transformed_output["actions"][1, 7] == input_example["actions"][1, 7]  # Keep the gripper invariant
